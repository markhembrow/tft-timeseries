"""Tests for tft_timeseries.model — VariableSelectionNetwork and GatedResidualNetwork."""
from __future__ import annotations

import importlib
import sys
import os
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn


# ── helpers ────────────────────────────────────────────────────────────────

def _make_grn() -> nn.Module:
    """Minimal GatedResidualNetwork (pure Python, no file dependency)."""

    class _Grn(nn.Module):
        def __init__(self, d: int, d_ctx: int | None = None,
                     d_hidden: int | None = None, dropout: float = 0.1) -> None:
            super().__init__()
            d_h = d * 4 if d_hidden is None else d_hidden
            self.lin1 = nn.Linear(d, d_h)
            self.lin2 = nn.Linear(d_h, d)
            self.lin_ctx = nn.Linear(d_ctx, d_h, bias=False) if d_ctx else None
            self.lin_gate = nn.Linear(d, d_h)
            self.lin_skip = nn.Linear(d, d)
            self.drop1 = nn.Dropout(dropout)
            self.drop2 = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(d)

        def forward(self, xi: torch.Tensor, context: torch.Tensor | None = None):
            h = F.silu(self.lin1(xi))
            if context is not None and self.lin_ctx is not None:
                h = h + self.lin_ctx(context)
            g = F.silu(self.lin_gate(xi))
            h = self.drop1(h * g)
            h = self.lin2(h)
            skip = self.lin_skip(xi)
            return self.norm(skip + self.drop2(h))

    return _Grn()


@pytest.fixture(autouse=True, scope="module")
def force_cpu():
    """Run every model test on CPU regardless of available GPU."""
    torch.set_default_device("cpu")


# Skip entire module if torch / cuda unexpected
collect_ok = True
try:
    import torch  # noqa: F401
except Exception:
    collect_ok = False

pytestmark = pytest.mark.skipif(not collect_ok, reason="torch not available")


# ════════════════════════════════════════════════════════════════════════════
# VSN tests  (real implementations — live-import from model.py)
# ════════════════════════════════════════════════════════════════════════════


def _import_model():
    """Dynamically load tft_timeseries.model, patching the module into sys.modules."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(pkg_dir)          # /home/mctouch/code/tft-timeseries
    sys.path.insert(0, project_root)

    pkg_name = "tft_timeseries"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [os.path.join(project_root, pkg_name)]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    mod_name = f"{pkg_name}.model"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])

    model_file = os.path.join(project_root, pkg_name, "model.py")
    spec = importlib.util.spec_from_file_location(mod_name, model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _get_vsn():
    mod = _import_model()                       # raises ImportError if model.py absent/invalid
    assert hasattr(mod, "VariableSelectionNetwork"), "VariableSelectionNetwork missing from tft_timeseries.model"
    assert hasattr(mod, "GatedResidualNetwork"),     "GatedResidualNetwork missing from tft_timeseries.model"
    return mod.VariableSelectionNetwork


# ── 1. Forward-pass output shape ────────────────────────────────────────────

def test_vsn_output_shape():
    """Output must be (B, D_model) regardless of (B, S, D_in) input."""
    VSN = _get_vsn()
    B, S, D_in, D_model = 4, 3, 8, 32
    xi = torch.randn(B, S, D_in)
    vsn = VSN(num_inputs=S, d_model=D_model, d_hidden=D_model * 4)
    out = vsn(xi)
    assert out.shape == (B, D_model), (
        f"Expected ({B}, {D_model}), got {tuple(out.shape)}"
    )


# ── 2. Attention weights sum to 1 ────────────────────────────────────────────

def test_vsn_weights_sum_to_one():
    """Per-sample softmax over S variables should sum to 1."""
    VSN = _get_vsn()
    B, S, D_in, D_model = 2, 4, 16, 32
    xi = torch.randn(B, S, D_in)
    vsn = VSN(num_inputs=S, d_model=D_model)
    # VSN stores the last weight vector as vsn._last_w  (attached in forward)
    out = vsn(xi)
    assert hasattr(vsn, "_last_w"), "VSN must store last computed weight vector in self._last_w"
    w = vsn._last_w
    assert w.shape == (B, S), f"Expected weights shape ({B}, {S}), got {tuple(w.shape)}"
    ones = torch.ones(B, device=w.device)
    assert torch.allclose(w.sum(dim=-1), ones, atol=1e-6), (
        f"Weights do not sum to 1: {w.sum(dim=-1)}"
    )


# ── 3. Gradients are non-zero post-backward ──────────────────────────────────

def test_vsn_gradients_nonzero():
    """All leaf parameters with non-zero std must receive non-zero gradients."""
    VSN = _get_vsn()
    B, S, D_in, D_model = 4, 3, 8, 32
    xi = torch.randn(B, S, D_in, requires_grad=False)
    vsn = VSN(num_inputs=S, d_model=D_model)
    out = vsn(xi)
    loss = out.sum()
    loss.backward()
    for name, p in vsn.named_parameters():
        assert p.requires_grad, f"Parameter {name} must require_grad=True"
        assert p.grad is not None,          f"Parameter {name} has no grad"
        assert not torch.allclose(p.grad, torch.zeros_like(p.grad)), (
            f"Zero gradient for parameter {name}"
        )


# ── 4. No NaN / Inf in output ───────────────────────────────────────────────

def test_vsn_no_nan_no_inf():
    """Forward pass must produce finite values for any input."""
    VSN = _get_vsn()
    B, S, D_in, D_model = 4, 3, 8, 32
    xi = torch.randn(B, S, D_in)
    vsn = VSN(num_inputs=S, d_model=D_model)
    out = vsn(xi)
    assert not torch.any(torch.isnan(out)),  "Output contains NaN"
    assert not torch.any(torch.isinf(out)),  "Output contains Inf"


# ── 5. Edge-case: S=1 (single variable) ─────────────────────────────────────

def test_vsn_single_variable():
    """With S=1 the output is just the GRN-transformed variable (weight = 1)."""
    VSN = _get_vsn()
    B, S, D_in, D_model = 2, 1, 8, 32
    xi = torch.randn(B, S, D_in)
    vsn = VSN(num_inputs=1, d_model=D_model)
    out = vsn(xi)
    assert out.shape == (B, D_model)
    assert torch.allclose(vsn._last_w, torch.ones(B, 1), atol=1e-6)


# ── 6. Context-aware weighting ──────────────────────────────────────────────

def test_vsn_with_context():
    """Providing a context vector should not crash and still sum weights to 1."""
    VSN = _get_vsn()
    B, S, D_in, D_model, D_ctx = 2, 3, 8, 32, 16
    xi      = torch.randn(B, S, D_in)
    context = torch.randn(B, D_ctx)
    vsn = VSN(num_inputs=S, d_model=D_model, d_hidden=D_model * 4)
    out = vsn(xi, context=context)
    assert out.shape == (B, D_model)
    assert torch.allclose(vsn._last_w.sum(dim=-1), torch.ones(B), atol=1e-6)
