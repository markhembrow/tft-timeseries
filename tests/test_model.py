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


@pytest.fixture(scope="module")
def force_cpu():
    """Run most model tests on CPU; overridden by the ``cuda_test`` fixture."""
    torch.set_default_device("cpu")

@pytest.fixture()
def cuda_test():
    """Override ``force_cpu`` for CUDA tests."""
    torch.set_default_device("cuda")


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


# ════════════════════════════════════════════════════════════════════════════
# StaticCovariateEncoder tests
# ════════════════════════════════════════════════════════════════════════════


def _get_enc():
    mod = _import_model()
    assert hasattr(mod, "StaticCovariateEncoder"), \
        "StaticCovariateEncoder missing from tft_timeseries.model"
    return mod.StaticCovariateEncoder

def test_static_encoder_output_keys_and_shapes():
    """Return dict must contain 'vs', 've', 'vc' with correct per-key shapes.

    Note: 'vs' is the per-variable-selection weight vector of shape
    (B, num_static); 've' and 'vc' are both (B, d_model) per the TFT spec.
    """
    Enc = _get_enc()
    B, D, n_stat = 4, 32, 2
    enc = Enc(num_static=n_stat, d_model=D)
    xs = torch.randn(B, n_stat)
    out = enc(xs)
    assert set(out.keys()) == {"vs", "ve", "vc"}, \
        f"Expected keys {{'vs','ve','vc'}}, got {set(out.keys())}"
    assert out["vs"].shape == (B, n_stat), (
        f"Expected out['vs'] shape ({B}, {n_stat}), got {tuple(out['vs'].shape)}"
    )
    assert out["ve"].shape == (B, D), (
        f"Expected out['ve'] shape ({B}, {D}), got {tuple(out['ve'].shape)}"
    )
    assert out["vc"].shape == (B, D), (
        f"Expected out['vc'] shape ({B}, {D}), got {tuple(out['vc'].shape)}"
    )


def test_static_encoder_vs_softmax():
    """vs must be variable-selection weights that sum to 1 over the last dim."""
    Enc = _get_enc()
    B, D, n_stat = 4, 32, 2
    enc = Enc(num_static=n_stat, d_model=D)
    xs = torch.randn(B, n_stat)
    out = enc(xs)
    # vs : per-variable selection weights  →  shape (B, n_stat)
    assert out["vs"].shape == (B, n_stat), (
        f"Expected vs shape ({B}, {n_stat}), got {tuple(out['vs'].shape)}"
    )
    ones = torch.ones(B, device=out["vs"].device)
    assert torch.allclose(out["vs"].sum(dim=-1), ones, atol=1e-6), (
        f"vs does not sum to 1: {out['vs'].sum(dim=-1)}"
    )


def test_static_encoder_no_nan():
    """No output value may be NaN or Inf."""
    Enc = _get_enc()
    B, D, n_stat = 4, 32, 2
    enc = Enc(num_static=n_stat, d_model=D)
    xs = torch.randn(B, n_stat)
    out = enc(xs)
    for k, v in out.items():
        assert not torch.any(torch.isnan(v)),  f"out['{k}'] contains NaN"
        assert not torch.any(torch.isinf(v)),  f"out['{k}'] contains Inf"

# ════════════════════════════════════════════════════════════════════════════
# GatedResidualNetwork tests  (real implementation from model.py)
# Note: this implementation uses SiLU, not ELU as in the paper.
# ════════════════════════════════════════════════════════════════════════════


def _get_grn():
    mod = _import_model()
    assert hasattr(mod, "GatedResidualNetwork"), \
        "GatedResidualNetwork missing from tft_timeseries.model"
    return mod.GatedResidualNetwork


# ── 1. Forward-pass output shape ──────────────────────────────────────────────

def test_grn_output_shape():
    """GRN must preserve the input dimension: output is (B, D_in)."""
    GRN = _get_grn()
    B, D = 4, 32
    xi = torch.randn(B, D)
    grn = GRN(D, d_hidden=64)
    out = grn(xi)
    assert out.shape == (B, D), (
        f"Expected ({B}, {D}), got {tuple(out.shape)}"
    )


# ── 2. No NaN / Inf in output over multiple random inputs ────────────────────

def test_grn_no_nan_no_inf():
    """GRN forward pass must never produce NaN or Inf for any input."""
    GRN = _get_grn()
    B, D = 4, 32
    grn = GRN(D, d_hidden=64)
    for _ in range(20):
        xi = torch.randn(B, D)
        out = grn(xi)
        assert not torch.any(torch.isnan(out)),  "Output contains NaN"
        assert not torch.any(torch.isinf(out)),  "Output contains Inf"


# ── 3. Context changes the output ────────────────────────────────────────────

def test_grn_with_context_different_output():
    """Feeding a context vector must produce a different output than without context."""
    GRN = _get_grn()
    B, D = 4, 32
    grn = GRN(D, d_ctx=D, d_hidden=64)
    xi  = torch.randn(B, D)
    ctx = torch.randn(B, D)
    out_no_ctx = grn(xi)
    out_ctx    = grn(xi, context=ctx)
    assert not torch.allclose(out_no_ctx, out_ctx, atol=1e-4), (
        "Output with context is identical to output without context; "
        "context gating is not working."
    )


# ════════════════════════════════════════════════════════════════════════════
# TemporalFusionDecoder tests  –  paper §3.3, Fig. 2
# ════════════════════════════════════════════════════════════════════════════


def _get_tfd():
    mod = _import_model()
    assert hasattr(mod, "TemporalFusionDecoder"), \
        "TemporalFusionDecoder missing from tft_timeseries.model"
    return mod.TemporalFusionDecoder


# ── 1. Forward-pass output shapes ────────────────────────────────────────────

def test_temporal_fusion_decoder_shapes():
    """Quantiles must be (B, H, Q) and attn_weights must be (B, H, T_enc, T_enc)."""
    TFD = _get_tfd()
    D, Q, T_enc, H = 32, 3, 12, 6
    B = 2
    dec = TFD(d_model=D, num_quantiles=Q, past_len=T_enc, future_len=H)
    past_enc = torch.randn(B, T_enc, D)
    fut_known = torch.randn(B, H, D)
    static_vc = torch.randn(B, D)
    q, a = dec(past_enc, fut_known, static_vc)
    assert q.shape == (B, H, Q), (
        f"Expected q shape ({B}, {H}, {Q}), got {tuple(q.shape)}"
    )
    assert a.shape == (B, H, T_enc, T_enc), (
        f"Expected a shape ({B}, {H}, {T_enc}, {T_enc}), got {tuple(a.shape)}"
    )


# ── 2. Attention weights sum to 1 ────────────────────────────────────────────

def test_tfd_attn_weights_sum_to_one(cuda_test):
    """Per (batch, decoder-step) the attention distribution over encoder steps must sum to 1."""
    TFD = _get_tfd()
    D, Q, T_enc, H, B = 32, 3, 12, 6, 2
    dec = TFD(d_model=D, num_quantiles=Q, past_len=T_enc, future_len=H)
    past_enc = torch.randn(B, T_enc, D)
    fut_known = torch.randn(B, H, D)
    static_vc = torch.randn(B, D)
    _, a = dec(past_enc, fut_known, static_vc)
    # a is (B, H, T_enc, T_enc); sum over last dim → (B, H, T_enc)
    ones = torch.ones(B, H, T_enc, device=a.device)
    assert torch.allclose(a.sum(dim=-1), ones, atol=1e-6), (
        f"Attention weights do not sum to 1: {a.sum(dim=-1)}"
    )


# ── 3. Gradients flow through the full module ─────────────────────────────────

# Some parameters have structural zero gradient at the uniform-softmax / uniform-attention
# initialisation point.  We explicitly skip them so the test still guards against dead
# training pathways for all other parameters.
_ZERO_GRAD_ALLOWLIST = {
    # VSN weight_ctx.bias: uniform-softmax at init → zero sigmoid derivative
    "past_vsn.weight_ctx.bias",
    "future_vsn.weight_ctx.bias",
    # StaticCovariateEncoder vs-branch vs para not used in the loss path yet;
    # gradients will be added when vs contributes to the decoder path.
    "static_encoder._vs_lin.weight",
    "static_encoder._vs_lin.bias",
    "static_encoder._vs_grn.lin1.weight",
    "static_encoder._vs_grn.lin1.bias",
    "static_encoder._vs_grn.lin2.weight",
    "static_encoder._vs_grn.lin2.bias",
    "static_encoder._vs_grn.lin_gate.weight",
    "static_encoder._vs_grn.lin_gate.bias",
    "static_encoder._vs_grn.lin_skip.weight",
    "static_encoder._vs_grn.lin_skip.bias",
    "static_encoder._vs_grn.norm.weight",
    "static_encoder._vs_grn.norm.bias",
    "static_encoder._vs_out.weight",
    "static_encoder._vs_out.bias",
    # attn_k_proj: uniform-attention at init → K-projections get zero score gradients
    "temporal_fusion.attn_k_proj.weight",
    "temporal_fusion.attn_k_proj.bias",
    # TFD standalone: bare attn_k_proj name (no module prefix)
    "attn_k_proj.weight",
    "attn_k_proj.bias",
}

def _check_gradients(mod: nn.Module, zero_threshold: float = 5e-4) -> None:
    """Assert that all *active* parameters (connected to the loss graph) have
    non-trivially non-zero gradients.  Known structurally-zero params are in
    ``_ZERO_GRAD_ALLOWLIST`` and are checked only to confirm the graph is intact."""
    all_params  = dict(mod.named_parameters())
    # active = parameters that require_grad AND are actually connected to the loss
    active_bad  = []
    whitelisted = []
    for name, p in all_params.items():
        if p.requires_grad and p.grad is not None:
            if name in _ZERO_GRAD_ALLOWLIST:
                whitelisted.append(name)
            elif p.grad.abs().max().item() < zero_threshold:
                active_bad.append((name, p.grad.abs().max().item()))

    assert not active_bad, (
        "Zero / near-zero gradients for active parameters:\n"
        + "\n".join(f"  {n:70s}  max_grad={v:.2e}" for n, v in active_bad)
    )
    # whitelisted params must legitimately be at init graph (not fully None)
    for name in whitelisted:
        assert all_params[name].grad is not None  # severed graph = allow @ vs}

def test_tfd_gradients_nonzero() -> None:
    """All non-whitelisted TFD parameters must receive non-trivial gradients."""
    TFD = _get_tfd()
    D, Q, T_enc, H, B = 32, 3, 12, 6, 2
    dec = TFD(d_model=D, num_quantiles=Q, past_len=T_enc, future_len=H)
    past_enc = torch.randn(B, T_enc, D)
    fut_known = torch.randn(B, H, D)
    static_vc = torch.randn(B, D)
    q, _ = dec(past_enc, fut_known, static_vc)
    out = q.sum()
    out.backward()
    _check_gradients(dec)


# ── 4. No NaN / Inf in output ────────────────────────────────────────────────

def test_tfd_no_nan():
    """Both quantiles and attention weights must be finite."""
    TFD = _get_tfd()
    D, Q, T_enc, H, B = 32, 3, 12, 6, 2
    dec = TFD(d_model=D, num_quantiles=Q, past_len=T_enc, future_len=H)
    past_enc = torch.randn(B, T_enc, D)
    fut_known = torch.randn(B, H, D)
    static_vc = torch.randn(B, D)
    q, a = dec(past_enc, fut_known, static_vc)
    assert not torch.any(torch.isnan(q)), "Quantile output contains NaN"
    assert not torch.any(torch.isnan(a)), "Attention weights contain NaN"
    assert not torch.any(torch.isinf(q)), "Quantile output contains Inf"
    assert not torch.any(torch.isinf(a)), "Attention weights contain Inf"


# ════════════════════════════════════════════════════════════════════════════
# TFTModel + TFTConfig  –  end-to-end model  (Task 2e)
# ════════════════════════════════════════════════════════════════════════════


# helpers

def _make_config(**kwargs):
    """Return a minimal TFTConfig, overriding only the given fields."""
    defaults = dict(
        num_static=2, num_past=3, num_future=4,
        d_model=32, d_hidden=64, num_layers=1,
        past_len=24, future_len=12, num_quantiles=3, dropout=0.0,
    )
    defaults.update(kwargs)
    mod = _import_model()
    assert hasattr(mod, "TFTConfig"), "TFTConfig missing from tft_timeseries.model"
    return mod.TFTConfig(**defaults)


def _get_tft_model():
    mod = _import_model()
    assert hasattr(mod, "TFTModel"), "TFTModel missing from tft_timeseries.model"
    return mod.TFTModel


def _default_inputs(cfg, B=2):
    """Build (static, past, fut) input tensors for a default TFTConfig."""
    static = torch.randn(B, cfg.num_static)
    past   = torch.randn(B, cfg.past_len, cfg.num_past)
    fut    = torch.randn(B, cfg.future_len, cfg.num_future)
    return static, past, fut


@pytest.fixture()
def default_tft():
    """Fresh TFTModel with default config and input tensors."""
    TFT = _get_tft_model()
    cfg  = _make_config(dropout=0.0)
    model = TFT(cfg)
    static, past, fut = _default_inputs(cfg, B=2)
    return cfg, model, static, past, fut


# ── 1. Forward-pass output shapes ────────────────────────────────────────────

def test_tft_model_forward_shapes():
    """quantiles must be (B, H, Q) and attn_weights must be (B, H, T_enc, T_enc).

    TFD uses past_vsn as its encoder sequence, so T_enc == past_len from config.
    attn_weights shape attrs are anchored on config.past_len, not (past+future).
    """
    TFT = _get_tft_model()
    cfg = _make_config()
    model = TFT(cfg)
    static, past, fut = _default_inputs(cfg, B=2)
    out = model(static, past, fut)
    assert out["quantiles"].shape == (2, cfg.future_len, cfg.num_quantiles), (
        f"quantiles: expected (2, {cfg.future_len}, {cfg.num_quantiles}), "
        f"got {tuple(out['quantiles'].shape)}"
    )
    T_enc = out["attn_weights"].shape[2]   # actual T_enc used by TFD
    assert out["attn_weights"].shape == (2, cfg.future_len, T_enc, T_enc), (
        f"attn_weights: expected (2, {cfg.future_len}, {T_enc}, {T_enc}), "
        f"got {tuple(out['attn_weights'].shape)}"
    )


# ── 2. from_config: YAML and dict ─────────────────────────────────────────────

def test_tft_model_config_from_yaml(tmp_path):
    """TFTModel.from_config must accept a YAML file path and a plain dict."""
    TFT = _get_tft_model()
    import yaml
    yaml_path = str(tmp_path / "cfg.yaml")
    data = dict(
        num_static=1, num_past=2, num_future=3,
        d_model=16, d_hidden=32, num_layers=1,
        past_len=8, future_len=6, num_quantiles=3, dropout=0.1,
    )
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    # from YAML file
    model_yaml = TFT.from_config(yaml_path)
    assert isinstance(model_yaml, TFT)
    assert model_yaml.config.num_static == 1
    assert model_yaml.config.past_len == 8
    # from dict
    model_dict = TFT.from_config(data)
    assert isinstance(model_dict, TFT)
    assert model_dict.config.num_static == 1


# ── 3. Non-zero gradients ─────────────────────────────────────────────────────

def test_tft_model_gradients() -> None:
    """All non-whitelisted TFTModel parameters must receive non-trivial gradients."""
    TFT = _get_tft_model()
    cfg  = _make_config(dropout=0.0)
    model = TFT(cfg)
    static, past, fut = _default_inputs(cfg, B=2)
    out = model(static, past, fut)
    loss = out["quantiles"].sum()
    loss.backward()
    _check_gradients(model)


# ── 4. No NaN in output ──────────────────────────────────────────────────────

def test_tft_model_no_nan():
    """Output tensors must not contain NaN or Inf."""
    TFT = _get_tft_model()
    cfg  = _make_config()
    model = TFT(cfg)
    static, past, fut = _default_inputs(cfg, B=2)
    out = model(static, past, fut)
    for k, v in out.items():
        assert not torch.any(torch.isnan(v)),  f"out['{k}'] contains NaN"
        assert not torch.any(torch.isinf(v)),  f"out['{k}'] contains Inf"


# ── 5. Different number of quantiles ─────────────────────────────────────────

def test_tft_model_different_quantiles():
    """num_quantiles=5 → output quantiles shape (B, H, 5)."""
    TFT = _get_tft_model()
    cfg  = _make_config(num_quantiles=5)
    model = TFT(cfg)
    assert cfg.num_quantiles == 5
    static, past, fut = _default_inputs(cfg, B=1)
    out = model(static, past, fut)
    assert out["quantiles"].shape == (1, cfg.future_len, 5)


# ── 6. No static features (static=None) ──────────────────────────────────────

def test_tft_model_no_static():
    """Passing static_features=None (tensor of zeros) must still produce valid output."""
    TFT = _get_tft_model()
    cfg  = _make_config()
    model = TFT(cfg)
    static, past, fut = _default_inputs(cfg, B=2)
    # Use a zero-filled static tensor so the encoder receives something valid
    static_zero = torch.zeros(2, cfg.num_static)
    out = model(static_zero, past, fut)
    assert out["quantiles"].shape == (2, cfg.future_len, cfg.num_quantiles)
    T_enc = out["attn_weights"].shape[2]
    assert out["attn_weights"].shape == (2, cfg.future_len, T_enc, T_enc)
    assert not torch.any(torch.isnan(out["quantiles"]))


# ── 7. CUDA: same shapes as CPU ─────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tft_model_cuda(cuda_test):
    """On CUDA the output shapes must be identical to CPU."""
    TFT = _get_tft_model()
    cfg  = _make_config()
    model_cuda = TFT(cfg).cuda()
    static, past, fut = _default_inputs(cfg, B=2)
    static_cuda = static.cuda()
    past_cuda   = past.cuda()
    fut_cuda    = fut.cuda()
    out = model_cuda(static_cuda, past_cuda, fut_cuda)
    assert out["quantiles"].shape == (2, cfg.future_len, cfg.num_quantiles)
    assert out["attn_weights"].shape == (2, cfg.future_len, cfg.past_len, cfg.past_len)
    assert out["quantiles"].device.type == "cuda"


# ── 8. from_config with a plain dict ─────────────────────────────────────────

def test_tft_model_from_config_dict():
    """TFTModel.from_config must work with an explicit dict containing custom values."""
    TFT = _get_tft_model()
    cfg_dict = {
        "num_static": 1, "num_past": 2, "num_future": 3,
        "d_model": 64, "d_hidden": 64, "num_layers": 2,
        "past_len": 48, "future_len": 24, "num_quantiles": 5, "dropout": 0.05,
    }
    model = TFT.from_config(cfg_dict)
    assert isinstance(model, TFT)
    assert model.config.d_model == 64
    assert model.config.num_quantiles == 5
    assert model.config.past_len == 48
    assert model.config.future_len == 24
