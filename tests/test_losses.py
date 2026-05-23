"""Tests for tft_timeseries.losses — quantile (pinball) loss."""

from __future__ import annotations

import sys, os
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tft_timeseries.losses import quantile_loss, QuantileLoss


# ── helpers ────────────────────────────────────────────────────────────────────

def _pinball(y_true, y_pred, q):
    """Manual pinball loss for reference."""
    r = y_true - y_pred
    return q * r if r >= 0 else (1 - q) * (-r)


# ── 1. Shape check ─────────────────────────────────────────────────────────────

def test_quantile_loss_shape():
    B, H, Q = 4, 6, 3
    y_hat = torch.zeros(B, H, Q)
    y     = torch.randn(B, H)
    loss  = quantile_loss(y_hat, y, [0.1, 0.5, 0.9])
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"


# ── 2. Perfect prediction: pinball = 0 ⟹ mean × q · median ────────────────────

@pytest.mark.parametrize("q", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_quantile_loss_zero_at_perfect_prediction(q):
    B, H = 3, 4
    y_hat = torch.empty(B, H, 1)
    y_hat[:, :, 0] = 7.0
    y = torch.full((B, H), 7.0)
    loss = quantile_loss(y_hat, y, [q])
    assert loss.item() < 1e-10, f"Loss at perfect prediction: {loss.item()}"


# ── 3. Asymmetric penalty ──────────────────────────────────────────────────────

def test_quantile_loss_asymmetry():
    B, H = 3, 4
    q  = 0.1          # conservative — penalises under-forecast more
    # y_hat = 5, y = 10  → under-forecast by 5 → penalty = q * 5
    # y_hat = 5, y =  0  → over-forecast by 5 → penalty = (1-q) * 5
    y_hat = torch.full((B, H, 1), 5.0)
    y_under  = torch.full((B, H), 10.0)
    y_over   = torch.full((B, H),  0.0)

    loss_under = quantile_loss(y_hat, y_under, [q])
    loss_over  = quantile_loss(y_hat, y_over,  [q])
    ratio    = (loss_under / loss_over).item()
    expected = q / (1.0 - q)
    assert abs(ratio - expected) < 0.01, (
        f"Under={loss_under.item():.4f}, Over={loss_over.item():.4f}, "
        f"ratio-cur={ratio:.4f}, expected={expected:.4f}"
    )


# ── 4. QuantileLoss module wrapper ─────────────────────────────────────────────

def test_quantile_loss_module():
    B, H = 2, 4
    loss_fn = QuantileLoss([0.1, 0.5, 0.9])
    y_hat = torch.ones(B, H, 3) * 5.0
    y = torch.full((B, H), 7.0)
    loss = loss_fn(y_hat, y)
    assert loss.shape == ()


# ── 5. Multiple quantiles: loss = mean over Q ──────────────────────────────────

def test_quantile_loss_multiple_quantiles():
    B, H = 1, 3
    y_hat = torch.randn(B, H, 5)
    y = torch.randn(B, H)
    # should not crash
    loss = quantile_loss(y_hat, y, [0.05, 0.1, 0.5, 0.9, 0.95])
    assert loss >= 0.0, "Quantile loss must be non-negative"
