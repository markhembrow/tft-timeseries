"""Losses for the Temporal Fusion Transformer.

All losses follow the paper (Lim et al. 2021 §3.4).
"""

from __future__ import annotations

import torch
from torch import nn


def quantile_loss(y_hat: torch.Tensor, y: torch.Tensor,
                  quantiles: list[float]) -> torch.Tensor:
    """Pinball / quantile regression loss.

    Parameters
    ----------
    y_hat     : ``(B, H, Q)`` predicted quantile levels.
    y         : ``(B, H)``     ground-truth target.
    quantiles : list of quantile levels in ascending order, e.g. ``[0.1, 0.5, 0.9]``.

    Returns
    -------
    scalar mean pinball loss over all batch × horizon × quantile entries.
    """
    q_tensor = torch.tensor(quantiles, device=y_hat.device, dtype=y_hat.dtype)
    # (Q,) → (1, 1, Q) for broadcasting over (B, H, Q)
    q = q_tensor.view(1, 1, -1)

    residual = y.unsqueeze(-1) - y_hat          # (B, H, Q)
    pb = torch.where(residual >= 0,
                      q * residual,
                     (1.0 - q) * (-residual))
    return pb.mean()


class QuantileLoss(nn.Module):
    """Module wrapper :class:`quantile_loss`."""

    def __init__(self, quantiles: list[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return quantile_loss(y_hat, y, self.quantiles)
