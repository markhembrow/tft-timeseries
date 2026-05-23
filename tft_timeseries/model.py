"""TFT model components.

Implements the architecture described in Lim et al. (2021)
*Temporal Fusion Transformers for Interpretable Multi-Horizon Time-Series
Forecasting*.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# ════════════════════════════════════════════════════════════════════════════
# Gated Residual Network (GRN)  –  paper §3, Eq. 4
# ════════════════════════════════════════════════════════════════════════════

class GatedResidualNetwork(nn.Module):
    """Residual block with GLU-style gating and optional context infusion.

    Parameters
    ----------
    d:        input / output dimension
    d_ctx:    context feature dimension (``None`` => no context)
    d_hidden: hidden-layer dimension (default: ``4 * d``)
    dropout:  dropout probability
    """

    def __init__(
        self,
        d: int,
        d_ctx: Optional[int] = None,
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_h: int = d * 4 if d_hidden is None else d_hidden
        self._d = d

        self.lin1 = nn.Linear(d, d_h)
        self.lin2 = nn.Linear(d_h, d)

        self.lin_ctx: Optional[nn.Linear] = None
        if d_ctx is not None:
            self.lin_ctx = nn.Linear(d_ctx, d_h, bias=False)

        self.lin_gate = nn.Linear(d, d_h)
        self.lin_skip = nn.Linear(d, d)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    # ------------------------------------------------------------------
    def forward(
        self,
        xi: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """GRN forward pass.

        Parameters
        ----------
        xi:      ``(B, D)`` — primary input
        context: ``(B, D_ctx)`` — optional context vector infused before the gate

        Returns
        -------
        ``(B, D)`` — gated residual output
        """
        h = F.silu(self.lin1(xi))
        if context is not None and self.lin_ctx is not None:
            h = h + self.lin_ctx(context)

        g = F.silu(self.lin_gate(xi))
        h = self.drop1(h * g)
        h = self.lin2(h)
        skip = self.lin_skip(xi)
        return self.norm(skip + self.drop2(h))


# ════════════════════════════════════════════════════════════════════════════
# Variable Selection Network (VSN)  –  paper §3.1, Eq. 1-3, Fig. 2
# ════════════════════════════════════════════════════════════════════════════

class VariableSelectionNetwork(nn.Module):
    """Learns per-variable importance weights and produces a weighted embedding.

    Algorithm (Lim et al. 2021 §3.1):

    1. **Flatten & project**: ``xi`` of shape ``(B, S, D_in)`` is flattened
       to ``(B, S * D_in)`` and projected via a single ``Linear`` to
       ``(B, S * D_model)``, then reshaped to ``(B, S, D_model)``.
    2. **Per-variable GRN**: each slice ``[..., i, :]`` is transformed by an
       *independent* ``GatedResidualNetwork`` (weight-sharing is NOT applied).
    3. **Softmax weights**: the GRN outputs are collapsed to ``(B, S)`` via a
       dummy linear layer, then ``softmax`` is applied along ``dim=-1`` to
       produce importance weights ``w`` that sum to 1.
    4. **Weighted sum**: the scalar-weighted GRN outputs are summed across
       the variable dimension and projected to ``D_model``.

    Parameters
    ----------
    num_inputs: number of input feature variables *S*
    d_model:    output embedding dimension *D*
    dropout:    dropout rate
    d_hidden:   GRN hidden dim (default: ``4 * d_model``)
    """

    def __init__(
        self,
        num_inputs: int,
        d_model: int,
        dropout: float = 0.1,
        d_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._num_inputs = num_inputs
        self._d_model = d_model

        # Step 1 – single linear that covers all variables at once
        self.flatten = nn.Flatten(start_dim=1)          # (B, S, D_in) → (B, S*D_in)
        # input dim is inferred on first forward pass

        # Step 2 – independent GRN per variable
        d_h: int = d_model * 4 if d_hidden is None else d_hidden
        self.grn_list = nn.ModuleList(
            [GatedResidualNetwork(d_model, d_hidden=d_h, dropout=dropout) for _ in range(num_inputs)]
        )

        # Step 3 – weight head (maps each GRN output back to a scalar)
        self.weight_ctx = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=-1)

        # Step 4 – final projection (after weighted sum)
        self.project = nn.Linear(d_model, d_model)

    # ------------------------------------------------------------------
    def _build_flatten(self, d_in: int) -> None:
        """Lazily build the flatten→project layer once ``d_in`` is known."""
        if not hasattr(self, "_flat_proj"):
            self._flat_proj = nn.Linear(self._num_inputs * d_in, self._num_inputs * self._d_model)

    # ------------------------------------------------------------------
    def forward(
        self,
        xi: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute variable-weighted embedding.

        Parameters
        ----------
        xi:      ``(B, S, D_in)`` — raw or embedded per-variable features
        context: ``(B, D_ctx)`` — optional context embedding for importance
                 weighting (currently unused in the weight head, reserved
                 for future use)

        Returns
        -------
        ``(B, D_model)`` — weighted feature vector
        """
        B, S, D_in = xi.shape
        assert S == self._num_inputs, (
            f"Expected {self._num_inputs} variables, got {S}"
        )

        # Step 1 – lazy init projecting linear + flatten
        self._build_flatten(D_in)
        flat = self.flatten(xi)                          # (B, S*D_in)
        var_embeds = self._flat_proj(flat).view(B, S, self._d_model)  # (B, S, D_model)

        # Step 2 – independent GRN per variable
        grn_outputs: list[torch.Tensor] = []
        for i, grn in enumerate(self.grn_list):
            grn_outputs.append(grn(var_embeds[:, i, :]))  # (B, D_model) each
        grn_stack = torch.stack(grn_outputs, dim=1)        # (B, S, D_model)

        # Step 3 – softmax weights over variables  (B, S)
        # Collapse D_model → 1 via a linear head, then softmax
        w_logits = self.weight_ctx(grn_stack).squeeze(-1)  # (B, S)
        w = self.softmax(w_logits)
        self._last_w = w.detach()                          # expose for tests / inspection

        # Step 4 – weighted sum across variables, project to D_model
        weighted = grn_stack * w.unsqueeze(-1)             # (B, S, D_model)
        combined = weighted.sum(dim=1)                      # (B, D_model)
        return self.project(combined)


# ════════════════════════════════════════════════════════════════════════════
# StaticCovariateEncoder  –  paper §3.1, "Static inputs" branch, Fig. 2
# ════════════════════════════════════════════════════════════════════════════

class StaticCovariateEncoder(nn.Module):
    """Encodes all static features into a triple used by other TFT components.

    The static features path (paper §3.1, Fig. 2, "Static inputs" branch)
    takes the raw static vector **v** of shape ``(B, num_static)`` and
    passes it through three specialised branches:

    * **vs** – variable-selection weights (per-static-variable softmax),
      shape ``(B, num_static)``.
    * **ve** – enriched static embedding for the LSTM initial hidden /
      cell state, shape ``(B, d_model)``.
    * **vc** – static context vector that conditions subsequent GRN calls
      in the sequential feature path, shape ``(B, d_model)``.

    Parameters
    ----------
    num_static : number of static input variables  *N*
    d_model    : TFT hidden dimension  *D*
    d_hidden   : GRN hidden dimension (default ``4 * d_model``)
    dropout    : dropout probability
    """

    def __init__(
        self,
        num_static: int,
        d_model: int,
        d_hidden: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._num_static = num_static
        self._d_model = d_model

        d_h: int = d_model * 4 if d_hidden is None else d_hidden

        # Step 1 – embed each raw static variable scalar → d_model
        self._embed = nn.Linear(num_static, d_model)

        # Step 2 – shared low-level projection GRN  (B, D_model) → hidden
        self._hs_proj = GatedResidualNetwork(d_model, d_hidden=d_h, dropout=dropout)

        # Step 3 – three independent specialised GRN heads
        #   vs head: selects variables → (B, num_static)
        self._vs_lin = nn.Linear(d_model, d_model)  # pre-GRN
        self._vs_grn = GatedResidualNetwork(d_model, d_hidden=d_h, dropout=dropout)
        self._vs_out = nn.Linear(d_model, num_static)  # scalar per variable
        self._softmax = nn.Softmax(dim=-1)

        # ve head: enriched LSTM context → (B, d_model)
        self._ve_lin = nn.Linear(d_model, d_model)
        self._ve_grn = GatedResidualNetwork(d_model, d_hidden=d_h, dropout=dropout)

        # vc head: static context → (B, d_model)
        self._vc_lin = nn.Linear(d_model, d_model)
        self._vc_grn = GatedResidualNetwork(d_model, d_hidden=d_h, dropout=dropout)

    # ------------------------------------------------------------------
    def forward(self, xs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode a static feature vector into variable-selection triple.

        Parameters
        ----------
        xs : ``(B, num_static)`` — raw static features (one value per variable)

        Returns
        -------
        dict with keys:

        ``'vs'`` : ``(B, num_static)`` — variable selection softmax weights
        ``'ve'`` : ``(B, d_model)``  — enriched embedding for LSTM init state
        ``'vc'`` : ``(B, d_model)``  — static context vector for sequencial GRN
        """
        # Step 1 – embed each scalar variable to d_model
        hs = self._hs_proj(self._embed(xs))   # (B, d_model)

        # — vs branch: weights over static variables
        vs_logits = self._softmax(
            self._vs_out(self._vs_grn(self._vs_lin(hs)))
        )  # (B, num_static)

        # — ve branch: LSTM init enrichment
        ve = self._ve_grn(self._ve_lin(hs))   # (B, d_model)

        # — vc branch: static context
        vc = self._vc_grn(self._vc_lin(hs))   # (B, d_model)

        return {"vs": vs_logits, "ve": ve, "vc": vc}
