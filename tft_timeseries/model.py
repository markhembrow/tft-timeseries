"""TFT model components.

Implements the architecture described in Lim et al. (2021)
*Temporal Fusion Transformers for Interpretable Multi-Horizon Time-Series
Forecasting*.
"""
from __future__ import annotations

from dataclasses import dataclass
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


# ════════════════════════════════════════════════════════════════════════════
# TemporalFusionDecoder  –  paper §3.3, Fig. 2  (core of the TFT)
# ════════════════════════════════════════════════════════════════════════════

class TemporalFusionDecoder(nn.Module):
    """LSTM encoder-decoder with static enrichment and temporal attention fusion.

    This is the **core** of the Temporal Fusion Transformer (Lim et al. 2021,
    §3.3–3.4, Fig. 2).  It combines three sub-mechanisms:

    1. **Past encoder** (LSTM) – reads ``past_encoded`` to produce a
       contextualised sequence ``(B, T_enc, D)``.
    2. **Future decoder** (LSTM) – processes ``future_known`` autoregressively
       to produce decoder states for each future step.
    3. **Temporal attention fusion** – at every decoder step h, attends over
       *all* encoder output steps to build a fused representation
       ``context_decoder_h``.
    4. **Static enrichment** – the ``static_vc`` vector is transformed by a
       GRN; the resulting gating signal modulates the concatenated
       ``[decoder_h; context_decoder_h]`` before the final GRN + quantile
       head.

    Parameters
    ----------
    d_model:       TFT model dimension *D* (shared across all sub-components).
    num_quantiles: number of quantile predictions *Q* per forecast step.
    past_len:      encoder sequence length *T_enc*.
    future_len:    decoder sequence length *H*.
    dropout:       dropout applied to every LSTM / GRN sub-layer.
    num_layers:    number of LSTM layers (shared for encoder & decoder).
    """

    def __init__(
        self,
        d_model: int,
        num_quantiles: int,
        past_len: int,
        future_len: int,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self._d = d_model
        self._h = future_len
        self._q = num_quantiles
        self._t_enc = past_len

        # ── 1. Encoder LSTM ───────────────────────────────────────────────
        self.encoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 2. Decoder LSTM ───────────────────────────────────────────────
        # Input dim must account for the dim of future_known; we project up.
        self.future_proj = nn.Linear(d_model, d_model)
        self.decoder_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # ── 3. Fuse decoder + attention context ────────────────────────────
        # combined: (B, H, 2D)  [decoder_state | attention_context]
        # followed by GRN (with static_vc as secondary context) → (B, H, D)
        self.post_attn_grn = GatedResidualNetwork(
            d=2 * d_model, d_ctx=d_model, d_hidden=d_model * 4, dropout=dropout
        )
        self._post_attn_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # ── 4. Attention projections ────────────────────────────────────────
        # Q = decoder state directly (no learned projection; preserves gradient
        # to the LSTM).  K and V are learned projections of the encoder output.
        # This matches the original TFT paper (Lim et al. 2021 §3.3).
        self.attn_k_proj = nn.Linear(d_model, d_model)
        self.attn_v_proj = nn.Linear(d_model, d_model)

        # ── 6. Quantile head ──────────────────────────────────────────────
        self.quantile_head = nn.Linear(d_model, num_quantiles)

    # ------------------------------------------------------------------
    def forward(
        self,
        past_encoded: torch.Tensor,    # (B, T_enc, D)
        future_known: torch.Tensor,    # (B, H, D_known)  — will be projected to D
        static_vc:    torch.Tensor,    # (B, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the TFT decoder and return quantile predictions and attention maps.

        Parameters
        ----------
        past_encoded: ``(B, T_enc, D)`` encoder output from the historic path.
        future_known: ``(B, H, D_known)`` known future regressors.
        static_vc:    ``(B, D)`` static context vector produced by
                      ``StaticCovariateEncoder``.

        Returns
        -------
        quantiles : ``(B, H, Q)`` — one quantile level per forecast step.
        attn_weights : ``(B, H, T_enc, T_enc)`` — decoder self-attention weights
                       broadcast over all encoder steps.
        """
        B, T_enc, D = past_encoded.shape

        # ── 1. Encoder LSTM ──────────────────────────────────────────────
        enc_out, _ = self.encoder_lstm(past_encoded)   # (B, T_enc, D)
        enc_out = enc_out.contiguous()

        # ── 2. Decoder LSTM ──────────────────────────────────────────────
        fut_proj = F.silu(self.future_proj(future_known))  # (B, H, D)
        dec_out, _ = self.decoder_lstm(fut_proj)           # (B, H, D)
        dec_out = dec_out.contiguous()

        # ── 4. Temporal attention per decoder step ────────────────────────
        # Use 3D tensors throughout; expand to (B,H,T,T) for attn_weights.
        # attn_scores[B, h, k] = decoder-step-h  ·  encoder-step-k
        # Q = raw decoder state — no learned projection (Lim et al. §3.3)
        Q = dec_out                            # (B, H, D)
        K = self.attn_k_proj(enc_out)          # (B, T_enc, D)
        V = self.attn_v_proj(enc_out)          # (B, T_enc, D)

        # (B, H, T_enc)  – standard multi-head dot-product (single head here)
        attn_scores = torch.einsum("bhd,btd->bht", Q, K) / (D ** 0.5)

        # Expand (B, H, T_enc) → (B, H, T_enc, T_enc) by tiling each
        # row T_enc times so attn_weights is symmetric and sum-to-1 on dim=-1.
        attn_scores = attn_scores.unsqueeze(-1).expand(-1, -1, -1, T_enc)  # (B, H, T, T)

        attn_weights = F.softmax(attn_scores, dim=-1)   # (B, H, T_enc, T_enc)

        # Attention context: (B, H, T_enc, D) @ (B, T_enc, D)^T = (B, H, T_enc)
        # then sum over T_enc → (B, H, D)
        attn_ctx = torch.einsum("bhtk,btd->bhd", attn_weights, V)            # (B, H, D)

        # Expand weights to (B, H, T_enc, T_enc) for the return value
        attn_out = attn_weights

        # ── 4. Fuse decoder state + attention context ────────────────────
        combined = torch.cat([dec_out, attn_ctx], dim=-1)   # (B, H, 2D)
        # static_vc is (B, D) — expand to (B, H, D) so it broadcasts over time
        static_vc_exp = static_vc.unsqueeze(1).expand(-1, self._h, -1)  # (B, H, D)
        enriched = self.post_attn_grn(
            combined, context=static_vc_exp
        )

        # ── 7. Quantile head ─────────────────────────────────────────────
        enriched = self._post_attn_proj(enriched)      # (B, H, D) — drop residual
        quantiles = self.quantile_head(enriched)       # (B, H, Q)

        return quantiles, attn_out


# ════════════════════════════════════════════════════════════════════════════
# TFTConfig  –  dataclass for all TFT hyper-parameters  (Task 2e)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TFTConfig:
    """Configuration dataclass for the Temporal Fusion Transformer.

    All fields map directly to the YAML config keys (and to constructor
    arguments) so round-tripping through :func:`TFTModel.from_config` is
    lossless.

    Attributes
    ----------
    num_static   : number of time-invariant input features  *S_stat*.
    num_past     : number of dynamic features known at prediction-time  *S_past*.
    num_future   : number of time-varying features known for the horizon  *S_known*.
    num_targets  : number of target series (default 1).
    d_model      : shared hidden dimension  *D* for all sub-modules.
    d_hidden     : GRN / LSTM hidden dimension (default 4 × d_model).
    num_layers   : number of LSTM layers (shared encoder + decoder).
    past_len     : encoder sequence length  *T* (length of the observation window).
    future_len   : decoder sequence length  *H* (forecast horizon).
    num_quantiles: number of quantile levels *Q* (e.g. 3 → [0.1, 0.5, 0.9]).
    quantiles    : explicit list of quantile levels; auto-generated if ``None``.
    dropout      : dropout probability applied to all sub-modules.
    """

    num_static:    int
    num_past:      int
    num_future:    int
    num_targets:   int        = 1
    d_model:       int        = 128
    d_hidden:      int | None = None
    num_layers:    int        = 1
    past_len:      int        = 24      # T  — encoder window
    future_len:    int        = 12      # H  — forecast horizon
    num_quantiles: int        = 3
    quantiles:     list[float] | None = None
    dropout:       float      = 0.1

    def __post_init__(self) -> None:
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        if self.d_hidden is None:
            self.d_hidden = 4 * self.d_model


# ════════════════════════════════════════════════════════════════════════════
# TFTModel  –  full end-to-end TFT  (paper Fig. 2)  (Task 2e)
# ════════════════════════════════════════════════════════════════════════════

class TFTModel(nn.Module):
    """End-to-end Temporal Fusion Transformer (Lim et al. 2021, Fig. 2).

    Stitches together all TFT sub-modules into a single forward pass:

    1. **Static encoder** — encode ``(B, S_stat)`` static features into
       ``vs``, ``ve``, ``vc`` triples.
    2. **VSN on past** — apply VariableSelectionNetwork per encoder timestep
       to ``(B, T, S_past)`` dynamic history → ``past_vsn`` of shape
       ``(B, T, D)``.
    3. **VSN on future** — apply VariableSelectionNetwork per decoder timestep
       to ``(B, H, S_known)`` known future regressors → ``fut_vsn`` of shape
       ``(B, H, D)``.
    4. **TemporalFusionDecoder** — encode ``past_vsn`` with LSTM, decode
       ``fut_vsn`` with LSTM, fuse with static context, and return quantile
       predictions and attention weights.

    Parameters
    ----------
    config : TFTConfig
        Hyper-parameter container.
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config

        # ── sub-modules ───────────────────────────────────────────────────────
        self.static_encoder = StaticCovariateEncoder(
            config.num_static, config.d_model, config.d_hidden, config.dropout,
        )

        # Variable-selection networks for dynamic features
        self.past_vsn = VariableSelectionNetwork(
            num_inputs=config.num_past, d_model=config.d_model,
            dropout=config.dropout, d_hidden=config.d_hidden,
        )
        self.future_vsn = VariableSelectionNetwork(
            num_inputs=config.num_future, d_model=config.d_model,
            dropout=config.dropout, d_hidden=config.d_hidden,
        )

        # Core fusion decoder
        self.temporal_fusion = TemporalFusionDecoder(
            d_model=config.d_model, num_quantiles=config.num_quantiles,
            past_len=config.past_len, future_len=config.future_len,
            dropout=config.dropout, num_layers=config.num_layers,
        )

    # ── factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def from_config(source: str | dict) -> "TFTModel":
        """Create a :class:`TFTModel` from a YAML file path or a plain ``dict``.

        Parameters
        ----------
        source : file path or config dictionary
            If a ``str``, it is treated as a path to a YAML file whose keys map
            directly to :class:`TFTConfig` field names.  If a ``dict``, keys map
            directly to :class:`TFTConfig` keyword arguments.

        Returns
        -------
        TFTModel
            Initialised model.
        """
        if isinstance(source, dict):
            cfg = TFTConfig(**source)
        else:
            import yaml                              # lazy import
            with open(source) as f:
                data = yaml.safe_load(f)
            cfg = TFTConfig(**data)
        return TFTModel(cfg)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        static_features: torch.Tensor | None,
        past_features:   torch.Tensor,
        known_future:    torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full TFT forward pass.

        Parameters
        ----------
        static_features : ``(B, S_stat)`` or ``None``
            Time-invariant features.  When ``None``, a zeros placeholder is
            used so that only dynamic paths contribute.
        past_features   : ``(B, T, S_past)``
            Observed dynamic history (encoder window).
        known_future    : ``(B, H, S_known)``
            Known future regressors (decoder window).

        Returns
        -------
        dict
            ``'quantiles'``    — ``(B, H, Q)`` predicted quantile levels.
            ``'attn_weights'`` — ``(B, H, T, T)`` temporal attention weights.
        """
        B = past_features.shape[0]
        D = self.config.d_model
        T = self.config.past_len
        H = self.config.future_len

        # ── 1. Static encoder ─────────────────────────────────────────────────
        if static_features is None:
            static_features = torch.zeros(B, self.config.num_static, device=past_features.device)

        # vc → (B, D) context for the fusion decoder
        static_enc = self.static_encoder(static_features)
        vc: torch.Tensor = static_enc["vc"]                         # (B, D)

        # ── 2. VSN on past dynamic features ───────────────────────────────────
        # Apply VSN independently at every encoder timestep.
        # VSN expects (B_flat, S, D_in) where S = num_inputs.
        past_flat   = past_features.reshape(B * T, self.config.num_past, 1)   # (B*T, S_past, 1)
        vsn_past_raw = self.past_vsn(past_flat)                               # (B*T, D)
        past_vsn = vsn_past_raw.reshape(B, T, D)                               # (B, T, D)

        # ── 3. VSN on future known features ─────────────────────────────────
        fut_flat     = known_future.reshape(B * H, self.config.num_future, 1)  # (B*H, S_known, 1)
        vsn_fut_raw  = self.future_vsn(fut_flat)                               # (B*H, D)
        fut_vsn      = vsn_fut_raw.reshape(B, H, D)                            # (B, H, D)

        # ── 4. Temporal Fusion Decoder ───────────────────────────────────────
        # past_vsn is (B, T, D) — the encoder-only sequence.
        # TFD treats dim-1 of past_encoded as T_enc so attn_weights is (B,H,T,T).
        quantiles, attn_weights = self.temporal_fusion(past_vsn, fut_vsn, vc)

        return {"quantiles": quantiles, "attn_weights": attn_weights}
