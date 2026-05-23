"""TFTInferenceEngine — deploy TFT checkpoints for production inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger(__name__)


class TFTInferenceEngine:
    """Load a trained :class:`~tft_timeseries.model.TFTModel` and serve
    point- and multi-step forecasts.

    Parameters
    ----------
    checkpoint_path : path to a ``state_dict`` ``.pt`` file saved by
        :meth:`TTrainer.save_checkpoint`.
    config          : :class:`~tft_timeseries.model.TFTConfig` or ``None``.
        If ``None``, attempts to load ``config.yaml`` from the same directory
        as *checkpoint_path*.
    device          : ``'auto'``, ``'cpu'``, or ``'cuda'``.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Optional[dict] = None,
        device: str = "auto",
    ) -> None:
        from tft_timeseries.model import TFTModel, TFTConfig

        self.device = device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = Path(checkpoint_path)
        # ── load config ────────────────────────────────────────────────────
        if config is None:
            cfg_path = ckpt.parent / "config.yaml"
            if cfg_path.exists():
                self.model = TFTModel.from_config(str(cfg_path))
            else:
                raise FileNotFoundError(
                    f"No config.yaml found next to {ckpt}. "
                    "Pass config explicitly or save config.yaml with the checkpoint."
                )
        else:
            self.model = TFTModel.from_config(config)

        state = torch.load(ckpt, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint from %s", ckpt)
        self.model.to(self.device).eval()

        self._cfg = self.model.config

    # ── single-batch prediction ────────────────────────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self,
        static:  np.ndarray,
        past:    np.ndarray,
        future:  np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run a forward pass on one batch.

        Parameters
        ----------
        static  : ``(B, S_stat)``
        past    : ``(B, T, S_past)``
        future  : ``(B, H, S_known)``

        Returns
        -------
        dict with ``'quantiles'`` of shape ``(B, H, Q)`` and
        ``'attn_weights'`` of shape ``(B, H, T, T)`` — both numpy arrays.
        """
        device = torch.device(self.device)
        xs = torch.from_numpy(static).float().to(device)
        xp = torch.from_numpy(past  ).float().to(device)
        xf = torch.from_numpy(future).float().to(device)

        out = self.model(xs, xp, xf)
        return {k: v.cpu().numpy() for k, v in out.items()}

    # ── rolling multi-step forecast ────────────────────────────────────────────

    @torch.no_grad()
    def predict_series(
        self,
        past_static:  np.ndarray,   # (B, S_stat)
        past_hist:    np.ndarray,   # (B, T, S_past + S_target_in_history)
        known_future: np.ndarray,   # (B, H, S_known)
        target_col:   int = -1,     # index of target col in known_future if absent
    ) -> dict[str, np.ndarray]:
        """Run forecast for a single time-series panel.

        *past_hist* is the full history including targets; the target column
        is separated from the dynamic feature columns before the forward pass.
        """
        # slice off the target column from past_hist and known_future
        cfg = self._cfg
        B = past_hist.shape[0]
        T = cfg.past_len
        H = cfg.future_len

        xp = past_hist[:, :, :cfg.num_past]         # (B, T, S_past)
        xf = known_future[:, :, :cfg.num_future]    # (B, H, S_known)
        xs = past_static[:, :cfg.num_static]         # (B, S_stat)
        return self.predict_batch(xs, xp, xf)
