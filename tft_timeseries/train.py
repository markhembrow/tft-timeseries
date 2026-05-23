"""DataModule for TFT training.

Wraps :class:`~tft_timeseries.data.TimeSeriesDataset` with batching,
scaling, and PyTorch-appropriate collation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tft_timeseries.data import TimeSeriesDataset, scale_data, inverse_scale


class TFTDataModule:
    """Wraps :class:`TimeSeriesDataset` for training and evaluation.

    Parameters
    ----------
    train / val / test : numpy arrays (N, ...) with matching N
    past_len   : encoder sequence length ``T``.
    future_len : forecast horizon ``H``.
    batch_size : samples per mini-batch.
    num_workers: DataLoader workers (default 0).
    """

    def __init__(
        self,
        train_static: np.ndarray,
        train_past: np.ndarray,
        train_known: np.ndarray,
        train_target: np.ndarray,
        val_static:   np.ndarray,
        val_past:     np.ndarray,
        val_known:    np.ndarray,
        val_target:   np.ndarray,
        test_static:  Optional[np.ndarray] = None,
        test_past:    Optional[np.ndarray] = None,
        test_known:   Optional[np.ndarray] = None,
        test_target:  Optional[np.ndarray] = None,
        past_len:     int = 24,
        future_len:   int = 12,
        batch_size:   int = 64,
        num_workers:  int = 0,
    ) -> None:
        self.past_len   = past_len
        self.future_len = future_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_ds = TimeSeriesDataset(
            train_static, train_past, train_known, train_target,
            past_len, future_len,
        )
        self.val_ds = TimeSeriesDataset(
            val_static, val_past, val_known, val_target,
            past_len, future_len,
        )
        self.test_ds: Optional[TimeSeriesDataset] = None
        if test_target is not None:
            self.test_ds = TimeSeriesDataset(
                test_static, test_past, test_known, test_target,
                past_len, future_len,
            )

        # infer feature dims from train arrays
        self.num_static  = train_static.shape[1]
        self.num_past    = train_past.shape[2]
        self.num_future  = train_known.shape[2]
        self.num_targets = train_target.shape[2]

        # target scaler
        flat_targets = train_target.reshape(-1, self.num_targets)
        _, self._target_scaler = scale_data(flat_targets)

    # ── DataLoaders ────────────────────────────────────────────────────────────

    def _collate(self, batch: list[tuple]) -> tuple[torch.Tensor, ...]:
        statics, pasts, knowns, targets = zip(*batch)
        return (
            torch.stack([torch.from_numpy(s).float() for s in statics]),
            torch.stack([torch.from_numpy(p).float() for p in pasts]),
            torch.stack([torch.from_numpy(k).float() for k in knowns]),
            torch.stack([torch.from_numpy(t).float() for t in targets]),
        )

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )

    def test_loader(self) -> Optional[DataLoader]:
        if self.test_ds is None:
            return None
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate,
        )
