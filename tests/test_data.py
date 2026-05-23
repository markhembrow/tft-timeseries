import numpy as np
import pytest

from tft_timeseries.data import TimeSeriesDataset


def test_dataset_returns_correct_window_shapes():
    N, T, H = 4, 6, 2
    static = np.zeros((N, 2))
    past   = np.zeros((N, T, 3))
    known  = np.zeros((N, T+H, 1))
    target = np.arange(N*T).reshape(N, T, 1).astype(float)
    ds = TimeSeriesDataset(static, past, known, target, past_len=T, future_len=H)
    s, p, k, t = ds[1]
    assert s.shape == (2,),   f"static shape {s.shape}"
    assert p.shape == (T, 3),  f"past shape   {p.shape}"
    assert k.shape == (T+H, 1),f"known shape  {k.shape}"
    assert t.shape == (H, 1),  f"target shape {t.shape}"


def test_dataset_length():
    ds = TimeSeriesDataset(np.zeros((3,1)), np.zeros((3,4,2)),
                           np.zeros((3,6,1)), np.zeros((3,4,1)),
                           past_len=4, future_len=2)
    assert len(ds) == 3
