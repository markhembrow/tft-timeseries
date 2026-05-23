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


def test_scale_standardizes_features():
    """scale_data standardises each feature to mean 0 / std 1."""
    from tft_timeseries.data import scale_data, inverse_scale
    rng = np.random.default_rng(42)
    arr = rng.normal(loc=5.0, scale=2.0, size=(100, 2))
    scaled, scalers = scale_data(arr)
    # mean ~ 0  and  std ~ 1  for every column
    means = scaled.mean(axis=0)
    stds  = scaled.std(axis=0)
    assert np.allclose(means, 0.0, atol=1e-7), f"per-feature means: {means}"
    assert np.allclose(stds,  1.0, atol=1e-7), f"per-feature stds: {stds}"
    # round-trip restores original values
    recon = inverse_scale(scaled, scalers)
    assert np.allclose(recon, arr, atol=1e-7), f"max diff: {np.abs(recon - arr).max()}"


def test_inverse_scale_roundtrip():
    """inverse_scale round-trips to original array to within 1e-10."""
    from tft_timeseries.data import scale_data, inverse_scale
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(50, 3)).astype(np.float64)
    scaled, scalers = scale_data(arr)
    recon = inverse_scale(scaled, scalers)
    assert np.allclose(recon, arr, atol=1e-10), f"max diff: {np.abs(recon - arr).max()}"
