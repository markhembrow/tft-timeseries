import numpy as np
from typing import Tuple, Dict


class TimeSeriesDataset:
    """Windowed time-series dataset for TFT.

    Parameters
    ----------
    static : (N, S)  static features
    past   : (N, T, D_past)  observed history
    known  : (N, T+H, D_known) known-known + known-unknown features
    target : (N, T, D_target) target values in history window
    past_len  : T  length of encoder window
    future_len: H  length of decoder/horizon window
    """

    def __init__(self, static, past, known, target, past_len, future_len):
        assert static.shape[0] == past.shape[0] == known.shape[0] == target.shape[0], \
            "All arrays must share the same batch dimension N"
        self.static  = static
        self.past    = past
        self.known   = known
        self.target  = target
        self.past_len    = past_len
        self.future_len  = future_len

    def __len__(self):
        return len(self.static)

    def __getitem__(self, i):
        return (self.static[i], self.past[i], self.known[i], self.target[i, -self.future_len:])


def scale_data(arr: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Per-feature standardisation.

    Normalises each feature dimension (column) independently so the
    scaled array has mean 0 and standard deviation 1 along axis 0.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape ``(T, D)`` or ``(B, T, D)`` (or any broadcastable
        shape).  Normalisation is applied along axis 0 regardless of the
        number of leading dimensions.

    Returns
    -------
    scaled : np.ndarray
        Standardised copy of *arr*, same shape.
    scalers : dict
        Mapping ``{'means': ..., 'stds': ...}`` carrying the per-feature
        (axis-0) statistics needed by :func:`inverse_scale`.

    Notes
    -----
    * 0-D and 1-D arrays are handled as degenerate cases: a single scalar
      has mean equal to itself and std 1, so the round-trip is an identity.
    * Features whose standard deviation is 0 (constant) are assigned a
      std of 1 to avoid division by zero; they will map back to the
      constant value after an inverse scale.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim < 2:
        # 0-D (scalar) or 1-D — treat as a single-entry array, then squeeze
        mean = arr.mean(axis=0, keepdims=True) if arr.ndim == 1 else np.array(0.0)
        std  = arr.std(axis=0, keepdims=True) if arr.ndim == 1 else np.array(1.0)
        has_std = np.zeros_like(std, dtype=bool)
    else:
        mean = arr.mean(axis=0, keepdims=True)
        std  = arr.std(axis=0, keepdims=True)
        has_std = std > 0

    # Avoid division by zero for constant features
    std = np.where(has_std, std, 1.0)

    scalers = {"means": mean, "stds": std, "has_std": has_std}
    scaled = (arr - mean) / std
    return scaled, scalers


def create_windowed_samples(features: np.ndarray, past_len: int, future_len: int):
    """Convert flat (T, D) matrix into overlapping windows for TFT.

    Parameters
    ----------
    features  : (T, D)   full time-series matrix
    past_len  : T_enc    encoder length
    future_len: H        horizon / decoder length

    Returns
    -------
    past  : (N_w, T_enc, D)
    future: (N_w, H, D)
    target: (N_w, H, D)
    where N_w = max(0, T - T_enc - H + 1)
    """
    features = np.asarray(features, dtype=float)
    T, D = features.shape
    N_w = max(0, T - past_len - future_len + 1)
    if N_w == 0:
        empty = np.empty((0, past_len, D))
        return empty, np.empty((0, future_len, D)), np.empty((0, future_len, D))

    def _slice(start: int, length: int) -> np.ndarray:
        if start >= T:
            return np.zeros((length, D), dtype=features.dtype)
        end = min(start + length, T)
        chunk = features[start:end]
        if chunk.shape[0] < length:
            chunk = np.pad(chunk, ((0, length - chunk.shape[0]), (0, 0)))
        return chunk

    past_arr   = np.stack([_slice(i,                  past_len)        for i in range(N_w)])
    future_arr = np.stack([_slice(i + past_len,      future_len)      for i in range(N_w)])
    target_arr = np.stack([_slice(i + past_len + future_len, future_len) for i in range(N_w)])
    return past_arr, future_arr, target_arr


def inverse_scale(scaled: np.ndarray, scalers: dict) -> np.ndarray:
    """Apply inverse of :func:`scale_data`.

    Parameters
    ----------
    scaled : np.ndarray
        Standardised array, same shape as the original input to
        :func:`scale_data`.
    scalers : dict
        Dictionary returned by :func:`scale_data`, containing ``'means'``
        and ``'stds'``.

    Returns
    -------
    np.ndarray
        Array restored to the original scale.
    """
    mean = scalers["means"]
    std  = scalers["stds"]
    return scaled * std + mean
