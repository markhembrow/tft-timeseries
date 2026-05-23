import numpy as np


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
