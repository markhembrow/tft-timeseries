from dataclasses import dataclass
from typing import Optional, List

@dataclass
class TFTConfig:
    """Configuration dataclass for the Temporal Fusion Transformer (JAX version)."""
    num_static: int
    num_past: int
    num_future: int
    num_targets: int = 1
    d_model: int = 128
    d_hidden: Optional[int] = None
    num_layers: int = 1
    past_len: int = 24      # T  — encoder window
    future_len: int = 12    # H  — forecast horizon
    num_quantiles: int = 3
    quantiles: Optional[List[float]] = None
    dropout: float = 0.1

    def __post_init__(self):
        if self.quantiles is None:
            self.quantiles = [0.1, 0.5, 0.9]
        if self.d_hidden is None:
            self.d_hidden = 4 * self.d_model