import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) as in the TFT paper.
    
    Attributes:
        d: input / output dimension
        d_ctx: context feature dimension (optional)
        d_hidden: hidden-layer dimension (default: 4 * d)
        dropout: dropout probability
    """
    d: int
    d_ctx: Optional[int] = None
    d_hidden: Optional[int] = None
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, deterministic: bool = True) -> jnp.ndarray:
        """Applies the GRN module.
        
        Args:
            x: input tensor of shape (..., d)
            context: optional context tensor of shape (..., d_ctx)
            deterministic: if False, apply dropout
        
        Returns:
            output tensor of shape (..., d)
        """
        d_hidden = self.d_hidden or 4 * self.d
        
        # First linear layer + SiLU
        h = nn.silu(nn.Dense(d_hidden)(x))
        
        # Add context if provided
        if context is not None and self.d_ctx is not None:
            h = h + nn.Dense(d_hidden, use_bias=False)(context)
        
        # Gate linear layer + SiLU
        g = nn.silu(nn.Dense(d_hidden)(x))
        
        # Apply gate and dropout
        h = h * g
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        
        # Second linear layer
        h = nn.Dense(self.d)(h)
        
        # Skip connection
        skip = nn.Dense(self.d)(x)
        
        # Add and apply dropout
        h = h + skip
        h = nn.Dropout(rate=self.dropout)(h, deterministic=deterministic)
        
        # Layer normalization
        h = nn.LayerNorm()(h)
        
        return h