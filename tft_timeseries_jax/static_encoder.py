import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from .grn import GatedResidualNetwork

class StaticCovariateEncoder(nn.Module):
    """Static Covariate Encoder as in the TFT paper.
    
    Encodes static features into three outputs: variable selection weights (vs),
    enriched static embedding (ve), and static context (vc).
    
    Attributes:
        num_static: number of static input variables
        d_model: model dimension
        d_hidden: GRN hidden dimension (default: 4 * d_model)
        dropout: dropout probability
    """
    num_static: int
    d_model: int
    d_hidden: Optional[int] = None
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> dict:
        """Encodes static features.
        
        Args:
            x: static features of shape (batch, num_static)
            deterministic: if False, apply dropout
        
        Returns:
            A dictionary with keys:
                'vs': variable selection weights of shape (batch, num_static)
                've': enriched embedding for LSTM initial state of shape (batch, d_model)
                'vc': static context vector of shape (batch, d_model)
        """
        # Step 1: embed each static variable to d_model
        # We use a Dense layer that maps from num_static to d_model
        hs = nn.Dense(self.d_model)(x)  # (batch, d_model)
        # Apply a GRN (shared low-level projection)
        hs = GatedResidualNetwork(
            d=self.d_model,
            d_ctx=None,
            d_hidden=self.d_hidden,
            dropout=self.dropout
        )(hs, deterministic=deterministic)  # (batch, d_model)
        
        # Step 2: variable selection (vs) branch
        # We project hs to d_model, then apply a GRN, then project to num_static
        vs_logits = nn.Dense(self.d_model)(hs)
        vs_logits = GatedResidualNetwork(
            d=self.d_model,
            d_ctx=None,
            d_hidden=self.d_hidden,
            dropout=self.dropout
        )(vs_logits, deterministic=deterministic)
        vs_logits = nn.Dense(self.num_static)(vs_logits)  # (batch, num_static)
        vs = nn.softmax(vs_logits, axis=-1)  # (batch, num_static)
        
        # Step 3: enriched static embedding (ve) branch
        ve = nn.Dense(self.d_model)(hs)
        ve = GatedResidualNetwork(
            d=self.d_model,
            d_ctx=None,
            d_hidden=self.d_hidden,
            dropout=self.dropout
        )(ve, deterministic=deterministic)  # (batch, d_model)
        
        # Step 4: static context (vc) branch
        vc = nn.Dense(self.d_model)(hs)
        vc = GatedResidualNetwork(
            d=self.d_model,
            d_ctx=None,
            d_hidden=self.d_hidden,
            dropout=self.dropout
        )(vc, deterministic=deterministic)  # (batch, d_model)
        
        return {
            'vs': vs,
            've': ve,
            'vc': vc
        }