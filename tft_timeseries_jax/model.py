import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Dict, Tuple
from .config import TFTConfig
from .static_encoder import StaticCovariateEncoder
from .vsn import VariableSelectionNetwork
from .temporal_fusion_decoder import TemporalFusionDecoder

class TFTModel(nn.Module):
    """End-to-end Temporal Fusion Transformer (JAX version).
    
    Attributes:
        config: TFTConfig instance
    """
    config: TFTConfig

    def setup(self):
        """Sets up the sub-modules."""
        self.static_encoder = StaticCovariateEncoder(
            num_static=self.config.num_static,
            d_model=self.config.d_model,
            d_hidden=self.config.d_hidden,
            dropout=self.config.dropout
        )
        self.past_vsn = VariableSelectionNetwork(
            num_inputs=self.config.num_past,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            d_hidden=self.config.d_hidden
        )
        self.future_vsn = VariableSelectionNetwork(
            num_inputs=self.config.num_future,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            d_hidden=self.config.d_hidden
        )
        self.temporal_fusion = TemporalFusionDecoder(
            d_model=self.config.d_model,
            num_quantiles=self.config.num_quantiles,
            past_len=self.config.past_len,
            future_len=self.config.future_len,
            dropout=self.config.dropout,
            num_layers=self.config.num_layers
        )

    def __call__(self, 
                 static_features: Optional[jnp.ndarray], 
                 past_features: jnp.ndarray, 
                 known_future: jnp.ndarray,
                 deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        """Applies the TFT model.
        
        Args:
            static_features: static features of shape (batch, num_static) or None
            past_features: past features of shape (batch, past_len, num_past)
            known_future: known future features of shape (batch, future_len + num_past?, num_future) 
                           Note: In the PyTorch version, the known_future is of shape (batch, T+H, S_known)
                                 and we take the last H steps for the decoder.
            deterministic: if False, apply dropout
        
        Returns:
            A dictionary with keys:
                'quantiles': (batch, future_len, num_quantiles)
                'attn_weights': (batch, future_len, past_len, past_len)
        """
        batch_size = past_features.shape[0]
        past_len = self.config.past_len
        future_len = self.config.future_len
        
        # --- 1. Static Encoder ---
        if static_features is None:
            static_features = jnp.zeros((batch_size, self.config.num_static))
        static_enc = self.static_encoder(static_features, deterministic=deterministic)
        vc = static_enc['vc']  # (batch, d_model)
        
        # --- 2. VSN on past dynamic features ---
        # Reshape past_features to (batch * past_len, num_past, 1) as in the PyTorch version
        past_flat = past_features.reshape(-1, self.config.num_past, 1)  # (batch * past_len, num_past, 1)
        vsn_past_raw = self.past_vsn(past_flat, deterministic=deterministic)  # (batch * past_len, d_model)
        past_vsn = vsn_past_raw.reshape(batch_size, past_len, self.config.d_model)  # (batch, past_len, d_model)
        
        # --- 3. VSN on horizon-time known features ---
        # In the PyTorch version, known_future is of shape (batch, T+H, S_known)
        # We take the last future_len steps for the decoder.
        known_horizon = known_future[:, -self.config.future_len :, :]  # (batch, future_len, num_future)
        # Reshape to (batch * future_len, num_future, 1)
        fut_flat = known_horizon.reshape(-1, self.config.num_future, 1)  # (batch * future_len, num_future, 1)
        vsn_fut_raw = self.future_vsn(fut_flat, deterministic=deterministic)  # (batch * future_len, d_model)
        fut_vsn = vsn_fut_raw.reshape(batch_size, future_len, self.config.d_model)  # (batch, future_len, d_model)
        
        # --- 4. Temporal Fusion Decoder ---
        quantiles, attn_weights = self.temporal_fusion(
            past_encoded=past_vsn,      # (batch, past_len, d_model)
            future_known=fut_vsn,       # (batch, future_len, d_model)
            static_vc=vc,               # (batch, d_model)
            deterministic=deterministic
        )
        
        return {
            "quantiles": quantiles,  # (batch, future_len, num_quantiles)
            "attn_weights": attn_weights  # (batch, future_len, past_len, past_len)
        }