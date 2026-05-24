import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple
from .grn import GatedResidualNetwork

class TemporalFusionDecoder(nn.Module):
    """Temporal Fusion Decoder as in the TFT paper.
    
    Attributes:
        d_model: model dimension
        num_quantiles: number of quantile levels
        past_len: encoder sequence length (T)
        future_len: decoder sequence length (H)
        dropout: dropout probability
        num_layers: number of LSTM layers
        d_hidden: hidden dimension for GRNs (default: 4 * d_model)
    """
    d_model: int
    num_quantiles: int
    past_len: int
    future_len: int
    dropout: float = 0.1
    num_layers: int = 1
    d_hidden: Optional[int] = None

    def setup(self):
        """Sets up the layers."""
        # Determine d_hidden for GRNs
        d_hidden = self.d_hidden if self.d_hidden is not None else 4 * self.d_model
        # LSTM cells
        self.encoder_lstm_cell = nn.LSTMCell(self.d_model)
        self.decoder_lstm_cell = nn.LSTMCell(self.d_model)
        # Projection for future_known
        self.future_proj = nn.Dense(self.d_model)
        # Attention projections
        self.attn_k_proj = nn.Dense(self.d_model)
        self.attn_v_proj = nn.Dense(self.d_model)
        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            d=2 * self.d_model,
            d_ctx=self.d_model,
            d_hidden=d_hidden,
            dropout=self.dropout
        )
        # Quantile head
        self.quantile_head = nn.Dense(self.num_quantiles)
        # Projection after GRN (as in PyTorch version)
        self._post_attn_proj = nn.Dense(self.d_model)  # Flax uses use_bias=True by default

    def __call__(self, 
                 past_encoded: jnp.ndarray,    # (batch, T, d_model)
                 future_known: jnp.ndarray,    # (batch, H, d_model)  [already projected?]
                 static_vc: jnp.ndarray,       # (batch, d_model)
                 deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Applies the Temporal Fusion Decoder.
        
        Args:
            past_encoded: encoder output from the past VSN (batch, T, d_model)
            future_known: known future covariates (batch, H, d_model) 
                          Note: In the PyTorch version, the future_known is projected to d_model.
            static_vc: static context vector from the static encoder (batch, d_model)
            deterministic: if False, apply dropout
        
        Returns:
            quantiles: (batch, H, num_quantiles)
            attn_weights: (batch, H, T, T)  [attention weights for each decoder step over encoder steps]
        """
        batch_size = past_encoded.shape[0]
        
        # --- 1. Encoder LSTM (processes past_encoded) ---
        # Initialize hidden and cell states to zeros
        encoder_hidden = jnp.zeros((batch_size, self.d_model))
        encoder_cell = jnp.zeros((batch_size, self.d_model))
        encoder_outputs = []  # will store the hidden state for each time step
        
        for t in range(self.past_len):
            x_t = past_encoded[:, t, :]  # (batch, d_model)
            carry = (encoder_hidden, encoder_cell)
            new_carry, y = self.encoder_lstm_cell(carry, x_t)
            encoder_hidden, encoder_cell = new_carry
            encoder_outputs.append(y)
        # Stack to get (batch, T, d_model)
        encoder_outputs = jnp.stack(encoder_outputs, axis=1)  # (batch, T, d_model)
        
        # --- 2. Decoder LSTM (processes future_known) ---
        # First, project future_known to d_model and apply silu activation
        future_known_proj = nn.silu(self.future_proj(future_known))  # (batch, H, d_model)
        
        # Initialize hidden and cell states to zeros
        decoder_hidden = jnp.zeros((batch_size, self.d_model))
        decoder_cell = jnp.zeros((batch_size, self.d_model))
        decoder_outputs = []  # will store the hidden state for each time step
        
        for t in range(self.future_len):
            x_t = future_known_proj[:, t, :]  # (batch, d_model)
            carry = (decoder_hidden, decoder_cell)
            new_carry, y = self.decoder_lstm_cell(carry, x_t)
            decoder_hidden, decoder_cell = new_carry
            decoder_outputs.append(y)
        # Stack to get (batch, H, d_model)
        decoder_outputs = jnp.stack(decoder_outputs, axis=1)  # (batch, H, d_model)
        
        # --- 3. Temporal Attention ---
        # We use the decoder_outputs as the query and the encoder_outputs as the key and value.
        # We follow the PyTorch version: 
        #   Q = decoder_outputs (batch, H, d_model)
        #   K = self.attn_k_proj(encoder_outputs) (batch, T, d_model)
        #   V = self.attn_v_proj(encoder_outputs) (batch, T, d_model)
        #
        # Then compute attention scores: Q @ K^T / sqrt(d_model)
        # Then softmax to get attention weights (batch, H, T)
        # Then we expand to (batch, H, T, T) by repeating the attention weights T times? 
        # Actually, in the PyTorch version, they do:
        #   attn_scores = torch.einsum("bhd,btd->bht", Q, K) / (D ** 0.5)
        #   attn_scores = attn_scores.unsqueeze(-1).expand(-1, -1, -1, T_enc)  # (B, H, T, T)
        #   attn_weights = F.softmax(attn_scores, dim=-1)   # (B, H, T, T)
        #
        # We'll do the same.
        #
        Q = decoder_outputs  # (batch, H, d_model)  [no projection for Q, as in the paper]
        K = self.attn_k_proj(encoder_outputs)  # (batch, T, d_model)
        V = self.attn_v_proj(encoder_outputs)  # (batch, T, d_model)
        
        # Compute attention scores
        attn_scores = jnp.einsum('bhd,btd->bht', Q, K) / jnp.sqrt(self.d_model)  # (batch, H, T)
        # Expand to (batch, H, T, T)
        attn_scores = attn_scores[:, :, :, None]  # (batch, H, T, 1)
        attn_scores = jnp.broadcast_to(attn_scores, (batch_size, self.future_len, self.past_len, self.past_len))
        # Apply softmax
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (batch, H, T, T)
        
        # Compute attention context: (batch, H, T, T) @ (batch, T, d_model) -> (batch, H, T, d_model) 
        # then sum over the T dimension? 
        # Actually, in the PyTorch version:
        #   attn_ctx = torch.einsum("bhtk,btd->bhd", attn_weights, V)  # (batch, H, d_model)
        attn_ctx = jnp.einsum('bhtk,btd->bhd', attn_weights, V)  # (batch, H, d_model)
        
        # --- 4. Fuse decoder state and attention context ---
        # Concatenate decoder_outputs and attn_ctx: (batch, H, 2*d_model)
        combined = jnp.concatenate([decoder_outputs, attn_ctx], axis=-1)  # (batch, H, 2*d_model)
        # Expand static_vc to (batch, H, d_model)
        static_vc_exp = jnp.broadcast_to(static_vc[:, None, :], (batch_size, self.future_len, self.d_model))
        # Apply GRN with static_vc as context
        fused = self.post_attn_grn(combined, context=static_vc_exp, deterministic=deterministic)  # (batch, H, 2*d_model)
        # Project back to d_model (as in the PyTorch version: they have a projection after the GRN)
        fused = self._post_attn_proj(fused)  # (batch, H, d_model)
        
        # --- 5. Quantile head ---
        quantiles = self.quantile_head(fused)  # (batch, H, num_quantiles)
        
        return quantiles, attn_weights