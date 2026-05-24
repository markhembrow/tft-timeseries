import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from .grn import GatedResidualNetwork

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) as in the TFT paper.
    
    Attributes:
        num_inputs: number of input feature variables (S)
        d_model: output embedding dimension (D)
        dropout: dropout rate
        d_hidden: GRN hidden dimension (default: 4 * d_model)
    """
    num_inputs: int
    d_model: int
    dropout: float = 0.1
    d_hidden: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, context: Optional[jnp.ndarray] = None, deterministic: bool = True) -> jnp.ndarray:
        """Applies the VSN module.
        
        Args:
            x: input tensor of shape (batch, seq_len, num_inputs, input_dim) 
               but note: in the TFT, each variable is embedded to d_model first? 
               Actually, the original implementation flattens the last two dimensions? 
               We follow the PyTorch version: input is (batch, seq_len, num_inputs) 
               and each variable is considered to have dimension 1 (then embedded).
               However, to be consistent with the PyTorch version, we assume the input 
               is already embedded to d_model per variable? 
               Let's clarify: 
               In the PyTorch version, the VSN expects input of shape (B, S, D_in) 
               where D_in is the input dimension per variable (often 1 for raw features, 
               but can be more if pre-processed). 
               Then, it projects to (B, S, d_model) via a linear layer.
               
               We'll design the JAX version similarly: 
               Input: (batch, seq_len, num_inputs, input_dim) 
               But to match the PyTorch version which collapses the input_dim into the 
               projection, we can allow input_dim to be arbitrary and then project to d_model.
               
               However, for simplicity and to match the PyTorch version exactly, we note that 
               in the TFT, the input to the VSN for past and future is already embedded? 
               Actually, looking at the PyTorch TFTModel forward pass:
                 past_flat = past_features.reshape(B * T, self.config.num_past, 1)   # (B*T, S_past, 1)
                 vsn_past_raw = self.past_vsn(past_flat)                               # (B*T, D)
               So the input to VSN is (B*T, S_past, 1) -> output (B*T, D)
               Similarly for future.
               
               Therefore, we design the VSN to accept input of shape (..., num_inputs, input_dim) 
               and output (..., d_model). We'll treat the leading dimensions as batch.
               
        Returns:
            output tensor of shape (..., d_model)
        """
        # Input shape: (..., num_inputs, input_dim)
        input_shape = x.shape
        num_inputs = input_shape[-2]
        input_dim = input_shape[-1]
        
        # Flatten the last two dimensions: (..., num_inputs * input_dim)
        flat_x = x.reshape(-1, num_inputs * input_dim)
        
        # Project to (..., num_inputs * d_model)
        # We use a Dense layer that outputs num_inputs * d_model
        projected = nn.Dense(self.num_inputs * self.d_model)(flat_x)
        # Reshape to (..., num_inputs, d_model)
        projected = projected.reshape(-1, self.num_inputs, self.d_model)
        
        # Apply GRN to each variable independently (no weight sharing)
        # We'll use a map over the num_inputs dimension.
        # We can use jax.vmap or a loop. Since num_inputs is small, we can use a loop.
        # But note: we are in a flax module, we want to avoid loops in the module definition?
        # Actually, we can use a list of GRNs and then stack.
        
        # Create a list of GRNs (one per variable)
        grn_outputs = []
        for i in range(self.num_inputs):
            grn = GatedResidualNetwork(
                d=self.d_model,
                d_ctx=None,  # VSN does not use context in the GRN step? 
                d_hidden=self.d_hidden,
                dropout=self.dropout
            )
            # Apply GRN to the i-th variable
            grn_out = grn(projected[:, i, :], deterministic=deterministic)
            grn_outputs.append(grn_out)
        
        # Stack to get (batch, num_inputs, d_model)
        grn_stack = jnp.stack(grn_outputs, axis=1)  # (batch, num_inputs, d_model)
        
        # Compute weights: apply a linear layer to get logits of shape (batch, num_inputs)
        # Then softmax
        # We use a Dense layer that outputs 1 per variable, then squeeze.
        w_logits = nn.Dense(1)(grn_stack).squeeze(-1)  # (batch, num_inputs)
        weights = nn.softmax(w_logits, axis=-1)  # (batch, num_inputs)
        
        # Weighted sum: (batch, num_inputs, d_model) * (batch, num_inputs, 1) -> (batch, d_model)
        weighted = grn_stack * weights[:, :, None]
        combined = jnp.sum(weighted, axis=1)  # (batch, d_model)
        
        # Final projection (as in the PyTorch version)
        output = nn.Dense(self.d_model)(combined)
        
        return output