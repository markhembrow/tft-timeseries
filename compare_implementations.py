"""
Comparison script for PyTorch vs JAX TFT implementations.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np
import torch
import sys
import os

# Add the current directory to the path to import the models
sys.path.append(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tft_timeseries_jax'))

# Import PyTorch model
from tft_timeseries.model import TFTModel as PyTorchTFTModel
from tft_timeseries.model import TFTConfig as PyTorchTFTConfig

# Import JAX model
from tft_timeseries_jax.model import TFTModel as JaxTFTModel
from tft_timeseries_jax.config import TFTConfig as JaxTFTConfig

def create_pytorch_config():
    """Create a PyTorch TFTConfig for testing."""
    return PyTorchTFTConfig(
        num_static=2,
        num_past=3,
        num_future=4,
        d_model=32,
        d_hidden=64,
        num_layers=1,
        past_len=24,
        future_len=12,
        num_quantiles=3,
        dropout=0.1
    )

def create_jax_config():
    """Create a JAX TFTConfig for testing."""
    return JaxTFTConfig(
        num_static=2,
        num_past=3,
        num_future=4,
        d_model=32,
        d_hidden=64,
        num_layers=1,
        past_len=24,
        future_len=12,
        num_quantiles=3,
        dropout=0.1
    )

def count_parameters_pytorch(model):
    """Count the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters_jax(model):
    """Count the number of parameters in a JAX/Flax model."""
    # We'll create a dummy input to initialize the model and then count parameters
    # This is a bit tricky because we need to initialize the model first.
    # For simplicity, we'll assume the model has been initialized and we can access its parameters.
    # Alternatively, we can use jax.tree_util.tree_reduce to count the leaves of the parameter tree.
    # But note: we haven't initialized the model yet in this function.
    # We'll do the initialization in the main function and pass the initialized model.
    pass

def main():
    print("=== TFT Model Comparison: PyTorch vs JAX ===\n")
    
    # Create configurations
    pytorch_config = create_pytorch_config()
    jax_config = create_jax_config()
    
    print("PyTorch Config:", pytorch_config)
    print("JAX Config:", jax_config)
    print()
    
    # Initialize models
    print("Initializing models...")
    pytorch_model = PyTorchTFTModel(pytorch_config)
    # For JAX, we need to initialize the model with a random key
    jax_model = JaxTFTModel(jax_config)
    
    # Count parameters (PyTorch)
    pytorch_params = count_parameters_pytorch(pytorch_model)
    print(f"PyTorch model parameters: {pytorch_params:,}")
    
    # We'll count JAX parameters after initialization with dummy data
    print()
    
    # Create dummy input data
    batch_size = 4
    # Static features: (batch_size, num_static)
    static_features_np = np.random.randn(batch_size, pytorch_config.num_static).astype(np.float32)
    # Past features: (batch_size, past_len, num_past)
    past_features_np = np.random.randn(batch_size, pytorch_config.past_len, pytorch_config.num_past).astype(np.float32)
    # Known future features: (batch_size, past_len + future_len, num_future)
    # Note: In the TFT, the known_future includes both past and future known values?
    # Actually, looking at the PyTorch model forward pass:
    #   known_future: (B, T+H, S_known) where T is past_len and H is future_len
    #   Then we take the last H steps for the decoder.
    # So we need to provide (batch_size, past_len + future_len, num_future)
    known_future_np = np.random.randn(
        batch_size, 
        pytorch_config.past_len + pytorch_config.future_len, 
        pytorch_config.num_future
    ).astype(np.float32)
    
    # Convert to PyTorch tensors
    static_features_torch = torch.from_numpy(static_features_np)
    past_features_torch = torch.from_numpy(past_features_np)
    known_future_torch = torch.from_numpy(known_future_np)
    
    # Convert to JAX arrays
    static_features_jax = jnp.array(static_features_np)
    past_features_jax = jnp.array(past_features_np)
    known_future_jax = jnp.array(known_future_np)
    
    print("Input shapes:")
    print(f"  Static: {static_features_np.shape}")
    print(f"  Past: {past_features_np.shape}")
    print(f"  Known Future: {known_future_np.shape}")
    print()
    
    # --- PyTorch Forward Pass ---
    print("Running PyTorch forward pass...")
    pytorch_model.eval()  # Set to evaluation mode
    start_time = time.time()
    with torch.no_grad():
        pytorch_output = pytorch_model(
            static_features_torch,
            past_features_torch,
            known_future_torch
        )
    pytorch_time = time.time() - start_time
    print(f"PyTorch forward pass time: {pytorch_time:.4f} seconds")
    print(f"PyTorch output quantiles shape: {pytorch_output['quantiles'].shape}")
    print(f"PyTorch output attn_weights shape: {pytorch_output['attn_weights'].shape}")
    print()
    
    # --- JAX Forward Pass ---
    print("Running JAX forward pass...")
    # Initialize the JAX model with random key and dummy data
    # We need to initialize the model parameters first
    rng = jax.random.PRNGKey(0)
    # Initialize the model by calling it with dummy data
    # Note: We have to set deterministic=True to disable dropout during initialization for consistent shapes
    # But note: the JAX model's __call__ method has a deterministic argument
    # We'll create a wrapper to initialize the model
    def jax_model_init(rng, static, past, known_future):
        # We need to create a model instance and then call it
        model = JaxTFTModel(jax_config)
        return model.init(rng, static, past, known_future, deterministic=True)
    
    # Initialize parameters
    jax_variables = jax_model_init(rng, static_features_jax, past_features_jax, known_future_jax)
    # Count parameters in JAX model
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))
    jax_params = count_params(jax_variables['params'])
    print(f"JAX model parameters: {jax_params:,}")
    
    # Now run the forward pass
    def jax_model_apply(variables, static, past, known_future):
        model = JaxTFTModel(jax_config)
        return model.apply(variables, static, past, known_future, deterministic=True)
    
    start_time = time.time()
    jax_output = jax_model_apply(jax_variables, static_features_jax, past_features_jax, known_future_jax)
    # Note: The first call to a JAX function includes compilation time.
    # To get a fair comparison, we should run it multiple times and ignore the first run.
    # But for simplicity, we'll just report the time and note that it includes compilation.
    jax_time = time.time() - start_time
    print(f"JAX forward pass time: {jax_time:.4f} seconds (includes compilation)")
    print(f"JAX output quantiles shape: {jax_output['quantiles'].shape}")
    print(f"JAX output attn_weights shape: {jax_output['attn_weights'].shape}")
    print()
    
    # --- Comparison ---
    print("=== Comparison Summary ===")
    print(f"Parameters - PyTorch: {pytorch_params:,}, JAX: {jax_params:,}")
    print(f"Time - PyTorch: {pytorch_time:.4f}s, JAX: {jax_time:.4f}s")
    quantiles_match = pytorch_output['quantiles'].shape == jax_output['quantiles'].shape
    attn_match = pytorch_output['attn_weights'].shape == jax_output['attn_weights'].shape
    print(f"Output Shapes Match: {quantiles_match and attn_match}")
    print(f"  Quantiles shape match: {quantiles_match}")
    print(f"  Attention weights shape match: {attn_match}")
    print()
    
    # Note on numerical equivalence
    print("Note: Numerical values are expected to differ due to:")
    print("  - Different random initializations")
    print("  - Potential differences in layer implementations (e.g., LSTM, GRN)")
    print("  - Different numerical precision handling between PyTorch and JAX")
    print("  - The JAX timing includes JIT compilation overhead")

if __name__ == "__main__":
    main()