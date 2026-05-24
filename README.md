# Temporal Fusion Transformer (TFT) for Time Series Forecasting

A complete implementation of the Temporal Fusion Transformer (TFT) architecture for interpretable multi-horizon time series forecasting, available in both PyTorch and JAX versions. This repository includes the core model, training scripts, test suite, data processing utilities, and a Gradio-based user interface for electricity demand forecasting.

## Overview

The Temporal Fusion Transformer (Lim et al., 2021) combines the strengths of recurrent networks, temporal attention mechanisms, and transformer architectures to provide accurate and interpretable forecasts across multiple time horizons.

This implementation includes:
- Complete TFT architecture with Variable Selection Networks (VSN), Gated Residual Networks (GRN), and Temporal Fusion Decoder in both PyTorch and JAX
- Configurable hyperparameters via YAML or programmatic interface
- Training scripts for the AEMO NSW electricity demand dataset
- Comprehensive test suite covering model components
- Gradio web interface for forecasting, evaluation, attention analysis, and what-if scenarios
- Accessible via Tailscale VPN for secure remote access
- Performance comparison between PyTorch and JAX implementations

## Repository Structure

```
tft-timeseries/
├── tft_timeseries/                 # Core TFT implementation (PyTorch)
│   ├── model.py                    # TFTModel, TFTConfig, and sub-components
│   ├── data.py                     # Data processing utilities
│   └── losses.py                   # QuantileLoss implementation
├── tft_timeseries_jax/             # Core TFT implementation (JAX/Flax)
│   ├── model.py                    # TFTModel, TFTConfig, and sub-components
│   ├── grn.py                      # Gated Residual Network
│   ├── vsn.py                      # Variable Selection Network
│   ├── static_encoder.py           # Static Covariate Encoder
│   └── temporal_fusion_decoder.py  # Temporal Fusion Decoder
├── scripts/                        # Training and utility scripts
│   ├── train_aemo.py               # Main training script for AEMO dataset
│   ├── prepare_aemo_data.py        # Data preparation for AEMO dataset
│   ├── query_model.py              # Model inference utilities
│   └── evaluate_example.py         # Example evaluation script
├── tests/                          # Test suite
│   ├── test_model.py               # Model component tests
│   ├── test_data.py                # Data processing tests
│   └── test_losses.py              # Loss function tests
├── checkpoint_aemo/                # Trained model checkpoints (example)
├── data/                           # Data storage
│   └── aemo/                       # AEMO NSW electricity demand data
├── simple_interface.py             # Gradio web interface (uses PyTorch model)
├── TFT_INTERFACE_GUIDE.md          # Detailed interface guide
├── compare_implementations.py      # Performance comparison script (PyTorch vs JAX)
└── README.md                       # This file
```

## Key Features

### Dual Implementation: PyTorch and JAX

The repository provides two implementations of the TFT architecture:
1. **PyTorch Implementation** (`tft_timeseries/`): The original and fully featured implementation
2. **JAX Implementation** (`tft_timeseries_jax/`): A reimplementation using JAX and Flax for functional programming and potential performance benefits

### Custom TFT Implementation

Both implementations include:
1. **Variable Selection Network (VSN)** - Learns importance weights for input variables
2. **Gated Residual Network (GRN)** - Residual blocks with GLU-style gating
3. **Static Covariate Encoder** - Processes time-invariant features
4. **Temporal Fusion Decoder** - LSTM encoder-decoder with temporal attention
5. **End-to-End TFTModel** - Combines all components into a complete forecasting model

### Test Data and Tests

The repository includes comprehensive tests for:
- Variable Selection Network (output shape, attention weights, gradients)
- Gated Residual Network (shape preservation, context effects)
- Temporal Fusion Decoder (quantile outputs, attention weights)
- Full TFT Model (end-to-end functionality, CUDA compatibility)
- Data processing utilities (windowing, scaling, inversion)

Run tests with:
```bash
# PyTorch tests
python -m pytest tests/ -v

# JAX tests (requires JAX installation)
python -m pytest tests/ -v  # Note: JAX tests would need to be written separately
```

### User Interface

The Gradio interface (`simple_interface.py`) provides:
- **Forecast Tab**: Generate 12-hour ahead predictions with confidence intervals (P10, P50, P90)
- **Evaluate Tab**: View model performance metrics on test data
- **Attention Tab**: Analyze temporal attention weights
- **What-If Tab**: Test scenarios (demand spikes, price spikes, holiday effects)
- **Model Info Tab**: View architecture details and parameter counts
- **Export Tab**: CSV export functionality (placeholder)

Access the interface via Tailscale at: `http://100.115.213.88:7862/`

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/markhembrow/tft-timeseries.git
cd tft-timeseries
```

2. Create and activate conda environment (optional but recommended):
```bash
conda create -n tft python=3.10
conda activate tft
```

3. Install dependencies for PyTorch version:
```bash
pip install torch torchvision torchaudio
pip install gradio pyyaml numpy scikit-learn pandas
```

4. Install dependencies for JAX version:
```bash
pip install "jax[cpu]" flax optax  # For CPU version
# For GPU version with CUDA support, follow JAX installation guide:
# pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Training the Model (PyTorch)

To train the TFT model on the AEMO NSW electricity demand dataset using PyTorch:

```bash
python scripts/train_aemo.py
```

Key training parameters (with defaults):
- `--epochs 30`: Number of training epochs
- `--batch-size 256`: Batch size for training
- `--lr 0.001`: Learning rate
- `--checkpoint-dir ./checkpoint_aemo`: Directory for model checkpoints
- `--device auto`: Automatically uses CUDA if available

### Using the Interface

1. Start the Gradio interface:
```bash
python simple_interface.py
```

2. Access via your Tailscale network:
   - Primary URL: http://100.115.213.88:7862/
   - IPv6 Alternative: http://[fd7a:115c:a1e0::4538:d558]:7862/
   - Local (via SSH tunnel): http://localhost:7862/

### Performance Comparison

To compare the PyTorch and JAX implementations:

```bash
# For CPU comparison (works on all systems)
JAX_PLATFORMS=cpu python compare_implementations.py

# For GPU comparison (if JAX GPU is installed)
python compare_implementations.py
```

The comparison script evaluates:
- Model parameter count
- Forward pass execution time
- Output shape consistency
- Note: Numerical values differ due to different random initializations and implementation details

## Model Architecture Details

### TFT Components

The implementation follows the architecture from "Temporal Fusion Transformers for Interpretable Multi-Horizon Time-Series Forecasting" (Lim et al., 2021):

1. **Static Encoder Branch**: Processes time-invariant features through:
   - Variable selection (`vs`) - weights for static variables
   - Enriched embedding (`ve`) - for LSTM initialization
   - Context vector (`vc`) - conditions temporal processing

2. **Variable Selection Networks**: Applied to both past and future inputs to learn time-varying importance weights

3. **Temporal Fusion Decoder**: Combines:
   - Past encoder LSTM (processes historical data)
   - Future decoder LSTM (processes known future inputs)
   - Temporal attention mechanism (attends over encoder outputs)
   - Static enrichment (incorporates static context)

### Configuration

Model behavior is controlled via `TFTConfig`:
- `num_static`: Number of static features
- `num_past`: Number of past dynamic features
- `num_future`: Number of future known features
- `d_model`: Hidden dimension size
- `d_hidden`: GRN/LSTM hidden dimension (default: 4 × d_model)
- `num_layers`: Number of LSTM layers
- `past_len`: Encoder sequence length (observation window)
- `future_len`: Decoder sequence length (forecast horizon)
- `num_quantiles`: Number of quantile levels for prediction intervals
- `dropout`: Dropout probability

## Performance Comparison Results

As of the latest comparison, here are the results for a sample configuration (d_model=32, past_len=24, future_len=12, batch_size=4):

| Metric | PyTorch | JAX |
|--------|---------|-----|
| Parameters | 142,663 | 143,463 |
| Forward Pass Time | ~0.02s | ~0.41s (includes JIT compilation) |
| Output Shapes Match | ✓ | ✓ |

**Notes:**
- The JAX timing includes JIT compilation overhead on the first run. Subsequent runs would be faster.
- The JAX implementation currently uses a single LSTM layer (matching the PyTorch default of num_layers=1).
- Small differences in parameter count arise from implementation details of LSTMCell vs LSTM layer and bias handling.
- Numerical outputs differ due to different random initializations and floating-point handling between frameworks.

## References

Lim, B., Zohren, S., & Roberts, S. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time-Series Forecasting. International Journal of Forecasting, 37(3), 1748-1764.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original TFT paper authors: Bryan Lim, Søren Zohren, and Stephen Roberts
- PyTorch, JAX, Flax, and Gradio communities for excellent deep learning and UI frameworks