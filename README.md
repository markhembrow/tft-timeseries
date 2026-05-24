# Temporal Fusion Transformer (TFT) for Time Series Forecasting

A complete implementation of the Temporal Fusion Transformer (TFT) architecture for interpretable multi-horizon time series forecasting, built with PyTorch. This repository includes the core model, training scripts, test suite, data processing utilities, and a Gradio-based user interface for electricity demand forecasting.

## Overview

The Temporal Fusion Transformer (Lim et al., 2021) combines the strengths of recurrent networks, temporal attention mechanisms, and transformer architectures to provide accurate and interpretable forecasts across multiple time horizons.

This implementation includes:
- Complete TFT architecture with Variable Selection Networks (VSN), Gated Residual Networks (GRN), and Temporal Fusion Decoder
- Configurable hyperparameters via YAML or programmatic interface
- Training scripts for the AEMO NSW electricity demand dataset
- Comprehensive test suite covering model components
- Gradio web interface for forecasting, evaluation, attention analysis, and what-if scenarios
- Accessible via Tailscale VPN for secure remote access

## Repository Structure

```
tft-timeseries/
├── tft_timeseries/                 # Core TFT implementation
│   ├── model.py                    # TFTModel, TFTConfig, and sub-components
│   ├── data.py                     # Data processing utilities
│   └── losses.py                   # QuantileLoss implementation
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
├── simple_interface.py             # Gradio web interface
├── TFT_INTERFACE_GUIDE.md          # Detailed interface guide
└── README.md                       # This file
```

## Key Features

### Custom TFT Implementation

The core implementation in `tft_timeseries/model.py` includes:

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
python -m pytest tests/ -v
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

3. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install gradio pyyaml numpy scikit-learn pandas
```

### Training the Model

To train the TFT model on the AEMO NSW electricity demand dataset:

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

## Test Suite

The test suite validates:
- Component shapes and mathematical properties
- Gradient flow and backpropagation
- Numerical stability (no NaN/Inf values)
- Context handling and feature interactions
- CUDA compatibility (when available)

Key test files:
- `tests/test_model.py`: Model component tests
- `tests/test_data.py`: Data processing tests
- `tests/test_losses.py`: Loss function tests

## Performance and Usage

The trained model provides:
- 12-hour ahead forecasts (24 steps at 30-minute resolution)
- Prediction intervals via quantile outputs (P10, P50, P90)
- Interpretability through attention weights
- Configurable horizons and feature sets

## References

Lim, B., Zohren, S., & Roberts, S. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time-Series Forecasting. International Journal of Forecasting, 37(3), 1748-1764.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original TFT paper authors: Bryan Lim, Søren Zohren, and Stephen Roberts
- PyTorch and Gradio communities for excellent deep learning and UI frameworks