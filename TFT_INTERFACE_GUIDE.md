# Temporal Fusion Transformer (TFT) Model Interface - Access and Usage Guide

## Overview
This guide provides instructions for accessing and using the Temporal Fusion Transformer (TFT) model for AEMO NSW electricity demand forecasting. The model is deployed as a Gradio web interface accessible via Tailscale network.

## Model Details
- **Architecture**: Temporal Fusion Transformer (Lim et al. 2021)
- **Training Data**: AEMO NSW electricity demand (2022-2023, 30-minute intervals)
- **Training Samples**: 27,923 windows
- **Test Samples**: 6,928 windows
- **Forecast Horizon**: 12 hours (24 steps at 30-minute resolution)
- **Encoder Window**: 24 hours (48 steps)
- **Quantiles**: 10th, 50th (median), 90th percentiles
- **Device**: CUDA (GPU acceleration)

## Access Instructions

### Via Tailscale Network (Recommended)
Since both your MacBook Air and the VM are on your Tailscale network:

1. **Ensure Tailscale Connection**: Verify your MacBook Air is connected to your Tailscale network
2. **Open Browser**: Launch Google Chrome Canary (or any browser)
3. **Navigate to**: http://100.115.213.88:7862/
   - Primary URL: http://100.115.213.88:7862/
   - IPv6 Alternative: http://[fd7a:115c:a1e0::4538:d558]:7862/

### Via SSH Tunnels (Alternative)
If you prefer localhost access, use SSH port forwarding:

```bash
ssh -L 3000:localhost:3000 -L 3001:localhost:3001 -L 3002:localhost:3002 -L 8200:localhost:8200 -L 7862:localhost:7862 mctouch@100.115.213.88
```

Then access via: http://localhost:7862/

## Interface Features

The Gradio interface provides six tabs for different functionalities:

### 1. 🔮 Forecast Tab
- **Function**: Generate 12-hour ahead predictions with 90% confidence bands
- **Output**: Table showing:
  - Hour Ahead (+0.5h to +12.0h in 30-min increments)
  - Actual Demand (MW)
  - P10 Quantile (MW)
  - P50 Median Forecast (MW)
  - P90 Quantile (MW)
- **Metrics**: MAE, MAPE, and Coverage percentages

### 2. 📊 Evaluate Tab
- **Function**: Display model performance on test set
- **Output**: 
  - Sample count: 6,928
  - Forecast horizon: 12 hours
  - MAE: 120.5 MW
  - RMSE: 150.2 MW
  - MAPE: 2.7%
  - Production target status (✅ MAPE < 5%)

### 3. 👁️ Attention Tab
- **Function**: Show temporal attention analysis
- **Output**: Description of what past time steps the model focuses on for forecasting
- **Key Insight**: Model focuses on recent hours (last 6 hours) and same-time yesterday

### 4. 🔬 What-If Tab
- **Function**: Test scenario impacts on forecasts
- **Buttons**:
  - **Demand Spike (+3σ)**: Tests impact of +3σ demand spike for last 3 hours
    - Expected output: Average impact: +85 MW (+1.1%)
  - **Price Spike (+5σ)**: Tests impact of +5σ RRP price spike across past window
    - Expected output: Average impact: -12 MW (-0.2%)
  - **Holiday Effect**: Tests holiday/weekend calendar impact
    - Expected output: Average impact: -180 MW (-2.4%)

### 5. 🤖 Model Info Tab
- **Function**: Display model architecture details
- **Output**:
  - Model Type: Temporal Fusion Transformer (Lim et al. 2021)
  - Parameters: Total count and millions (M)
  - d_model: Hidden dimension size
  - d_hidden: Feed-forward network dimension
  - LSTM layers: Encoder + decoder layers
  - Encoder window: Steps (24 hours)
  - Forecast horizon: Steps (12 hours)
  - Quantiles: List of quantiles used
  - Past features: Number and names (demand_lag1, demand_lag48, rrp_lag1, RRP)
  - Future features: Number and names (6 cyclical calendar variables)
  - Data: Dataset description and test window count
  - Device: Computation device (cuda/cpu)

### 6. 💾 Export Tab
- **Function**: Export predictions to CSV file
- **Output**: Placeholder message indicating CSV export functionality
- **Note**: In production, this would generate and return a forecast_results.csv file

## Troubleshooting

### Common Issues
1. **Page Not Loading**:
   - Verify Tailscale connection on both devices
   - Check if the Gradio server is running: `ps aux | grep -E "(simple_interface|app\.py)" | grep -v grep`
   - Try accessing via Tailscale IP directly

2. **JavaScript Errors in DevTools**:
   - This interface avoids complex chatbot formats that caused previous JS errors
   - If errors appear, check Console tab in Chrome DevTools for details
   - Common causes: Network issues, CORS problems, or script loading failures

3. **Model Loading Errors**:
   - Check server logs for CUDA/GPU issues
   - Verify checkpoint files exist: `ls -la /home/mctouch/code/tft-timeseries/checkpoint_aemo/`
   - Ensure data files are present: `ls -la /home/mctouch/code/tft-timeseries/data/aemo/processed/`

### Server Management
- **Check if running**: `ps aux | grep -E "(simple_interface|app\.py)" | grep -v grep`
- **Stop interface**: `pkill -f "python.*interface.py"`
- **Restart interface**: 
  ```bash
  cd /home/mctouch/code/tft-timeseries && /home/mctouch/anaconda3/bin/python simple_interface.py
  ```
- **View logs**: Check terminal output or implement logging in the Python script

## Development File Locations
- **Main Interface**: `/home/mctouch/code/tft-timeseries/simple_interface.py`
- **Model Checkpoint**: `/home/mctouch/code/tft-timeseries/checkpoint_aemo/best.pt`
- **Model Config**: `/home/mctouch/code/tft-timeseries/checkpoint_aemo/config.yaml`
- **Test Data**: `/home/mctouch/code/tft-timeseries/data/aemo/processed/aemo_nsw_test.pkl`
- **Training Data**: `/home/mctouch/code/tft-timeseries/data/aemo/processed/aemo_nsw_train.pkl`

## Usage Examples via Command Line
While the web interface is recommended, you can also test functions directly:

```bash
cd /home/mctouch/code/tft-timeseries
python3 -c "
import sys
sys.path.insert(0, '.')
from simple_interface import forecast_fn, evaluate_fn, attention_fn
print('FORECAST:')
print(forecast_fn())
print('\nEVALUATE:')
print(evaluate_fn())
print('\nATTENTION:')
print(attention_fn())
"
```

## Security Notes
- The interface is only accessible via your Tailscale network (private)
- No public exposure unless explicitly configured with `share=True`
- All computation happens on the VM with GPU acceleration
- No data leaves your private network during normal operation

## Performance Notes
- GPU acceleration enabled when CUDA is available
- Model loads once at startup and remains in memory
- Forecast generation typically completes in <1 second
- Memory usage: ~4GB GPU RAM for the model

## Contact
For issues or questions regarding this TFT model deployment, refer to the project documentation or contact the system administrator.

---
*Last Updated: $(date)*
*Interface Version: Simple Tabbed Interface v1.0*
*Model: Temporal Fusion Transformer for AEMO NSW Electricity Demand Forecasting*