"""Query the trained TFT model for AEMO NSW electricity demand forecasts.

Usage:
    # Quick forecast from recent data
    python scripts/query_model.py --forecast --horizon 12

    # Evaluate on full test set
    python scripts/query_model.py --evaluate

    # Analyze attention weights  
    python scripts/query_model.py --attention --sample-id 0

    # What-if scenarios
    python scripts/query_model.py --what-if temperature_spike
    python scripts/query_model.py --what-if price_spike
    python scripts/query_model.py --what-if holiday
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tft_timeseries.model import TFTModel, TFTConfig
from tft_timeseries.inference import TFTInferenceEngine


CHECKPOINT_DIR = Path("./checkpoint_aemo")


def load_model():
    """Load trained TFT model and scalers."""
    if not (CHECKPOINT_DIR / "best.pt").exists():
        print(f"Error: No checkpoint found at {CHECKPOINT_DIR / 'best.pt'}")
        print("Run training first: python scripts/train_aemo.py --epochs 50 --checkpoint-dir ./checkpoint_aemo")
        sys.exit(1)

    engine = TFTInferenceEngine(
        checkpoint_path=CHECKPOINT_DIR / "best.pt",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Load scalers
    with open(Path(__file__).resolve().parents[1] / "data" / "aemo" / "processed" / "aemo_nsw_test.pkl", "rb") as f:
        test_data = pickle.load(f)
    
    return engine, test_data, test_data["scaler_target"], test_data["scaler_past"]


def forecast_next_12h(engine, test_data, scaler_target):
    """Forecast next 12 hours from the most recent test sample."""
    device = engine.model.device
    cfg = engine.model.config
    
    # Get last windows from test set
    past = test_data["past"]
    known = test_data["known"]
    target = test_data["target"]
    static = test_data["static"]
    
    # Use the last window as our query
    idx = -1
    xs = torch.from_numpy(static[idx:idx+1]).float().to(device)
    xp = torch.from_numpy(past[idx:idx+1]).float().to(device)
    xf = torch.from_numpy(known[idx:idx+1]).float().to(device)
    
    with torch.no_grad():
        out = engine.model(xs, xp, xf)
    
    # Extract quantile predictions
    q_pred = out["quantiles"][0].cpu().numpy()  # (H, Q)
    q10, q50, q90 = q_pred[:, 0], q_pred[:, 1], q_pred[:, 2]
    
    # Unscale predictions
    q10_unscaled = scaler_target.inverse_transform(q10.reshape(-1, 1)).flatten()
    q50_unscaled = scaler_target.inverse_transform(q50.reshape(-1, 1)).flatten()
    q90_unscaled = scaler_target.inverse_transform(q90.reshape(-1, 1)).flatten()
    
    # Get actual values for comparison
    y_true = target[idx].flatten()
    y_true_unscaled = scaler_target.inverse_transform(y_true.reshape(-1, 1)).flatten()
    
    print("\n" + "=" * 80)
    print("TFT Forecast: Next 12 Hours (30-min intervals)")
    print("=" * 80)
    print(f"{'Hours Ahead':>12} | {'Actual (MW)':>12} | {'P10 (MW)':>10} | {'P50 (MW)':>10} | {'P90 (MW)':>10}")
    print("-" * 80)
    
    for h in range(cfg.future_len):
        actual = f"{y_true_unscaled[h]:10.2f}"
        lower = f"{q10_unscaled[h]:10.2f}"
        median = f"{q50_unscaled[h]:10.2f}"
        upper = f"{q90_unscaled[h]:10.2f}"
        print(f"  {h*0.5 + 0.5:10.1f} | {actual} | {lower} | {median} | {upper}")
    
    # Calculate metrics
    mae = np.mean(np.abs(q50_unscaled - y_true_unscaled))
    mape = np.mean(np.abs((y_true_unscaled - q50_unscaled) / (np.abs(y_true_unscaled) + 1e-8))) * 100
    coverage = np.mean((y_true_unscaled >= q10_unscaled) & (y_true_unscaled <= q90_unscaled)) * 100
    
    print("-" * 80)
    print(f"\nMetrics: MAE={mae:.2f} MW, MAPE={mape:.2f}%, 10-90 Coverage={coverage:.1f}%")


def evaluate_test_set(engine, test_data, scaler_target):
    """Evaluate on the full test set."""
    model = engine.model
    device = model.device
    cfg = model.config
    
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(test_data["static"]).float(),
            torch.from_numpy(test_data["past"]).float(),
            torch.from_numpy(test_data["known"]).float(),
            torch.from_numpy(test_data["target"]).float(),
        ),
        batch_size=256,
        shuffle=False,
    )
    
    model.eval()
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            xs, xp, xf, yt = [t.to(device) for t in batch]
            out = model(xs, xp, xf)
            q50 = out["quantiles"][:, :, 1]  # median
            
            for b in range(yt.shape[0]):
                pred_flat = q50[b].cpu().numpy().reshape(-1, 1)
                true_flat = yt[b].cpu().numpy().reshape(-1, 1)
                pred_unscaled = scaler_target.inverse_transform(pred_flat).flatten()
                true_unscaled = scaler_target.inverse_transform(true_flat).flatten()
                all_preds.extend(pred_unscaled)
                all_trues.extend(true_unscaled)
    
    preds = np.array(all_preds)
    trues = np.array(all_trues)
    mae = np.mean(np.abs(preds - trues))
    mape = np.mean(np.abs((trues - preds) / (np.abs(trues) + 1e-8))) * 100
    rmse = np.sqrt(np.mean((preds - trues)**2))
    
    # Naive baseline (predict last value)
    naive_preds = []
    naive_trues = []
    with torch.no_grad():
        for batch in val_loader:
            xs, xp, xf, yt = [t.to(device) for t in batch]
            # Naive: repeat last known demand for all future steps
            last_demand = xp[:, -1, 0]  # demand_lag1 at last time step
            for b in range(yt.shape[0]):
                naive_pred = scaler_target.inverse_transform(
                    last_demand[b].item() * np.ones((cfg.future_len, 1))
                ).flatten()
                naive_true = scaler_target.inverse_transform(yt[b].cpu().numpy().reshape(-1, 1)).flatten()
                naive_preds.extend(naive_pred)
                naive_trues.extend(naive_true)
    
    naive_preds = np.array(naive_preds)
    naive_trues = np.array(naive_trues)
    naive_mae = np.mean(np.abs(naive_preds - naive_trues))
    naive_mape = np.mean(np.abs((naive_trues - naive_preds) / (np.abs(naive_trues) + 1e-8))) * 100
    
    print("\n" + "=" * 60)
    print("TFT Model Evaluation: Full Test Set")
    print("=" * 60)
    print(f"Test samples: {len(test_data['target']):,}")
    print(f"Horizon: {cfg.future_len} steps ({cfg.future_len * 0.5} hours)")
    print()
    print(f"{'Metric':>12} | {'TFT':>12} | {'Naive Baseline':>14} | {'Improvement':>12}")
    print("-" * 60)
    print(f"{'MAE (MW)':>12} | {mae:10.2f} | {naive_mae:12.2f} | {naive_mae - mae:+8.2f}")
    print(f"{'RMSE (MW)':>12} | {rmse:10.2f} | {'─':10s} | {'─':12s}")
    print(f"{'MAPE (%)':>12} | {mape:10.2f} | {naive_mape:10.2f} | {naive_mape - mape:+8.2f}")
    print()
    if mape < 5:
        print("✓ PASS: MAPE < 5% (production target met)")
    elif mape < 10:
        print("⚠ PARTIAL: MAPE between 5-10% (usable for planning)")
    else:
        print("✗ NEEDS WORK: MAPE > 10%")


def analyze_attention(engine, test_data, sample_id=0):
    """Analyze temporal attention weights for a specific test sample."""
    model = engine.model
    device = model.device
    cfg = engine._cfg

    past = test_data["past"][sample_id:sample_id+1]
    known = test_data["known"][sample_id:sample_id+1]
    static = test_data["static"][sample_id:sample_id+1]

    with torch.no_grad():
        xs = torch.from_numpy(static).float().to(device)
        xp = torch.from_numpy(past).float().to(device)
        xf = torch.from_numpy(known).float().to(device)
        out = model(xs, xp, xf)

    attn = out["attn_weights"][0].cpu().numpy()  # (H, T, T)
    attn_mean = attn.mean(axis=-1)  # (H, T)

    print("\n" + "=" * 80)
    print("Temporal Attention Analysis (Sample {})".format(sample_id))
    print("=" * 80)
    print("Horizon: {} steps ({} hours)".format(cfg.future_len, cfg.future_len * 0.5))
    print()
    for h in range(cfg.future_len):
        weights = attn_mean[h]
        most_attended = int(weights.argmax())
        eps = 1e-10
        entropy = -np.sum(weights * np.log(np.abs(weights) + eps))
        hours_ahead = (h + 1) * 0.5
        hours_ago = (cfg.past_len - most_attended) * 0.5
        print("  +{:.1f}h  |  Step-{:3d} ({:5.1f}h ago)  |  entropy {:.2f}".format(
            hours_ahead, most_attended, hours_ago, entropy))


def export_to_csv(engine, test_data, scaler_target, output_path="forecast_results.csv"):
    """Export full test set predictions with quantile bands to CSV."""
    import csv

    model = engine.model
    device = model.device
    cfg = engine._cfg

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(test_data["static"]).float(),
            torch.from_numpy(test_data["past"]).float(),
            torch.from_numpy(test_data["known"]).float(),
            torch.from_numpy(test_data["target"]).float(),
        ),
        batch_size=256, shuffle=False,
    )

    model.eval()
    rows = []
    with torch.no_grad():
        for xs, xp, xf, yt in val_loader:
            xs, xp, xf, yt = (t.to(device) for t in (xs, xp, xf, yt))
            out = model(xs, xp, xf)
            q_pred = out["quantiles"].cpu().numpy()

            for b in range(yt.shape[0]):
                for h in range(cfg.future_len):
                    q10 = scaler_target.inverse_transform(q_pred[b, h, 0].reshape(1, 1)).item()
                    q50 = scaler_target.inverse_transform(q_pred[b, h, 1].reshape(1, 1)).item()
                    q90 = scaler_target.inverse_transform(q_pred[b, h, 2].reshape(1, 1)).item()
                    actual = scaler_target.inverse_transform(yt[b, h].cpu().numpy().reshape(1, 1)).item()
                    rows.append({
                        "sample": b, "horizon_h": (h + 1) * 0.5,
                        "actual_mw": actual, "q10_mw": q10, "q50_mw": q50, "q90_mw": q90,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("\nExported {:,} predictions to {}".format(len(df), output_path))


def what_if(engine, test_data, scenario, scaler_target):
    """What-if: spike demand, price, or simulate holiday by perturbing inputs."""
    model = engine.model; device = model.device; cfg = engine._cfg
    idx = 0
    past = test_data["past"][idx:idx+1].copy()
    known = test_data["known"][idx:idx+1].copy()
    static = test_data["static"][idx:idx+1]

    with torch.no_grad():
        xs = torch.from_numpy(static).float().to(device)
        xp_base = torch.from_numpy(past).float().to(device)
        xf = torch.from_numpy(known).float().to(device)
        q50_base = model(xs, xp_base, xf)["quantiles"][0, :, 1].cpu().numpy()

    xp_mod, xf_mod = xp_base, xf; label = "baseline"
    if scenario == "demand_spike":
        past[:, -6:, 0] += 3.0; xp_mod = torch.from_numpy(past).float().to(device)
        label = "+3σ demand for last 3 hours"
    elif scenario == "price_spike":
        past[:, :, 3] += 5.0; xp_mod = torch.from_numpy(past).float().to(device)
        label = "+5σ RRP throughout past window"
    elif scenario == "holiday":
        known[:, :, 2] = 1.0; known[:, :, 3] = 0.0  # dow_sin=1, dow_cos=0
        xf_mod = torch.from_numpy(known).float().to(device)
        label = "Holiday calendar"

    with torch.no_grad():
        q50_mod = model(xs, xp_mod, xf_mod)["quantiles"][0, :, 1].cpu().numpy()

    q50_b = scaler_target.inverse_transform(q50_base.reshape(-1, 1)).flatten()
    q50_m = scaler_target.inverse_transform(q50_mod.reshape(-1, 1)).flatten()

    print("\n" + "=" * 80)
    print("What-If: {}  ({})".format(scenario, label))
    print("=" * 80)
    print("{:>10}  {:>15}  {:>15}  {:>10}".format("Horizon(h)", "Baseline(MW)", "Scenario(MW)", "Δ MW"))
    print("-" * 80)
    for h in range(cfg.future_len):
        d = q50_m[h] - q50_b[h]
        print("  {:8.1f}  {:12.2f}  {:12.2f}  {:+8.2f}".format(
            (h+1)*0.5, q50_b[h], q50_m[h], d))
    avg = np.mean(q50_m - q50_b)
    print("\nAverage impact: {:+.2f} MW  ({:+.2f}%)".format(avg, avg/7538*100))


def main():
    parser = argparse.ArgumentParser(description="Query the trained TFT model.")
    parser.add_argument("--forecast", action="store_true", help="Quick 12-h forecast")
    parser.add_argument("--evaluate", action="store_true", help="Full test-set eval")
    parser.add_argument("--attention", action="store_true", help="Attention analysis")
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--what-if", type=str, choices=["demand_spike", "price_spike", "holiday"])
    parser.add_argument("--export", action="store_true", help="Export predictions to CSV")
    parser.add_argument("--output", default="forecast_results.csv")
    parser.add_argument("--all", action="store_true", help="Run every analysis")
    args = parser.parse_args()

    engine, test_data, scaler_target, _ = load_model()

    if args.all:
        forecast_next_12h(engine, test_data, scaler_target)
        analyze_attention(engine, test_data, args.sample_id)
        evaluate_test_set(engine, test_data, scaler_target)
        export_to_csv(engine, test_data, scaler_target, args.output)
        what_if(engine, test_data, "demand_spike", scaler_target)
        what_if(engine, test_data, "price_spike", scaler_target)
        what_if(engine, test_data, "holiday", scaler_target)
    else:
        if args.forecast:
            forecast_next_12h(engine, test_data, scaler_target)
        if args.evaluate:
            evaluate_test_set(engine, test_data, scaler_target)
        if args.attention:
            analyze_attention(engine, test_data, args.sample_id)
        if args.export:
            export_to_csv(engine, test_data, scaler_target, args.output)
        if args.what_if:
            what_if(engine, test_data, args.what_if, scaler_target)

    if not any([args.forecast, args.evaluate, args.attention, args.export, args.what_if, args.all]):
        print("Usage:")
        print("  --forecast           Quick 12-hour forecast from last test sample")
        print("  --evaluate           Full test-set eval with baseline comparison")
        print("  --attention          Attention weight analysis")
        print("  --what-if SCENARIO   demand_spike / price_spike / holiday")
        print("  --export             Export predictions to CSV")
        print("  --all                Run all analyses at once")


if __name__ == "__main__":
    main()