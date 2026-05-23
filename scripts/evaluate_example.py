"""Evaluate a trained TFT checkpoint and print a forecast summary.

Usage:
    python scripts/evaluate_example.py  [--checkpoint-dir ./checkpoint]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tft_timeseries.inference import TFTInferenceEngine
from tft_timeseries.model import TFTConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained TFT model.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint",
                        help="Directory containing best.pt and config.yaml")
    parser.add_argument("--n-sample", type=int, default=3,
                        help="How many random series to print forecasts for")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    engine = TFTInferenceEngine(
        checkpoint_path=ckpt_dir / "best.pt",
        config=str(ckpt_dir / "config.yaml"),
    )
    cfg = engine._cfg

    # ── generate a single mini-batch for evaluation ───────────────────────────
    n = args.n_sample
    rng = np.random.default_rng(123)
    static  = rng.normal(0, 1, (n, cfg.num_static)).astype(np.float32)
    past    = rng.normal(0, 1, (n, cfg.past_len, cfg.num_past)).astype(np.float32)
    known   = rng.normal(0, 1, (n, cfg.future_len + cfg.past_len, cfg.num_future))

    # build target for MAE computation
    target  = rng.normal(0, 1, (n, cfg.future_len, 1)).astype(np.float32)

    out = engine.predict_batch(static, past, known)
    q_pred = out["quantiles"]         # (B, H, Q)
    q10, q50, q90 = (q_pred[:, :, i] for i in range(3))

    y_true = target[:, :, 0]

    # ── compute metrics ────────────────────────────────────────────────────────
    mae  = np.mean(np.abs(q50 - y_true))
    mape = np.mean(np.abs((q50 - y_true) / (np.abs(y_true) + 1e-8))) * 100
    coverage = np.mean((y_true >= q10) & (y_true <= q90)) * 100

    print("\n─── Forecast Summary ────")
    print(f"  Samples          : {n}")
    print(f"  Horizon          : {cfg.future_len}")
    print(f"  Quantiles        : {cfg.quantiles}")
    print(f"  MAE (median)     : {mae:.4f}")
    print(f"  MAPE (median)    : {mape:.2f}%")
    print(f"  10-90 coverage   : {coverage:.1f}%")

    # ── per-sample table ────────────────────────────────────────────────────────
    print("\n─ Per-series (first horizon steps) ─")
    print(f"{'Series':>6}  {'Step':>5}  {'True':>8}  {'Q10':>8}  {'Q50':>8}  {'Q90':>8}  "
          f"{'InCI?':>6}")
    sep = "-" * 58
    print(sep)
    for b in range(args.n_sample):
        for h in range(min(5, cfg.future_len)):
            in_ci = "✓" if q10[b, h] <= y_true[b, h] <= q90[b, h] else "✗"
            print(
                f"  {b:4d}  {h:5d}  {y_true[b, h]:8.3f}"
                f"  {q10[b, h]:8.3f}  {q50[b, h]:8.3f}  {q90[b, h]:8.3f}  {in_ci:>6}"
            )
        print(sep)

    # ── save CSV ───────────────────────────────────────────────────────────────
    csv_path = Path(args.checkpoint_dir) / "forecast_results.csv"
    import pandas as pd
    rows = []
    for b in range(n):
        for h in range(cfg.future_len):
            rows.append(dict(
                series=b,
                horizon=h,
                true=y_true[b, h],
                q10=q10[b, h],
                q50=q50[b, h],
                q90=q90[b, h],
            ))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}")


if __name__ == "__main__":
    main()
