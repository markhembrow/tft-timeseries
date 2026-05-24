"""Download, clean, and prepare AEMO NEM12 NSW electricity demand data for TFT training.

This script:
1. Downloads monthly AEMO aggregated price/demand CSVs from AEMO NemWeb
2. Combines all months into a single DataFrame
3. Resamples from 5-min → 30-min intervals
4. Creates cyclical time features, lag features
5. Splits into train/test sets by time
6. Creates windowed arrays compatible with TimeSeriesDataset
7. Saves processed data as pickle files
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "aemo"
PROCESSED_DIR = DATA_DIR / "processed"


def download_aemo_files(years=(2022, 2023), regions=("NSW1",)):
    """Download monthly AEMO price/demand CSVs."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for year in years:
        for month in range(1, 13):
            for region in regions:
                fname = f"{year}{month:02d}.csv"
                fpath = DATA_DIR / fname
                if fpath.exists():
                    print(f"  Skipping {fname} (already exists)")
                    continue
                    
                url = (
                    f"https://www.aemo.com.au/aemo/data/nem/priceanddemand/"
                    f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"
                )
                print(f"  Downloading {fname}...")
                try:
                    urllib.request.urlretrieve(url, fpath)
                except Exception as e:
                    print(f"    Failed: {e}")
                    if fpath.exists():
                        fpath.unlink()
                    

def load_and_combine():
    """Combine all monthly CSVs into a single DataFrame with 30-min resampling."""
    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    
    print(f"Loading {len(csvs)} CSV files...")
    dfs = []
    for f in csvs:
        # Skip first row (header in AEMO format)
        df = pd.read_csv(
            f, skiprows=1, header=None, 
            names=["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE"]
        )
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    df = (
        df.sort_values("SETTLEMENTDATE")
          .drop_duplicates(subset=["SETTLEMENTDATE"])
          .set_index("SETTLEMENTDATE")
    )
    df = df[["TOTALDEMAND", "RRP"]]
    
    print(f"  Raw rows: {len(df):,}")
    print(f"  Date range: {df.index.min()} → {df.index.max()}")
    print(f"  Missing values: {df.isnull().sum().to_dict()}")
    
    # Resample to 30-min intervals
    df_30 = df.resample("30T").mean()
    
    # Fill gaps
    df_30 = df_30.interpolate(method="linear", limit=2)
    df_30 = df_30.fillna(method="ffill", limit=4)
    df_30 = df_30.fillna(method="bfill", limit=4)
    
    print(f"  After resampling: {len(df_30):,}")
    print(f"  Remaining NaN: {df_30.isnull().sum().to_dict()}")
    
    return df_30


def engineer_features(df_30):
    """Add cyclic time features and lag variables."""
    df = df_30.copy()
    
    # Time components
    df["hour"] = df.index.hour + df.index.minute / 60.0
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    
    # Lag features (48 steps = 24 hours at 30-min intervals)
    df["demand_lag1"] = df["TOTALDEMAND"].shift(1)
    df["demand_lag48"] = df["TOTALDEMAND"].shift(48)
    df["rrp_lag1"] = df["RRP"].shift(1)
    
    # Drop NaN rows created by lags
    df = df.dropna()
    
    print(f"  After feature engineering: {len(df):,}")
    
    return df


def create_windows(df, T=48, H=24, train_frac=0.8):
    """Create windowed arrays compatible with TimeSeriesDataset."""
    
    target_col = "TOTALDEMAND"
    past_cols = ["demand_lag1", "demand_lag48", "rrp_lag1", "RRP"]
    known_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    
    # Train/test split (time-based)
    split_idx = int(len(df) * train_frac)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # Scale features
    scaler_past = StandardScaler().fit(train[past_cols].values)
    scaler_target = StandardScaler().fit(train[[target_col]].values)
    
    train_past = scaler_past.transform(train[past_cols].values)
    test_past = scaler_past.transform(test[past_cols].values)
    train_target = scaler_target.transform(train[[target_col]].values)
    test_target = scaler_target.transform(test[[target_col]].values)
    train_known = train[known_cols].values  # Calendar features don't need scaling
    test_known = test[known_cols].values
    
    # Create windows
    def make_windows(features, known, target, T, H):
        n = len(target)
        n_windows = max(0, n - T - H + 1)
        past = np.zeros((n_windows, T, features.shape[1]))
        known_w = np.zeros((n_windows, H, known.shape[1]))
        tgt = np.zeros((n_windows, H, 1))
        for w in range(n_windows):
            past[w] = features[w:w+T]
            known_w[w] = known[w+T:w+T+H]
            tgt[w] = target[w+T:w+T+H]
        return past, known_w, tgt
    
    train_past_w, train_known_w, train_tgt_w = make_windows(
        train_past, train_known, train_target, T, H
    )
    test_past_w, test_known_w, test_tgt_w = make_windows(
        test_past, test_known, test_target, T, H
    )
    
    # Static features (constant for single region)
    train_static = np.zeros((len(train_tgt_w), 1))
    test_static = np.zeros((len(test_tgt_w), 1))
    
    print(f"\n  Train windows: {train_tgt_w.shape[0]}")
    print(f"  Test windows:  {test_tgt_w.shape[0]}")
    print(f"  Train target range (scaled): {train_tgt_w.min():.2f} to {train_tgt_w.max():.2f}")
    print(f"  Test target range (scaled):  {test_tgt_w.min():.2f} to {test_tgt_w.max():.2f}")
    
    return (
        train_static, train_past_w, train_known_w, train_tgt_w,
        test_static, test_past_w, test_known_w, test_tgt_w,
        scaler_target, scaler_past
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare AEMO data for TFT training.")
    parser.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    parser.add_argument("--region", default="NSW1")
    parser.add_argument("--T", type=int, default=48, help="Past window (default 48 = 24h)")
    parser.add_argument("--H", type=int, default=24, help="Forecast horizon (default 24 = 12h)")
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()
    
    print("=" * 60)
    print("AEMO NSW Electricity Demand Data Preparation")
    print("=" * 60)
    
    # 1. Download data if needed
    print("\n1. Downloading AEMO files...")
    download_aemo_files(args.years, [args.region])
    
    # 2. Load and resample
    print("\n2. Loading and resampling to 30-min...")
    df = load_and_combine()
    
    # 3. Feature engineering
    print("\n3. Engineering features...")
    df = engineer_features(df)
    
    # 4. Create windows
    print("\n4. Creating windows...")
    (train_static, train_past, train_known, train_target,
     test_static, test_past, test_known, test_target,
     scaler_target, scaler_past) = create_windows(df, args.T, args.H, args.train_frac)
    
    # 5. Save
    print("\n5. Saving processed data...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(PROCESSED_DIR / "aemo_nsw_train.pkl", "wb") as f:
        pickle.dump({
            "static": train_static, "past": train_past,
            "known": train_known, "target": train_target,
            "scaler_target": scaler_target, "scaler_past": scaler_past,
        }, f)
    
    with open(PROCESSED_DIR / "aemo_nsw_test.pkl", "wb") as f:
        pickle.dump({
            "static": test_static, "past": test_past,
            "known": test_known, "target": test_target,
            "scaler_target": scaler_target, "scaler_past": scaler_past,
        }, f)
    
    print(f"\nSaved to {PROCESSED_DIR}/")
    print(f"  aemo_nsw_train.pkl: {train_target.shape[0]} windows")
    print(f"  aemo_nsw_test.pkl:  {test_target.shape[0]} windows")
    print("\n✓ Done! Ready for TFT training.")


if __name__ == "__main__":
    main()
