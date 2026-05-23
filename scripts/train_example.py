"""Train TFT on synthetic sinusoidal + trend data.

Usage:
    python scripts/train_example.py  [--epochs 20] [--checkpoint-dir ./ckpt]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tft_timeseries.model import (
    TFTConfig, TFTModel,
    StaticCovariateEncoder, GatedResidualNetwork, VariableSelectionNetwork,
    TemporalFusionDecoder,
)
from tft_timeseries.losses import QuantileLoss, quantile_loss
from tft_timeseries.train import TFTDataModule


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("tft.train_example")


# ── 1. Synthetic data generation ──────────────────────────────────────────────

def make_synthetic(
    n_series: int = 32,
    T:        int = 120,
    H:        int = 24,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (static, past, known, target) arrays for a toy problem.

    each series index = one shop + product pair, with:
      trend + seasonal(7-day) + noise as the target.
    """
    rng = np.random.default_rng(seed)

    # static: [log-series-index, log-month-sin]  (S_stat=2)
    static = np.zeros((n_series, 2))
    static[:, 0] = np.log(rng.uniform(0.5, 5.0, n_series))  # scale prior

    # num_past=4 dynamic history vars: price, promo, day_sin, day_cos
    num_past = 4
    num_targets = 1

    # Build per-series trend + seasonality as the target
    series_scale = rng.uniform(5, 50, n_series)
    trend_slope  = rng.uniform(0.01, 0.15, n_series)

    past   = rng.normal(0, 1, (n_series, T, num_past)).astype(np.float32)
    known  = np.zeros((n_series, T + H, 2)).astype(np.float32)   # calendar vars (sin/cos)
    target = np.zeros((n_series, T, num_targets)).astype(np.float32)

    for s in range(n_series):
        t = np.arange(T + H)
        day_sin = np.sin(2 * np.pi * t / 7.0)
        day_cos = np.cos(2 * np.pi * t / 7.0)
        # known = calendar features for encoder + decoder
        known[s, :, 0] = day_sin
        known[s, :, 1] = day_cos
        # inject one calendar signal into past as feature 3 (day_sin)
        past[s, :, 2] = day_sin[:T]
        past[s, :, 3] = day_cos[:T]
        # target is linear trend + weekly season + noise
        noise = rng.normal(0, 2.0, T)
        target[s, :, 0] = series_scale[s] * day_sin[:T] + trend_slope[s] * np.arange(T) + noise

    return static, past, known, target


def train_val_test_split(
    arrays: list[np.ndarray],
    train_frac: float = 0.7,
    val_frac:   float = 0.15,
) -> tuple[list, list, list]:
    split_pt = int(len(arrays[0]) * train_frac)
    val_pt   = int(len(arrays[0]) * (train_frac + val_frac))
    train = [a[:split_pt] for a in arrays]
    val   = [a[split_pt:val_pt] for a in arrays]
    test  = [a[val_pt:] for a in arrays]
    return train, val, test


# ── 2. Training loop ──────────────────────────────────────────────────────────

def train(
    model:      nn.Module,
    dm:         TFTDataModule,
    loss_fn:    QuantileLoss,
    epochs:     int = 20,
    lr:         float = 1e-3,
    device:     str = "cpu",
    ckpt_dir:   Path | None = None,
) -> dict[str, list[float]]:
    """Full TFT training loop worth."""
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    history = dict(train_loss=[], val_loss=[])
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ── train ──────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        tr_batches = 0
        for xs, xp, xf, yt in dm.train_loader():
            xs, xp, xf, yt = xs.to(device), xp.to(device), xf.to(device), yt.to(device)
            opt.zero_grad()
            out = model(xs, xp, xf)
            # target is the last `future_len` steps of historical target
            y_true = yt[:, -dm.future_len:, :]
            loss = loss_fn(out["quantiles"], y_true.squeeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item()
            tr_batches += 1

        history["train_loss"].append(tr_loss / max(tr_batches, 1))

        # ── validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xs, xp, xf, yt in dm.val_loader():
                xs, xp, xf, yt = xs.to(device), xp.to(device), xf.to(device), yt.to(device)
                out     = model(xs, xp, xf)
                y_true  = yt[:, -dm.future_len:, :]
                v_loss  = loss_fn(out["quantiles"], y_true.squeeze(-1))
                val_loss += v_loss.item()
                val_batches += 1

        history["val_loss"].append(val_loss / max(val_batches, 1))

        logger.info(
            "epoch %3d/%3d  train_loss=%.4f  val_loss=%.4f",
            epoch, epochs, history["train_loss"][-1], history["val_loss"][-1],
        )

        # ── checkpoint ─────────────────────────────────────────────────────────
        if ckpt_dir and history["val_loss"][-1] < best_val:
            best_val = history["val_loss"][-1]
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            (ckpt_dir / "config.yaml").write_text(
                f"num_static: {dm.num_static}\n"
                f"num_past: {dm.num_past}\n"
                f"num_future: {dm.num_future}\n"
                f"d_model: 32\nd_hidden: 64\n"
                f"past_len: {dm.past_len}\nfuture_len: {dm.future_len}\n"
                f"num_quantiles: 3\nquantiles: [0.1, 0.5, 0.9]\ndropout: 0.1\n"
            )
            logger.info("  saved best checkpoint → %s", ckpt_path)

    return history


# ── 3. Entry-point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TFT on synthetic data.")
    parser.add_argument("--epochs",  type=int,   default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )

    T, H = 24, 12                           # past_len, forecast horizon

    # ── data ───────────────────────────────────────────────────────────────────
    (s_tr, p_tr, k_tr, t_tr), (s_val, p_val, k_val, t_val), (s_te, p_te, k_te, t_te) \
        = train_val_test_split(make_synthetic(n_series=32, T=T, H=H))
    dm = TFTDataModule(
        train_static=s_tr,  train_past=p_tr,  train_known=k_tr,  train_target=t_tr,
        val_static=s_val,   val_past=p_val,   val_known=k_val,   val_target=t_val,
        test_static=s_te,   test_past=p_te,   test_known=k_te,   test_target=t_te,
        past_len=T, future_len=H, batch_size=args.batch_size,
    )

    # ── model ──────────────────────────────────────────────────────────────────
    cfg = TFTConfig(
        num_static=2, num_past=4, num_future=2,
        d_model=32, num_layers=1,
        past_len=T, future_len=H, num_quantiles=3,
        quantiles=[0.1, 0.5, 0.9], dropout=0.1,
    )
    model = TFTModel(cfg)
    logger.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    # ── train ──────────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = train(model, dm, QuantileLoss(cfg.quantiles),
                    epochs=args.epochs, lr=args.lr,
                    device=device, ckpt_dir=ckpt_dir)
    final_loss = history["val_loss"][-1]
    logger.info("Training complete. Final val loss: %.4f", final_loss)
    assert final_loss < 0.5, f"Final val loss {final_loss:.4f} >= 0.5 — training did not converge"


if __name__ == "__main__":
    main()
