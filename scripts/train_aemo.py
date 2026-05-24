"""Train TFT on AEMO NSW electricity demand data.

Usage:
    python scripts/train_aemo.py  [--epochs 20] [--checkpoint-dir ./checkpoint_aemo]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tft_timeseries.model import TFTConfig, TFTModel
from tft_timeseries.losses import QuantileLoss
from tft_timeseries.data import scale_data, TimeSeriesDataset


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("tft.train_aemo")


def load_dataset(split, batch_size=64):
    """Load processed AEMO pickle into TimeSeriesDataset."""
    pkl_path = Path(__file__).resolve().parents[1] / "data" / "aemo" / "processed" / f"aemo_nsw_{split}.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    ds = TimeSeriesDataset(
        static=data["static"],
        past=data["past"],
        known=data["known"],
        target=data["target"],
        past_len=data["past"].shape[1],
        future_len=data["target"].shape[1],
    )
    
    def collate(batch):
        statics, pasts, knowns, targets = zip(*batch)
        return (
            torch.stack([torch.from_numpy(s).float() for s in statics]),
            torch.stack([torch.from_numpy(p).float() for p in pasts]),
            torch.stack([torch.from_numpy(k).float() for k in knowns]),
            torch.stack([torch.from_numpy(t).float() for t in targets]),
        )
    
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        collate_fn=collate,
    )
    return loader, data["scaler_target"], data["scaler_past"]


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    loss_fn: QuantileLoss,
    scaler_target,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
    ckpt_dir: Path | None = None,
) -> dict[str, list[float]]:
    """Full TFT training loop with quantile loss and TensorBoard logging."""
    tb_dir = (ckpt_dir / "tensorboard") if ckpt_dir else Path("./logs/tensorboard")
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_hparams({
        "d_model": model.config.d_model, "d_hidden": model.config.d_hidden,
        "num_layers": model.config.num_layers, "dropout": model.config.dropout,
        "batch_size": train_loader.batch_size, "lr": lr, "epochs": epochs,
    }, {})

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss, tr_batches = 0.0, 0
        for xs, xp, xf, yt in train_loader:
            xs, xp, xf, yt = xs.to(device), xp.to(device), xf.to(device), yt.to(device)
            opt.zero_grad()
            out = model(xs, xp, xf)
            loss = loss_fn(out["quantiles"], yt.squeeze(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item()
            tr_batches += 1

        avg_train = tr_loss / max(tr_batches, 1)
        history["train_loss"].append(avg_train)
        writer.add_scalar("Loss/train", avg_train, epoch)

        # --- Validate ---
        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for xs, xp, xf, yt in val_loader:
                xs, xp, xf, yt = xs.to(device), xp.to(device), xf.to(device), yt.to(device)
                out = model(xs, xp, xf)
                loss = loss_fn(out["quantiles"], yt.squeeze(-1))
                val_loss += loss.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        history["val_loss"].append(avg_val)
        writer.add_scalar("Loss/val", avg_val, epoch)
        writer.add_scalar("Loss/gap", avg_train - avg_val, epoch)

        logger.info("epoch %3d/%3d  train_loss=%.4f  val_loss=%.4f",
                     epoch, epochs, avg_train, avg_val)

        # --- Checkpoint ---
        if ckpt_dir and avg_val < best_val:
            best_val = avg_val
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            cfg = model.config
            (ckpt_dir / "config.yaml").write_text(
                f"num_static:   {cfg.num_static}\n"
                f"num_past:     {cfg.num_past}\n"
                f"num_future:   {cfg.num_future}\n"
                f"d_model:      {cfg.d_model}\n"
                f"d_hidden:     {cfg.d_hidden}\n"
                f"past_len:     {cfg.past_len}\n"
                f"future_len:   {cfg.future_len}\n"
                f"num_quantiles: {cfg.num_quantiles}\n"
                f"quantiles:    {cfg.quantiles}\n"
                f"dropout:      {cfg.dropout}\n"
            )
            logger.info("  saved best checkpoint -> %s (val_loss=%.4f)", ckpt_path, best_val)

    writer.flush()
    writer.close()
    logger.info("TensorBoard logs saved to %s", tb_dir)
    logger.info("Run: tensorboard --logdir %s --port 6006", tb_dir)

    return history


def evaluate(model, val_loader, scaler_target, device="cpu"):
    """Calculate MAE on unscaled target."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xs, xp, xf, yt in val_loader:
            xs, xp, xf, yt = xs.to(device), xp.to(device), xf.to(device), yt.to(device)
            out = model(xs, xp, xf)
            q50 = out["quantiles"][:, :, 1]  # median (index 1 of [0.1, 0.5, 0.9])
            
            # Unscale
            import numpy as np
            for b in range(yt.shape[0]):
                pred_flat = q50[b].cpu().numpy().reshape(-1, 1)
                true_flat = yt[b].cpu().numpy().reshape(-1, 1)
                
                pred_unscaled = scaler_target.inverse_transform(pred_flat)
                true_unscaled = scaler_target.inverse_transform(true_flat)
                
                preds.extend(pred_unscaled.flatten())
                trues.extend(true_unscaled.flatten())
    
    import numpy as np
    preds, trues = np.array(preds), np.array(trues)
    mae = np.mean(np.abs(preds - trues))
    mape = np.mean(np.abs((trues - preds) / (np.abs(trues) + 1e-8))) * 100
    
    logger.info(f"Evaluation: MAE={mae:.2f} MW, MAPE={mape:.2f}%")
    return mae, mape


def main():
    parser = argparse.ArgumentParser(description="Train TFT on AEMO NSW demand data.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoint_aemo")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    
    logger.info("Loading AEMO NSW datasets...")
    train_loader, scaler_target, scaler_past = load_dataset("train", batch_size=args.batch_size)
    val_loader, _, _ = load_dataset("test", batch_size=args.batch_size)
    
    # Configure model based on data shapes
    T = 48  # 24 hours * 2 (30-min intervals)
    H = 24  # 12 hours * 2
    num_past = 4   # demand_lag1, demand_lag48, rrp_lag1, RRP
    num_future = 6 # hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos
    
    cfg = TFTConfig(
        num_static=1, num_past=num_past, num_future=num_future,
        d_model=64, d_hidden=256, num_layers=2,
        past_len=T, future_len=H, num_quantiles=3,
        quantiles=[0.1, 0.5, 0.9], dropout=0.1,
    )
    
    model = TFTModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")
    
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training on {device}...")
    history = train(
        model, train_loader, val_loader,
        QuantileLoss(cfg.quantiles),
        scaler_target,
        epochs=args.epochs, lr=args.lr,
        device=device, ckpt_dir=ckpt_dir,
    )
    
    final_val = history["val_loss"][-1]
    logger.info(f"Training complete. Final val loss: {final_val:.4f}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    mae, mape = evaluate(model, val_loader, scaler_target, device)
    
    logger.info(f"Final results: MAE={mae:.2f} MW, MAPE={mape:.2f}%, Val Loss={final_val:.4f}")
    
    # Compare against naive baseline (predict last value)
    logger.info(f"Target: MAPE < 5% and MAE < 200 MW for production use")


if __name__ == "__main__":
    main()
