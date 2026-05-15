"""
training/train.py

Full training pipeline for the Underwater Image Enhancement model.

Usage:
    python main.py train
    python main.py train --epochs 100 --batch 8 --lr 0.0005
    python main.py train --resume checkpoints/model_final.pth

What this does:
  - Trains with combined loss: 0.6 × L1 + 0.3 × EdgeLoss + 0.1 × ColorLoss
  - Validates every epoch on EUVP val set (UCIQE + UIQM)
  - LR scheduler: ReduceLROnPlateau on validation UCIQE
  - Saves model_best.pth (best UCIQE) and model_final.pth (last epoch)
  - Periodic checkpoints every SAVE_EVERY epochs
  - CSV log to logs/train_log.csv
  - Training curve plot to results/training_curves.png
"""

import argparse
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import (
    EUVP_TRAIN_A, EUVP_TRAIN_B, EUVP_VAL,
    CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR,
    CHECKPOINT_FINAL, CHECKPOINT_BEST,
    BATCH_SIZE, EPOCHS, LR, SAVE_EVERY, IMAGE_SIZE,
    W_L1, W_EDGE, W_COLOR,
)
from datasets.euvp_dataset import EUVPDataset
from models.cnn_model import EnhancementCNN


def log(message=""):
    print(message, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class EdgeLoss(nn.Module):
    """Sobel-based structural loss — encourages sharpness."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, x):
        gray = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(gray, self.kx, padding=1)
        ey = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


class ColorLoss(nn.Module):
    """Per-channel spatial mean loss — penalises global colour cast."""

    def forward(self, pred, target):
        return F.l1_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_fn    = nn.L1Loss()
        self.edge_fn  = EdgeLoss()
        self.color_fn = ColorLoss()

    def forward(self, pred, target):
        l1    = self.l1_fn(pred, target)
        edge  = self.edge_fn(pred, target)
        color = self.color_fn(pred, target)
        total = W_L1 * l1 + W_EDGE * edge + W_COLOR * color
        return total, {"total": total.item(), "l1": l1.item(),
                       "edge": edge.item(), "color": color.item()}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (inline — no extra import needed during training)
# ─────────────────────────────────────────────────────────────────────────────

def compute_uciqe(img: np.ndarray) -> float:
    import cv2
    img = np.clip(img, 0.0, 1.0)
    img_u8  = (img * 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(img_lab)
    chroma  = np.sqrt(A ** 2 + B ** 2)
    return (0.4680 * np.std(chroma)
            + 0.2745 * (np.max(L) - np.min(L)) / 255.0
            + 0.2576 * np.mean(chroma) / 255.0)


def compute_uiqm(img: np.ndarray) -> float:
    import cv2
    img    = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    rg     = img_u8[:, :, 0].astype(np.float32) - img_u8[:, :, 1].astype(np.float32)
    yb     = (0.5 * (img_u8[:, :, 0].astype(np.float32) +
               img_u8[:, :, 1].astype(np.float32))
              - img_u8[:, :, 2].astype(np.float32))
    uicm   = float(np.std(rg) + np.std(yb))
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    uism   = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1))))
    uiconm = float(np.std(gray))
    return (0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(model, val_loader, device):
    model.eval()
    uciqe_scores, uiqm_scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = model(inp.to(device))
            for i in range(out.size(0)):
                img_np = out[i].cpu().numpy().transpose(1, 2, 0)
                uciqe_scores.append(compute_uciqe(img_np))
                uiqm_scores.append(compute_uiqm(img_np))
    return float(np.mean(uciqe_scores)), float(np.mean(uiqm_scores))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train underwater image enhancement model")
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--batch",      type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=LR)
    p.add_argument("--resume",     type=str,   default=None)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--save-every", type=int,   default=SAVE_EVERY)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    for d in (CHECKPOINTS_DIR, RESULTS_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")
    if device.type == "cuda":
        log(f"GPU    : {torch.cuda.get_device_name(0)}")

    train_ds = EUVPDataset(EUVP_TRAIN_A, EUVP_TRAIN_B, augment=True,  image_size=IMAGE_SIZE)
    val_ds   = EUVPDataset(EUVP_VAL,     None,          augment=False, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    log(f"Train  : {len(train_ds)} images  |  Val: {len(val_ds)} images")

    model = EnhancementCNN().to(device)

    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        log(f"Resumed from {args.resume}")

    log(f"Params : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler_kwargs = dict(
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_kwargs,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_kwargs,
        )
    criterion = CombinedLoss().to(device)

    log_path = os.path.join(LOGS_DIR, "train_log.csv")
    csv_file = open(log_path, "w", newline="")
    writer   = csv.writer(csv_file)
    writer.writerow(["epoch", "train_loss", "val_uciqe", "val_uiqm", "lr"])

    train_losses, val_uciqes, val_uiqms = [], [], []
    best_uciqe = -1.0

    log("\n" + "-" * 60)
    log("Starting training")
    log("-" * 60)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        progress_every = max(1, num_batches // 10)

        for batch_idx, (inp, tar) in enumerate(train_loader, start=1):
            inp, tar = inp.to(device), tar.to(device)
            out  = model(inp)
            loss, _ = criterion(out, tar)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % progress_every == 0 or batch_idx == num_batches:
                avg_loss = epoch_loss / batch_idx
                elapsed = time.time() - t0
                log(
                    f"  Epoch {epoch+1:>4}/{args.epochs}  "
                    f"Batch {batch_idx:>4}/{num_batches}  "
                    f"Avg Loss: {avg_loss:.4f}  "
                    f"({elapsed:.1f}s)"
                )

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        val_uciqe, val_uiqm = validate(model, val_loader, device)
        val_uciqes.append(val_uciqe)
        val_uiqms.append(val_uiqm)

        scheduler.step(val_uciqe)
        current_lr = optimizer.param_groups[0]["lr"]

        log(
            f"Epoch [{epoch+1:>4}/{args.epochs}]  "
            f"Loss: {epoch_loss:.4f}  UCIQE: {val_uciqe:.4f}  "
            f"UIQM: {val_uiqm:.4f}  LR: {current_lr:.2e}  "
            f"({time.time()-t0:.1f}s)"
        )
        writer.writerow([epoch + 1, epoch_loss, val_uciqe, val_uiqm, current_lr])
        csv_file.flush()

        if (epoch + 1) % args.save_every == 0:
            ckpt = os.path.join(CHECKPOINTS_DIR, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt)
            log(f"  Saved checkpoint -> {ckpt}")

        if val_uciqe > best_uciqe:
            best_uciqe = val_uciqe
            torch.save(model.state_dict(), CHECKPOINT_BEST)
            log(f"  New best UCIQE {best_uciqe:.4f} -> {CHECKPOINT_BEST}")

    csv_file.close()
    torch.save(model.state_dict(), CHECKPOINT_FINAL)
    log(f"\nFinal model saved -> {CHECKPOINT_FINAL}")

    # Training curves
    epochs_x = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_x, train_losses, color="steelblue")
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].grid(True)
    axes[1].plot(epochs_x, val_uciqes, color="darkorange")
    axes[1].set_title("Val UCIQE"); axes[1].set_xlabel("Epoch"); axes[1].grid(True)
    axes[2].plot(epochs_x, val_uiqms, color="seagreen")
    axes[2].set_title("Val UIQM"); axes[2].set_xlabel("Epoch"); axes[2].grid(True)
    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    log(f"Training curves saved -> {curve_path}")
    log(f"\nTraining complete. Best UCIQE: {best_uciqe:.4f}")


if __name__ == "__main__":
    main()
