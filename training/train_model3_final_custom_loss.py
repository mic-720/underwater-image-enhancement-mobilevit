"""
training/train_model3_final_custom_loss.py

Train the final Model 3 architecture with a custom combined loss:
    0.70 * L1 + 0.25 * Edge + 0.05 * Color

Usage:
    python training/train_model3_final_custom_loss.py
    python training/train_model3_final_custom_loss.py --epochs 100 --batch 8
    python training/train_model3_final_custom_loss.py --resume checkpoints/.../model3_final_best.pth
"""

import argparse
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    EUVP_TRAIN_A,
    EUVP_TRAIN_B,
    EUVP_VAL,
    IMAGE_SIZE,
    LR,
    RESULTS_DIR,
    SAVE_EVERY,
    LOGS_DIR,
)
from datasets.euvp_dataset import EUVPDataset
from models.model3_final import EnhancementCNN


DEFAULT_W_L1 = 0.70
DEFAULT_W_EDGE = 0.25
DEFAULT_W_COLOR = 0.05


def log(message=""):
    print(message, flush=True)


def make_weight_tag(w_l1: float, w_edge: float, w_color: float) -> str:
    return (
        f"l1_{int(round(w_l1 * 1000)):03d}_"
        f"edge_{int(round(w_edge * 1000)):03d}_"
        f"color_{int(round(w_color * 1000)):03d}"
    )


class EdgeLoss(nn.Module):
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
    def forward(self, pred, target):
        return F.l1_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))


class CombinedLoss(nn.Module):
    def __init__(self, w_l1: float, w_edge: float, w_color: float):
        super().__init__()
        self.w_l1 = w_l1
        self.w_edge = w_edge
        self.w_color = w_color
        self.l1_fn = nn.L1Loss()
        self.edge_fn = EdgeLoss()
        self.color_fn = ColorLoss()

    def forward(self, pred, target):
        l1 = self.l1_fn(pred, target)
        edge = self.edge_fn(pred, target)
        color = self.color_fn(pred, target)
        total = self.w_l1 * l1 + self.w_edge * edge + self.w_color * color
        return total, {
            "total": total.item(),
            "l1": l1.item(),
            "edge": edge.item(),
            "color": color.item(),
        }


def compute_uciqe(img: np.ndarray) -> float:
    import cv2

    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    img_lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)
    chroma = np.sqrt(a_chan ** 2 + b_chan ** 2)
    return (
        0.4680 * np.std(chroma)
        + 0.2745 * (np.max(l_chan) - np.min(l_chan)) / 255.0
        + 0.2576 * np.mean(chroma) / 255.0
    )


def compute_uiqm(img: np.ndarray) -> float:
    import cv2

    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    rg = img_u8[:, :, 0].astype(np.float32) - img_u8[:, :, 1].astype(np.float32)
    yb = (
        0.5 * (img_u8[:, :, 0].astype(np.float32) + img_u8[:, :, 1].astype(np.float32))
        - img_u8[:, :, 2].astype(np.float32)
    )
    uicm = float(np.std(rg) + np.std(yb))
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    uism = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1))))
    uiconm = float(np.std(gray))
    return (0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm) / 255.0


def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss_total = 0.0
    val_batches = 0
    uciqe_scores = []
    uiqm_scores = []

    with torch.no_grad():
        for batch in val_loader:
            inp = batch[0] if isinstance(batch, (list, tuple)) else batch
            target = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            inp = inp.to(device)
            out = model(inp)
            if target is not None:
                target = target.to(device)
                val_loss, _ = criterion(out, target)
                val_loss_total += val_loss.item()
                val_batches += 1
            for i in range(out.size(0)):
                img_np = out[i].cpu().numpy().transpose(1, 2, 0)
                uciqe_scores.append(compute_uciqe(img_np))
                uiqm_scores.append(compute_uiqm(img_np))

    mean_val_loss = val_loss_total / val_batches if val_batches else float("nan")
    return mean_val_loss, float(np.mean(uciqe_scores)), float(np.mean(uiqm_scores))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Model 3 final with custom weighted loss"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--w-l1", type=float, default=DEFAULT_W_L1)
    parser.add_argument("--w-edge", type=float, default=DEFAULT_W_EDGE)
    parser.add_argument("--w-color", type=float, default=DEFAULT_W_COLOR)
    return parser.parse_args()


def main():
    args = parse_args()
    weight_sum = args.w_l1 + args.w_edge + args.w_color
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Loss weights must sum to 1.0, got {weight_sum:.6f}."
        )

    experiment_tag = f"model3_final_{make_weight_tag(args.w_l1, args.w_edge, args.w_color)}"
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, experiment_tag)
    result_dir = os.path.join(RESULTS_DIR, experiment_tag)
    log_dir = os.path.join(LOGS_DIR, experiment_tag)

    for directory in (ckpt_dir, result_dir, log_dir):
        os.makedirs(directory, exist_ok=True)

    best_ckpt_path = os.path.join(ckpt_dir, "model3_final_best.pth")
    final_ckpt_path = os.path.join(ckpt_dir, "model3_final_final.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")
    if device.type == "cuda":
        log(f"GPU    : {torch.cuda.get_device_name(0)}")

    log(
        "Loss   : "
        f"{args.w_l1:.2f} * L1 + {args.w_edge:.2f} * Edge + {args.w_color:.2f} * Color"
    )
    log(f"Output : {experiment_tag}")

    train_ds = EUVPDataset(
        EUVP_TRAIN_A,
        EUVP_TRAIN_B,
        augment=True,
        image_size=IMAGE_SIZE,
    )
    val_ds = EUVPDataset(
        EUVP_VAL,
        None,
        augment=False,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    log(f"Train  : {len(train_ds)} images  |  Val: {len(val_ds)} images")

    model = EnhancementCNN().to(device)
    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        log(f"Resumed from {args.resume}")

    log(f"Params : {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )
    criterion = CombinedLoss(args.w_l1, args.w_edge, args.w_color).to(device)

    metrics_csv_path = os.path.join(log_dir, "model3_final_epoch_metrics.csv")
    train_loss_csv_path = os.path.join(log_dir, "model3_final_train_loss_log.csv")
    val_loss_csv_path = os.path.join(log_dir, "model3_final_val_loss_log.csv")

    metrics_file = open(metrics_csv_path, "w", newline="")
    train_loss_file = open(train_loss_csv_path, "w", newline="")
    val_loss_file = open(val_loss_csv_path, "w", newline="")

    metrics_writer = csv.writer(metrics_file)
    train_loss_writer = csv.writer(train_loss_file)
    val_loss_writer = csv.writer(val_loss_file)

    metrics_writer.writerow(["epoch", "train_loss", "val_uciqe", "val_uiqm", "lr"])
    train_loss_writer.writerow(["epoch", "train_loss"])
    val_loss_writer.writerow(["epoch", "val_loss"])

    train_losses = []
    val_uciqes = []
    val_uiqms = []
    best_uciqe = -1.0

    log("\n" + "-" * 60)
    log("Starting training")
    log("-" * 60)

    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            epoch_loss = 0.0
            num_batches = len(train_loader)
            progress_every = max(1, num_batches // 10)

            for batch_idx, (inp, tar) in enumerate(train_loader, start=1):
                inp = inp.to(device)
                tar = tar.to(device)

                out = model(inp)
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
                        f"  Epoch {epoch + 1:>4}/{args.epochs}  "
                        f"Batch {batch_idx:>4}/{num_batches}  "
                        f"Avg Loss: {avg_loss:.4f}  "
                        f"({elapsed:.1f}s)"
                    )

            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)

            val_loss, val_uciqe, val_uiqm = validate(model, val_loader, device, criterion)
            val_uciqes.append(val_uciqe)
            val_uiqms.append(val_uiqm)

            scheduler.step(val_uciqe)
            current_lr = optimizer.param_groups[0]["lr"]

            log(
                f"Epoch [{epoch + 1:>4}/{args.epochs}]  "
                f"Loss: {epoch_loss:.4f}  UCIQE: {val_uciqe:.4f}  "
                f"UIQM: {val_uiqm:.4f}  LR: {current_lr:.2e}  "
                f"({time.time() - t0:.1f}s)"
            )

            metrics_writer.writerow([epoch + 1, epoch_loss, val_uciqe, val_uiqm, current_lr])
            train_loss_writer.writerow([epoch + 1, epoch_loss])
            val_loss_writer.writerow([epoch + 1, val_loss])
            metrics_file.flush()
            train_loss_file.flush()
            val_loss_file.flush()

            if (epoch + 1) % args.save_every == 0:
                periodic_ckpt = os.path.join(ckpt_dir, f"model3_final_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), periodic_ckpt)
                log(f"  Saved checkpoint -> {periodic_ckpt}")

            if val_uciqe > best_uciqe:
                best_uciqe = val_uciqe
                torch.save(model.state_dict(), best_ckpt_path)
                log(f"  New best UCIQE {best_uciqe:.4f} -> {best_ckpt_path}")
    finally:
        metrics_file.close()
        train_loss_file.close()
        val_loss_file.close()

    torch.save(model.state_dict(), final_ckpt_path)
    log(f"\nFinal model saved -> {final_ckpt_path}")

    epochs_x = range(1, args.epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs_x, train_losses, color="steelblue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True)

    axes[1].plot(epochs_x, val_uciqes, color="darkorange")
    axes[1].set_title("Val UCIQE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True)

    axes[2].plot(epochs_x, val_uiqms, color="seagreen")
    axes[2].set_title("Val UIQM")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True)

    plt.tight_layout()
    curve_path = os.path.join(result_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close(fig)

    summary_path = os.path.join(result_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("Model: model3_final\n")
        summary_file.write(
            f"Loss: {args.w_l1:.2f} * L1 + {args.w_edge:.2f} * Edge + {args.w_color:.2f} * Color\n"
        )
        summary_file.write(f"Best UCIQE: {best_uciqe:.6f}\n")
        summary_file.write(f"Best checkpoint: {best_ckpt_path}\n")
        summary_file.write(f"Final checkpoint: {final_ckpt_path}\n")
        summary_file.write(f"Metrics CSV: {metrics_csv_path}\n")

    log(f"Training curves saved -> {curve_path}")
    log(f"Run summary saved -> {summary_path}")
    log(f"\nTraining complete. Best UCIQE: {best_uciqe:.4f}")


if __name__ == "__main__":
    main()
