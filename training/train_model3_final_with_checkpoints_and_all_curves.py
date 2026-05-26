"""
training/train_model3_final_with_checkpoints_and_all_curves.py

Train Final Model 3 with:
  - hybrid loss (L1 + Edge + Color)
  - periodic checkpoints
  - best checkpoint by validation PSNR
  - final checkpoint
  - four graphs:
      * training loss vs epoch
      * validation loss vs epoch
      * PSNR vs epoch
      * SSIM vs epoch

This script uses a paired split from EUVP trainA/trainB because PSNR and SSIM
require ground-truth target images.
"""

import argparse
import csv
import os
import random
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Subset

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    EUVP_TRAIN_A,
    EUVP_TRAIN_B,
    IMAGE_SIZE,
    LOGS_DIR,
    LR,
    RESULTS_DIR,
    SAVE_EVERY,
    W_COLOR,
    W_EDGE,
    W_L1,
)
from datasets.euvp_dataset import EUVPDataset
from models.model3_final import EnhancementCNN


RESULTS_SUBDIR = os.path.join(RESULTS_DIR, "model3_final_training_curves")
LOGS_SUBDIR = os.path.join(LOGS_DIR, "model3_final_training_curves")
CHECKPOINT_SUBDIR = os.path.join(CHECKPOINTS_DIR, "model3_final_training_curves")


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
            dtype=torch.float32,
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
    def __init__(self):
        super().__init__()
        self.l1_fn = nn.L1Loss()
        self.edge_fn = EdgeLoss()
        self.color_fn = ColorLoss()

    def forward(self, pred, target):
        l1 = self.l1_fn(pred, target)
        edge = self.edge_fn(pred, target)
        color = self.color_fn(pred, target)
        total = W_L1 * l1 + W_EDGE * edge + W_COLOR * color
        return total, {
            "total": total.item(),
            "l1": l1.item(),
            "edge": edge.item(),
            "color": color.item(),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Final Model 3 with checkpoints and all four epoch graphs"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inp, tar in loader:
            inp = inp.to(device)
            tar = tar.to(device)
            out = model(inp)
            loss, _ = criterion(out, tar)
            total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_psnr_ssim(model, loader, device):
    model.eval()
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for inp, tar in loader:
            inp = inp.to(device)
            out = model(inp).cpu().numpy()
            tar = tar.cpu().numpy()

            for pred, gt in zip(out, tar):
                pred_img = np.transpose(np.clip(pred, 0.0, 1.0), (1, 2, 0))
                gt_img = np.transpose(np.clip(gt, 0.0, 1.0), (1, 2, 0))
                psnr_scores.append(
                    peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
                )
                ssim_scores.append(
                    structural_similarity(
                        gt_img,
                        pred_img,
                        channel_axis=2,
                        data_range=1.0,
                    )
                )

    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def save_metric_csv(path, epochs, values, metric_name):
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch", metric_name])
        for epoch, value in zip(epochs, values):
            writer.writerow([epoch, value])


def save_single_curve(path, epochs, values, ylabel, title, color):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, color=color, linewidth=2, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_datasets(seed, val_ratio):
    train_augmented = EUVPDataset(
        EUVP_TRAIN_A,
        EUVP_TRAIN_B,
        augment=True,
        image_size=IMAGE_SIZE,
    )
    train_plain = EUVPDataset(
        EUVP_TRAIN_A,
        EUVP_TRAIN_B,
        augment=False,
        image_size=IMAGE_SIZE,
    )

    indices = list(range(len(train_plain)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_augmented, train_indices)
    val_subset = Subset(train_plain, val_indices)
    return train_subset, val_subset


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(RESULTS_SUBDIR, exist_ok=True)
    os.makedirs(LOGS_SUBDIR, exist_ok=True)
    os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_subset, val_subset = build_datasets(args.seed, args.val_ratio)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Train samples: {len(train_subset)}", flush=True)
    print(f"Validation samples: {len(val_subset)}", flush=True)

    model = EnhancementCNN().to(device)
    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}", flush=True)

    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )
    criterion = CombinedLoss().to(device)

    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []
    best_psnr = -1.0

    master_log_path = os.path.join(LOGS_SUBDIR, "model3_final_epoch_metrics.csv")
    with open(master_log_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "lr"])

        for epoch in range(args.epochs):
            t0 = time.time()
            model.train()
            running_train_loss = 0.0

            for inp, tar in train_loader:
                inp = inp.to(device)
                tar = tar.to(device)

                out = model(inp)
                loss, _ = criterion(out, tar)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)
            epoch_val_loss = evaluate_loss(model, val_loader, criterion, device)
            epoch_val_psnr, epoch_val_ssim = evaluate_psnr_ssim(model, val_loader, device)

            scheduler.step(epoch_val_psnr)
            current_lr = optimizer.param_groups[0]["lr"]

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            val_psnrs.append(epoch_val_psnr)
            val_ssims.append(epoch_val_ssim)

            writer.writerow(
                [epoch + 1, epoch_train_loss, epoch_val_loss, epoch_val_psnr, epoch_val_ssim, current_lr]
            )
            file_obj.flush()

            print(
                f"Epoch [{epoch + 1:>4}/{args.epochs}]  "
                f"Train Loss: {epoch_train_loss:.6f}  "
                f"Val Loss: {epoch_val_loss:.6f}  "
                f"PSNR: {epoch_val_psnr:.4f}  "
                f"SSIM: {epoch_val_ssim:.4f}  "
                f"LR: {current_lr:.2e}  "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(CHECKPOINT_SUBDIR, f"model3_final_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint -> {ckpt_path}", flush=True)

            if epoch_val_psnr > best_psnr:
                best_psnr = epoch_val_psnr
                best_ckpt_path = os.path.join(CHECKPOINT_SUBDIR, "model3_final_best.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"New best PSNR {best_psnr:.4f} -> {best_ckpt_path}", flush=True)

    final_ckpt_path = os.path.join(CHECKPOINT_SUBDIR, "model3_final.pth")
    torch.save(model.state_dict(), final_ckpt_path)

    epochs_x = list(range(1, args.epochs + 1))
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model3_final_train_loss_vs_epoch.png"),
        epochs_x,
        train_losses,
        "Training Loss",
        "Model 3 Final Training Loss vs Epoch",
        "steelblue",
    )
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model3_final_val_loss_vs_epoch.png"),
        epochs_x,
        val_losses,
        "Validation Loss",
        "Model 3 Final Validation Loss vs Epoch",
        "darkorange",
    )
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model3_final_psnr_vs_epoch.png"),
        epochs_x,
        val_psnrs,
        "PSNR",
        "Model 3 Final PSNR vs Epoch",
        "royalblue",
    )
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model3_final_ssim_vs_epoch.png"),
        epochs_x,
        val_ssims,
        "SSIM",
        "Model 3 Final SSIM vs Epoch",
        "seagreen",
    )

    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model3_final_train_loss_log.csv"),
        epochs_x,
        train_losses,
        "train_loss",
    )
    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model3_final_val_loss_log.csv"),
        epochs_x,
        val_losses,
        "val_loss",
    )
    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model3_final_psnr_log.csv"),
        epochs_x,
        val_psnrs,
        "val_psnr",
    )
    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model3_final_ssim_log.csv"),
        epochs_x,
        val_ssims,
        "val_ssim",
    )

    print(f"Saved plots to: {RESULTS_SUBDIR}", flush=True)
    print(f"Saved logs to: {LOGS_SUBDIR}", flush=True)
    print(f"Saved final checkpoint to: {final_ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
