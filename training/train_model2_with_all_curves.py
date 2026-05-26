"""
training/train_model2_with_all_curves.py

Train Model 2 on a paired train/validation split from EUVP trainA/trainB and save:
  - training loss vs epoch
  - validation loss vs epoch
  - PSNR vs epoch
  - SSIM vs epoch

Outputs are written to dedicated folders:
  - results/model2_training_curves/
  - logs/model2_training_curves/

Usage:
    python training/train_model2_with_all_curves.py
"""

import argparse
import csv
import os
import random
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, random_split

from config import BATCH_SIZE, CHECKPOINTS_DIR, EPOCHS, EUVP_TRAIN_A, EUVP_TRAIN_B, IMAGE_SIZE, LR, LOGS_DIR, RESULTS_DIR
from datasets.euvp_dataset import EUVPDataset
from models.model2 import EnhancementCNN


RESULTS_SUBDIR = os.path.join(RESULTS_DIR, "model2_training_curves")
LOGS_SUBDIR = os.path.join(LOGS_DIR, "model2_training_curves")
CHECKPOINT_SUBDIR = os.path.join(CHECKPOINTS_DIR, "model2_training_curves")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Model 2 and save loss, PSNR, and SSIM epoch curves"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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
            total_loss += criterion(out, tar).item()
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


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(RESULTS_SUBDIR, exist_ok=True)
    os.makedirs(LOGS_SUBDIR, exist_ok=True)
    os.makedirs(CHECKPOINT_SUBDIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    full_dataset = EUVPDataset(
        EUVP_TRAIN_A,
        EUVP_TRAIN_B,
        augment=False,
        image_size=IMAGE_SIZE,
    )

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

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
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []
    val_psnrs = []
    val_ssims = []

    master_log_path = os.path.join(LOGS_SUBDIR, "model2_epoch_metrics.csv")
    with open(master_log_path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim"])

        for epoch in range(args.epochs):
            model.train()
            running_train_loss = 0.0

            for inp, tar in train_loader:
                inp = inp.to(device)
                tar = tar.to(device)

                out = model(inp)
                loss = criterion(out, tar)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)
            epoch_val_loss = evaluate_loss(model, val_loader, criterion, device)
            epoch_val_psnr, epoch_val_ssim = evaluate_psnr_ssim(model, val_loader, device)

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            val_psnrs.append(epoch_val_psnr)
            val_ssims.append(epoch_val_ssim)

            writer.writerow(
                [epoch + 1, epoch_train_loss, epoch_val_loss, epoch_val_psnr, epoch_val_ssim]
            )
            file_obj.flush()

            print(
                f"Epoch {epoch + 1}/{args.epochs}  "
                f"Train Loss: {epoch_train_loss:.6f}  "
                f"Val Loss: {epoch_val_loss:.6f}  "
                f"PSNR: {epoch_val_psnr:.4f}  "
                f"SSIM: {epoch_val_ssim:.4f}",
                flush=True,
            )

    epochs_x = list(range(1, args.epochs + 1))

    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model2_train_loss_vs_epoch.png"),
        epochs_x,
        train_losses,
        "Training Loss",
        "Model 2 Training Loss vs Epoch",
        "steelblue",
    )
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model2_val_loss_vs_epoch.png"),
        epochs_x,
        val_losses,
        "Validation Loss",
        "Model 2 Validation Loss vs Epoch",
        "darkorange",
    )
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model2_psnr_vs_epoch.png"),
        epochs_x,
        val_psnrs,
        "PSNR",
        "Model 2 PSNR vs Epoch",
        "royalblue",
    )
    save_single_curve(
        os.path.join(RESULTS_SUBDIR, "model2_ssim_vs_epoch.png"),
        epochs_x,
        val_ssims,
        "SSIM",
        "Model 2 SSIM vs Epoch",
        "seagreen",
    )

    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model2_train_loss_log.csv"),
        epochs_x,
        train_losses,
        "train_loss",
    )
    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model2_val_loss_log.csv"),
        epochs_x,
        val_losses,
        "val_loss",
    )
    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model2_psnr_log.csv"),
        epochs_x,
        val_psnrs,
        "val_psnr",
    )
    save_metric_csv(
        os.path.join(LOGS_SUBDIR, "model2_ssim_log.csv"),
        epochs_x,
        val_ssims,
        "val_ssim",
    )

    final_checkpoint_path = os.path.join(CHECKPOINT_SUBDIR, "model2_final.pth")
    torch.save(model.state_dict(), final_checkpoint_path)

    print(f"Saved plots to: {RESULTS_SUBDIR}", flush=True)
    print(f"Saved logs to: {LOGS_SUBDIR}", flush=True)
    print(f"Saved final checkpoint to: {final_checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
