"""
Train the simple CNN model and save a training-vs-validation loss curve.

Validation loss here is computed from a paired split created from EUVP
trainA/trainB, because the project's EUVP validation folder is input-only.
"""

import argparse
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

from config import BATCH_SIZE, EPOCHS, EUVP_TRAIN_A, EUVP_TRAIN_B, IMAGE_SIZE, LR, RESULTS_DIR
from datasets.euvp_dataset import EUVPDataset
from models.simple_cnn_model import EnhancementCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train simple CNN and save training/validation loss curve"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "simple_cnn_train_val_curve.png"),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
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
            loss = criterion(out, tar)
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


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

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

        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"Train Loss: {epoch_train_loss:.6f}  "
            f"Val Loss: {epoch_val_loss:.6f}  "
            f"PSNR: {epoch_val_psnr:.4f}  "
            f"SSIM: {epoch_val_ssim:.4f}",
            flush=True,
        )

    epochs_x = range(1, args.epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_losses, label="Training Loss", color="steelblue", linewidth=2)
    plt.plot(epochs_x, val_losses, label="Validation Loss", color="darkorange", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    output_root, output_ext = os.path.splitext(args.output)
    psnr_output = f"{output_root}_psnr{output_ext}"
    ssim_output = f"{output_root}_ssim{output_ext}"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, val_psnrs, color="royalblue", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(psnr_output, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, val_ssims, color="seagreen", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM vs Epoch")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(ssim_output, dpi=300)
    plt.close()

    print(f"Curve saved to: {args.output}", flush=True)
    print(f"PSNR curve saved to: {psnr_output}", flush=True)
    print(f"SSIM curve saved to: {ssim_output}", flush=True)


if __name__ == "__main__":
    main()
