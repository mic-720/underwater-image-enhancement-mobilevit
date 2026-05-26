"""
training/train_with_val_loss_curve.py

Train the model and save only a train-vs-validation loss curve.

How validation loss is computed here:
- Since the current EUVP `validation` folder is input-only in this project,
  this script creates a validation split from the paired `trainA/trainB` data.

What it saves:
- one image containing both training loss and validation loss curves

What it does not save:
- no checkpoints
- no CSV logs
- no model weights
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from config import BATCH_SIZE, EPOCHS, EUVP_TRAIN_A, EUVP_TRAIN_B, IMAGE_SIZE, LR, RESULTS_DIR, W_COLOR, W_EDGE, W_L1
from datasets.euvp_dataset import EUVPDataset
from models.cnn_model import EnhancementCNN


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
    def __init__(self):
        super().__init__()
        self.l1_fn = nn.L1Loss()
        self.edge_fn = EdgeLoss()
        self.color_fn = ColorLoss()

    def forward(self, pred, target):
        l1 = self.l1_fn(pred, target)
        edge = self.edge_fn(pred, target)
        color = self.color_fn(pred, target)
        return W_L1 * l1 + W_EDGE * edge + W_COLOR * color


def parse_args():
    parser = argparse.ArgumentParser(description="Train model and save train/validation loss curve")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of paired train data used for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "train_val_loss_curve.png"),
        help="Path to save the train-vs-validation loss curve image",
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
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

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
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    val_losses = []

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

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch + 1}/{args.epochs}  "
            f"Train Loss: {epoch_train_loss:.6f}  "
            f"Val Loss: {epoch_val_loss:.6f}",
            flush=True,
        )

    epochs_x = range(1, args.epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_losses, label="Training Loss", color="steelblue", linewidth=2, marker="o")
    plt.plot(epochs_x, val_losses, label="Validation Loss", color="darkorange", linewidth=2, marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Curve saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
