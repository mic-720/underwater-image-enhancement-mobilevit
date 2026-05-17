"""
training/train_curve_only.py

Minimal training script for generating only the training loss curve.

What it does:
- trains on EUVP paired training data
- computes training loss for each epoch
- saves only one figure: training loss vs epoch

What it does not do:
- no validation
- no checkpoints
- no CSV logs
- no model saving
"""

import argparse
import os
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
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    EPOCHS,
    EUVP_TRAIN_A,
    EUVP_TRAIN_B,
    IMAGE_SIZE,
    LR,
    RESULTS_DIR,
    W_COLOR,
    W_EDGE,
    W_L1,
)
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
    parser = argparse.ArgumentParser(description="Train model and save only training loss curve")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "training_curve_only.png"),
        help="Path to save the training curve image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_ds = EUVPDataset(
        EUVP_TRAIN_A,
        EUVP_TRAIN_B,
        augment=True,
        image_size=IMAGE_SIZE,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = EnhancementCNN().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for inp, tar in train_loader:
            inp = inp.to(device)
            tar = tar.to(device)

            out = model(inp)
            loss = criterion(out, tar)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{args.epochs}  Loss: {epoch_loss:.6f}", flush=True)

    epochs_x = range(1, args.epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_losses, color="steelblue", linewidth=2, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    plt.close()

    print(f"Training curve saved to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
