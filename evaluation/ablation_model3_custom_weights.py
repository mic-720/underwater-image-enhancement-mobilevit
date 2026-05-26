"""
evaluation/ablation_model3_custom_weights.py

Run custom weighted-loss ablations for Final Model 3 using these variants:
  1) 0.70 * L1 + 0.25 * Edge + 0.05 * Color
  2) 0.75 * L1 + 0.20 * Edge + 0.05 * Color
  3) 0.65 * L1 + 0.25 * Edge + 0.10 * Color
  4) 0.65 * L1 + 0.30 * Edge + 0.05 * Color

This script trains each run from scratch, evaluates on EUVP validation using
UCIQE and UIQM, and saves outputs in separate model3-specific folders.

Usage:
    python evaluation/ablation_model3_custom_weights.py
    python evaluation/ablation_model3_custom_weights.py --epochs 20
    python evaluation/ablation_model3_custom_weights.py --eval-only
"""

import argparse
import csv
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EUVP_TRAIN_A,
    EUVP_TRAIN_B,
    EUVP_VAL,
    IMAGE_SIZE,
    LR,
    RESULTS_DIR,
)
from datasets.euvp_dataset import EUVPDataset
from models.model3_final import EnhancementCNN


ABLATION_TAG = "ablation_model3_custom_weights"
CKPT_DIR = os.path.join(CHECKPOINTS_DIR, ABLATION_TAG)
OUT_DIR = os.path.join(RESULTS_DIR, ABLATION_TAG)


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
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


class ColorLoss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred.mean(dim=[2, 3]), target.mean(dim=[2, 3]))


def compute_uciqe(img):
    img = np.clip(img, 0.0, 1.0)
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)
    chroma = np.sqrt(a_chan ** 2 + b_chan ** 2)
    return (
        0.4680 * np.std(chroma)
        + 0.2745 * (np.max(l_chan) - np.min(l_chan)) / 255.0
        + 0.2576 * np.mean(chroma) / 255.0
    )


def compute_uiqm(img):
    img = np.clip(img, 0.0, 1.0)
    u8 = (img * 255).astype(np.uint8)
    rg = u8[:, :, 0].astype(np.float32) - u8[:, :, 1].astype(np.float32)
    yb = (
        0.5 * (u8[:, :, 0].astype(np.float32) + u8[:, :, 1].astype(np.float32))
        - u8[:, :, 2].astype(np.float32)
    )
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    sharpness = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1)))
    return (0.0282 * (np.std(rg) + np.std(yb)) + 0.2953 * sharpness + 3.5753 * np.std(gray)) / 255.0


CUSTOM_CONFIGS = [
    {
        "name": "0.70 L1 + 0.25 Edge + 0.05 Color",
        "tag": "l1_070_edge_025_color_005",
        "losses": "0.70*L1 + 0.25*Edge + 0.05*Color",
        "w_l1": 0.70,
        "w_edge": 0.25,
        "w_color": 0.05,
        "note": "Best-result candidate",
    },
    {
        "name": "0.75 L1 + 0.20 Edge + 0.05 Color",
        "tag": "l1_075_edge_020_color_005",
        "losses": "0.75*L1 + 0.20*Edge + 0.05*Color",
        "w_l1": 0.75,
        "w_edge": 0.20,
        "w_color": 0.05,
        "note": "Upper-limit check",
    },
    {
        "name": "0.65 L1 + 0.25 Edge + 0.10 Color",
        "tag": "l1_065_edge_025_color_010",
        "losses": "0.65*L1 + 0.25*Edge + 0.10*Color",
        "w_l1": 0.65,
        "w_edge": 0.25,
        "w_color": 0.10,
        "note": "Higher color contribution",
    },
    {
        "name": "0.65 L1 + 0.30 Edge + 0.05 Color",
        "tag": "l1_065_edge_030_color_005",
        "losses": "0.65*L1 + 0.30*Edge + 0.05*Color",
        "w_l1": 0.65,
        "w_edge": 0.30,
        "w_color": 0.05,
        "note": "Edge-dominant variant",
    },
]


def train_one(cfg, epochs, device, train_loader):
    model = EnhancementCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    l1_fn = nn.L1Loss()
    edge_fn = EdgeLoss().to(device)
    color_fn = ColorLoss().to(device)

    print(f"\n{'=' * 65}")
    print(f"Run: {cfg['name']}")
    print(
        "  weights:"
        f" L1={cfg['w_l1']:.2f}"
        f" Edge={cfg['w_edge']:.2f}"
        f" Color={cfg['w_color']:.2f}"
    )
    print(f"  note   : {cfg['note']}")
    print(f"{'=' * 65}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inp, target in train_loader:
            inp = inp.to(device)
            target = target.to(device)
            out = model(inp)
            loss = (
                cfg["w_l1"] * l1_fn(out, target)
                + cfg["w_edge"] * edge_fn(out, target)
                + cfg["w_color"] * color_fn(out, target)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch + 1:>3}/{epochs}] Loss: {epoch_loss:.4f}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{cfg['tag']}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved -> {ckpt_path}")
    return model, ckpt_path


def evaluate_model(model, val_loader, device):
    model.eval()
    uciqe_scores = []
    uiqm_scores = []

    with torch.no_grad():
        for batch in val_loader:
            inp = batch[0] if isinstance(batch, (list, tuple)) else batch
            out = model(inp.to(device))
            for i in range(out.size(0)):
                img = out[i].cpu().numpy().transpose(1, 2, 0)
                uciqe_scores.append(compute_uciqe(img))
                uiqm_scores.append(compute_uiqm(img))

    return float(np.mean(uciqe_scores)), float(np.mean(uiqm_scores))


def parse_args():
    parser = argparse.ArgumentParser(description="Custom loss-weight ablation for Final Model 3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and only evaluate existing checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Custom Ablation - Final Model 3")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.epochs} per run")
    print(f"Configs : {len(CUSTOM_CONFIGS)}")

    train_ds = EUVPDataset(EUVP_TRAIN_A, EUVP_TRAIN_B, augment=True, image_size=IMAGE_SIZE)
    val_ds = EUVPDataset(EUVP_VAL, None, augment=False, image_size=IMAGE_SIZE)

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

    results = []

    for cfg in CUSTOM_CONFIGS:
        ckpt_path = os.path.join(CKPT_DIR, f"{cfg['tag']}.pth")

        if args.eval_only:
            if not os.path.isfile(ckpt_path):
                print(f"  [SKIP] Missing checkpoint for {cfg['name']}: {ckpt_path}")
                continue
            print(f"\nLoading checkpoint for evaluation: {cfg['name']}")
            model = EnhancementCNN().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            model, ckpt_path = train_one(cfg, args.epochs, device, train_loader)

        print(f"  Evaluating {cfg['name']} ...")
        uciqe, uiqm = evaluate_model(model, val_loader, device)
        print(f"  UCIQE: {uciqe:.4f}  UIQM: {uiqm:.4f}")

        results.append(
            {
                "config": cfg["name"],
                "losses": cfg["losses"],
                "note": cfg["note"],
                "w_l1": cfg["w_l1"],
                "w_edge": cfg["w_edge"],
                "w_color": cfg["w_color"],
                "val_uciqe": uciqe,
                "val_uiqm": uiqm,
                "checkpoint": ckpt_path,
            }
        )

    print("\n" + "=" * 72)
    print("CUSTOM ABLATION TABLE - Final Model 3 (EUVP validation set)")
    print("=" * 72)
    print(f"{'Configuration':<38} {'UCIQE':>8} {'UIQM':>8}")
    print("-" * 72)
    for row in results:
        print(f"{row['config']:<38} {row['val_uciqe']:>8.4f} {row['val_uiqm']:>8.4f}")
    print("=" * 72)

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "custom_weight_ablation_results.csv")
    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    txt_path = os.path.join(OUT_DIR, "custom_weight_ablation_table.txt")
    with open(txt_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"CUSTOM ABLATION TABLE - Final Model 3 ({args.epochs} epochs)\n")
        file_obj.write("=" * 72 + "\n")
        file_obj.write(f"{'Configuration':<38} {'UCIQE':>8} {'UIQM':>8}\n")
        file_obj.write("-" * 72 + "\n")
        for row in results:
            file_obj.write(f"{row['config']:<38} {row['val_uciqe']:>8.4f} {row['val_uiqm']:>8.4f}\n")
        file_obj.write("=" * 72 + "\n\n")
        file_obj.write("Model: Final Model 3\n")
        file_obj.write("All runs are trained from scratch on EUVP paired data.\n")
        file_obj.write("Custom weights are user-defined and sum to 1.0.\n")

    print("\nResults saved:")
    print(f"  CSV -> {csv_path}")
    print(f"  TXT -> {txt_path}")
    print(f"  CKPT DIR -> {CKPT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
