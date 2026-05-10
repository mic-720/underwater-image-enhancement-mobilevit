"""
evaluation/ablation_study.py

Runs three 20-epoch training runs to measure each loss component's contribution:
  Run A — Full loss      : L1 + Edge + Color  (your final model)
  Run B — No edge loss   : L1 + Color only
  Run C — No color loss  : L1 + Edge only

After all three runs, evaluates each checkpoint on EUVP validation and
prints a ready-to-paste Table 2 for your paper.

Usage:
    python evaluation/ablation_study.py
    python evaluation/ablation_study.py --epochs 20
    python evaluation/ablation_study.py --eval-only  (if training already done)

Time estimate: ~45–60 minutes total on a GPU, ~3 hours on CPU.
"""

import argparse
import csv
import os
import sys
import time
import copy

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
    EUVP_TRAIN_A, EUVP_TRAIN_B, EUVP_VAL,
    CHECKPOINTS_DIR, RESULTS_DIR, IMAGE_SIZE,
    BATCH_SIZE, LR,
)
from datasets.euvp_dataset import EUVPDataset
from models.cnn_model import EnhancementCNN


# ─────────────────────────────────────────────────────────────────────────────
# Loss components (inline — no config import needed for weights)
# ─────────────────────────────────────────────────────────────────────────────

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, x):
        gray = x.mean(dim=1, keepdim=True)
        return torch.sqrt(F.conv2d(gray, self.kx, padding=1)**2 +
                          F.conv2d(gray, self.ky, padding=1)**2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self._edges(pred), self._edges(target))


class ColorLoss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred.mean(dim=[2,3]), target.mean(dim=[2,3]))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_uciqe(img):
    img = np.clip(img, 0.0, 1.0)
    lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    chroma = np.sqrt(A**2 + B**2)
    return 0.4680*np.std(chroma) + 0.2745*(np.max(L)-np.min(L))/255.0 + 0.2576*np.mean(chroma)/255.0


def compute_uiqm(img):
    img   = np.clip(img, 0.0, 1.0)
    u8    = (img*255).astype(np.uint8)
    rg    = u8[:,:,0].astype(np.float32) - u8[:,:,1].astype(np.float32)
    yb    = 0.5*(u8[:,:,0].astype(np.float32)+u8[:,:,1].astype(np.float32)) - u8[:,:,2].astype(np.float32)
    gray  = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    return (0.0282*(np.std(rg)+np.std(yb)) + 0.2953*np.mean(np.abs(cv2.Sobel(gray,cv2.CV_64F,1,1))) + 3.5753*np.std(gray)) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = [
    {
        "name":    "Full loss (L1 + Edge + Color)",
        "tag":     "full",
        "w_l1":    0.6,
        "w_edge":  0.3,
        "w_color": 0.1,
    },
    {
        "name":    "No edge loss (L1 + Color)",
        "tag":     "no_edge",
        "w_l1":    0.9,
        "w_edge":  0.0,
        "w_color": 0.1,
    },
    {
        "name":    "No color loss (L1 + Edge)",
        "tag":     "no_color",
        "w_l1":    0.7,
        "w_edge":  0.3,
        "w_color": 0.0,
    },
]




# ─────────────────────────────────────────────────────────────────────────────
# Train one ablation run
# ─────────────────────────────────────────────────────────────────────────────

def train_one(cfg, epochs, batch, device, train_loader, val_loader):
    model     = EnhancementCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    l1_fn     = nn.L1Loss()
    edge_fn   = EdgeLoss().to(device)
    color_fn  = ColorLoss().to(device)

    print(f"\n{'='*55}")
    print(f"Run: {cfg['name']}")
    print(f"  w_l1={cfg['w_l1']}  w_edge={cfg['w_edge']}  w_color={cfg['w_color']}")
    print(f"{'='*55}")

    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for inp, tar in train_loader:
            inp, tar = inp.to(device), tar.to(device)
            out   = model(inp)
            loss  = (cfg["w_l1"]   * l1_fn(out, tar)
                   + cfg["w_edge"]  * edge_fn(out, tar)
                   + cfg["w_color"] * color_fn(out, tar))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)

        if (epoch+1) % 5 == 0 or epoch == epochs-1:
            print(f"  Epoch [{epoch+1:>3}/{epochs}]  Loss: {ep_loss:.4f}")

    # Save checkpoint
    ckpt_dir  = os.path.join(CHECKPOINTS_DIR, "ablation")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"ablation_{cfg['tag']}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")
    return model, ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, val_loader, device):
    model.eval()
    uciqe_scores, uiqm_scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            inp = batch[0] if isinstance(batch, (list,tuple)) else batch
            out = model(inp.to(device))
            for i in range(out.size(0)):
                img = out[i].cpu().numpy().transpose(1,2,0)
                uciqe_scores.append(compute_uciqe(img))
                uiqm_scores.append(compute_uiqm(img))
    return float(np.mean(uciqe_scores)), float(np.mean(uiqm_scores))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ablation study — 3 loss configurations")
    p.add_argument("--epochs",    type=int,  default=20)
    p.add_argument("--batch",     type=int,  default=BATCH_SIZE)
    p.add_argument("--workers",   type=int,  default=4)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, only evaluate existing ablation checkpoints")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ablation Study")
    print(f"Device  : {device}")
    print(f"Epochs  : {args.epochs} per run")
    print(f"Configs : {len(CONFIGS)}")

    train_ds = EUVPDataset(EUVP_TRAIN_A, EUVP_TRAIN_B, augment=True,  image_size=IMAGE_SIZE)
    val_ds   = EUVPDataset(EUVP_VAL,     None,          augment=False, image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    results = []
    ckpt_dir = os.path.join(CHECKPOINTS_DIR, "ablation")

    for cfg in CONFIGS:
        ckpt_path = os.path.join(ckpt_dir, f"ablation_{cfg['tag']}.pth")

        if args.eval_only:
            if not os.path.isfile(ckpt_path):
                print(f"  [SKIP] {cfg['tag']} checkpoint not found: {ckpt_path}")
                continue
            print(f"\nLoading {cfg['tag']} checkpoint for evaluation ...")
            model = EnhancementCNN().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            model, ckpt_path = train_one(cfg, args.epochs, args.batch, device,
                                         train_loader, val_loader)

        print(f"  Evaluating {cfg['name']} ...")
        uciqe, uiqm = evaluate_model(model, val_loader, device)
        print(f"  UCIQE: {uciqe:.4f}  UIQM: {uiqm:.4f}")

        results.append({
            "config":    cfg["name"],
            "w_l1":      cfg["w_l1"],
            "w_edge":    cfg["w_edge"],
            "w_color":   cfg["w_color"],
            "val_uciqe": uciqe,
            "val_uiqm":  uiqm,
            "checkpoint": ckpt_path,
        })

    # ── Print paper-ready Table 2 ─────────────────────────────────────────
    print("\n" + "="*60)
    print("TABLE 2 — Ablation Study (EUVP validation set)")
    print("="*60)
    print(f"{'Configuration':<35} {'UCIQE':>8} {'UIQM':>8}")
    print("-"*55)
    for r in results:
        print(f"{r['config']:<35} {r['val_uciqe']:>8.4f} {r['val_uiqm']:>8.4f}")
    print("="*60)

    # ── Save results ─────────────────────────────────────────────────────
    out_dir  = os.path.join(RESULTS_DIR, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        if results:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    txt_path = os.path.join(out_dir, "ablation_table2.txt")
    with open(txt_path, "w") as f:
        f.write("TABLE 2 — Ablation Study (EUVP validation, 20 epochs)\n")
        f.write("="*60 + "\n")
        f.write(f"{'Configuration':<35} {'UCIQE':>8} {'UIQM':>8}\n")
        f.write("-"*55 + "\n")
        for r in results:
            f.write(f"{r['config']:<35} {r['val_uciqe']:>8.4f} {r['val_uiqm']:>8.4f}\n")
        f.write("="*60 + "\n")
        f.write("\nNote: All runs trained from scratch for 20 epochs on EUVP paired data.\n")
        f.write("Full model uses L1:0.6 + Edge:0.3 + Color:0.1 weights.\n")

    print(f"\nResults saved:")
    print(f"  CSV → {csv_path}")
    print(f"  TXT → {txt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
