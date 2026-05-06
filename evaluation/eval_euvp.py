"""
evaluation/eval_euvp.py

No-reference evaluation (UCIQE + UIQM) on the EUVP validation set.
Consolidates eval.py and euvb_metric.py — both were identical.

Usage:
    python main.py eval --dataset euvp
    python main.py eval --dataset euvp --checkpoint checkpoints/model_best.pth
    python evaluation/eval_euvp.py
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import EUVP_VAL, CHECKPOINT_FINAL, RESULTS_DIR, IMAGE_SIZE
from models.cnn_model import EnhancementCNN

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_uciqe(img: np.ndarray) -> float:
    img     = np.clip(img, 0.0, 1.0)
    img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(img_lab)
    chroma  = np.sqrt(A ** 2 + B ** 2)
    return (0.4680 * np.std(chroma)
            + 0.2745 * (np.max(L) - np.min(L)) / 255.0
            + 0.2576 * np.mean(chroma) / 255.0)


def compute_uiqm(img: np.ndarray) -> float:
    img    = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    rg     = img_u8[:, :, 0].astype(np.float32) - img_u8[:, :, 1].astype(np.float32)
    yb     = (0.5 * (img_u8[:, :, 0].astype(np.float32) +
               img_u8[:, :, 1].astype(np.float32))
              - img_u8[:, :, 2].astype(np.float32))
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return ((0.0282 * (np.std(rg) + np.std(yb))
             + 0.2953 * np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1)))
             + 3.5753 * np.std(gray)) / 255.0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on EUVP validation set")
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_FINAL)
    p.add_argument("--input",      type=str, default=EUVP_VAL)
    p.add_argument("--size",       type=int, default=IMAGE_SIZE)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EUVP-Val] Device: {device}  Checkpoint: {args.checkpoint}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.isdir(args.input):
        raise FileNotFoundError(f"Validation dir not found: {args.input}")

    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    tfm   = T.Compose([T.Resize((args.size, args.size), antialias=True), T.ToTensor()])
    files = sorted(f for f in os.listdir(args.input)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)

    before_u, before_q, after_u, after_q = [], [], [], []

    with torch.no_grad():
        for fname in files:
            inp_t = tfm(Image.open(os.path.join(args.input, fname)).convert("RGB"))
            inp_t = inp_t.unsqueeze(0).to(device)
            out_t = model(inp_t)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)

            before_u.append(compute_uciqe(inp_np));  after_u.append(compute_uciqe(out_np))
            before_q.append(compute_uiqm(inp_np));   after_q.append(compute_uiqm(out_np))

    def pct(b, a): return (a - b) / (abs(b) + 1e-9) * 100

    b_uciqe, a_uciqe = float(np.mean(before_u)), float(np.mean(after_u))
    b_uiqm,  a_uiqm  = float(np.mean(before_q)), float(np.mean(after_q))

    print(f"\n[EUVP-Val]  n={len(files)}")
    print(f"  UCIQE : {b_uciqe:.4f} -> {a_uciqe:.4f}  ({pct(b_uciqe, a_uciqe):+.2f}%)")
    print(f"  UIQM  : {b_uiqm:.4f}  -> {a_uiqm:.4f}   ({pct(b_uiqm,  a_uiqm):+.2f}%)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "euvp_val_metrics.txt")
    with open(out_path, "w") as f:
        f.write(f"EUVP Validation Metrics\n")
        f.write(f"Images : {len(files)}\n")
        f.write(f"UCIQE  : {b_uciqe:.4f} -> {a_uciqe:.4f} ({pct(b_uciqe, a_uciqe):+.2f}%)\n")
        f.write(f"UIQM   : {b_uiqm:.4f}  -> {a_uiqm:.4f}  ({pct(b_uiqm,  a_uiqm):+.2f}%)\n")
    print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
