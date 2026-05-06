"""
evaluation/eval_uieb.py

Full UIEB evaluation — consolidates uieb_metrics.py, ch-60.py, and uieb_test.py.

Evaluates:
  - UIEB raw-890 vs reference-890  → PSNR, SSIM, UCIQE, UIQM (before/after)
  - UIEB challenging-60            → UCIQE, UIQM (before/after, no ground truth)

Usage:
    python main.py eval --dataset uieb
    python main.py eval --dataset uieb --checkpoint checkpoints/model_best.pth --save-images
    python evaluation/eval_uieb.py
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import (
    UIEB_RAW, UIEB_REF, UIEB_CHALLENGE,
    CHECKPOINT_FINAL, RESULTS_DIR, IMAGE_SIZE,
)
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


def pct_change(before, after):
    return (after - before) / (abs(before) + 1e-9) * 100


# ─────────────────────────────────────────────────────────────────────────────
# Evaluations
# ─────────────────────────────────────────────────────────────────────────────

def eval_paired_890(model, device, tfm, raw_dir, ref_dir, save_images, save_dir):
    """UIEB raw-890 vs reference-890 — full-reference + no-reference."""
    raw_files = sorted(f for f in os.listdir(raw_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    ref_set   = set(os.listdir(ref_dir))
    pairs     = [f for f in raw_files if f in ref_set]

    if not pairs:
        print("  [SKIP] UIEB-890: no matching filename pairs found.")
        return None

    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    records = []
    to_pil  = T.ToPILImage()

    with torch.no_grad():
        for fname in pairs:
            raw_t = tfm(Image.open(os.path.join(raw_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
            ref_t = tfm(Image.open(os.path.join(ref_dir, fname)).convert("RGB"))

            out_t   = model(raw_t).clamp(0, 1)
            raw_np  = raw_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            ref_np  = ref_t.numpy().transpose(1, 2, 0)
            out_np  = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            if save_images:
                to_pil(out_t.squeeze(0).cpu()).save(os.path.join(save_dir, fname))

            records.append({
                "image":        fname,
                "before_psnr":  peak_signal_noise_ratio(ref_np, raw_np, data_range=1.0),
                "after_psnr":   peak_signal_noise_ratio(ref_np, out_np, data_range=1.0),
                "before_ssim":  structural_similarity(ref_np, raw_np, channel_axis=2, data_range=1.0),
                "after_ssim":   structural_similarity(ref_np, out_np, channel_axis=2, data_range=1.0),
                "before_uciqe": compute_uciqe(raw_np),
                "after_uciqe":  compute_uciqe(out_np),
                "before_uiqm":  compute_uiqm(raw_np),
                "after_uiqm":   compute_uiqm(out_np),
            })

    return pd.DataFrame(records)


def eval_challenging_60(model, device, tfm, ch_dir, save_images, save_dir):
    """UIEB challenging-60 — no-reference only (no ground truth available)."""
    files  = sorted(f for f in os.listdir(ch_dir)
                    if os.path.splitext(f)[1].lower() in IMG_EXTS)

    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    records = []
    to_pil  = T.ToPILImage()

    with torch.no_grad():
        for fname in files:
            inp_t = tfm(Image.open(os.path.join(ch_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
            out_t = model(inp_t).clamp(0, 1)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            if save_images:
                to_pil(out_t.squeeze(0).cpu()).save(os.path.join(save_dir, fname))

            records.append({
                "image":        fname,
                "before_uciqe": compute_uciqe(inp_np),
                "after_uciqe":  compute_uciqe(out_np),
                "before_uiqm":  compute_uiqm(inp_np),
                "after_uiqm":   compute_uiqm(out_np),
            })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on UIEB dataset")
    p.add_argument("--checkpoint",  type=str,  default=CHECKPOINT_FINAL)
    p.add_argument("--size",        type=int,  default=IMAGE_SIZE)
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UIEB] Device: {device}  Checkpoint: {args.checkpoint}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    tfm      = T.Compose([T.Resize((args.size, args.size), antialias=True), T.ToTensor()])
    out_root = os.path.join(RESULTS_DIR, "UIEB")
    os.makedirs(out_root, exist_ok=True)

    # ── UIEB-890 ─────────────────────────────────────────────────────────────
    print("\nEvaluating UIEB raw-890 vs reference-890 ...")
    df_890 = eval_paired_890(
        model, device, tfm,
        UIEB_RAW, UIEB_REF,
        save_images=args.save_images,
        save_dir=os.path.join(out_root, "enhanced_raw_890"),
    )

    if df_890 is not None:
        m = df_890.mean(numeric_only=True)
        print(f"\n  [UIEB-890]  n={len(df_890)}")
        print(f"  PSNR  : {m['before_psnr']:.4f} -> {m['after_psnr']:.4f}  ({pct_change(m['before_psnr'],  m['after_psnr']):+.2f}%)")
        print(f"  SSIM  : {m['before_ssim']:.4f} -> {m['after_ssim']:.4f}  ({pct_change(m['before_ssim'],  m['after_ssim']):+.2f}%)")
        print(f"  UCIQE : {m['before_uciqe']:.4f} -> {m['after_uciqe']:.4f}  ({pct_change(m['before_uciqe'], m['after_uciqe']):+.2f}%)")
        print(f"  UIQM  : {m['before_uiqm']:.4f} -> {m['after_uiqm']:.4f}   ({pct_change(m['before_uiqm'],  m['after_uiqm']):+.2f}%)")
        df_890.to_csv(os.path.join(out_root, "uieb_890_per_image.csv"), index=False)

    # ── Challenging-60 ───────────────────────────────────────────────────────
    print("\nEvaluating UIEB challenging-60 ...")
    df_ch = eval_challenging_60(
        model, device, tfm,
        UIEB_CHALLENGE,
        save_images=args.save_images,
        save_dir=os.path.join(out_root, "enhanced_challenging_60"),
    )

    if not df_ch.empty:
        m = df_ch.mean(numeric_only=True)
        print(f"\n  [UIEB-Ch60]  n={len(df_ch)}")
        print(f"  UCIQE : {m['before_uciqe']:.4f} -> {m['after_uciqe']:.4f}  ({pct_change(m['before_uciqe'], m['after_uciqe']):+.2f}%)")
        print(f"  UIQM  : {m['before_uiqm']:.4f} -> {m['after_uiqm']:.4f}   ({pct_change(m['before_uiqm'],  m['after_uiqm']):+.2f}%)")
        df_ch.to_csv(os.path.join(out_root, "uieb_ch60_per_image.csv"), index=False)

    # ── Text summary ─────────────────────────────────────────────────────────
    summary_path = os.path.join(out_root, "uieb_summary.txt")
    with open(summary_path, "w") as f:
        if df_890 is not None:
            m = df_890.mean(numeric_only=True)
            f.write("UIEB-890 (Paired)\n=================\n")
            f.write(f"Images : {len(df_890)}\n")
            f.write(f"PSNR   : {m['before_psnr']:.4f} -> {m['after_psnr']:.4f} ({pct_change(m['before_psnr'],  m['after_psnr']):+.2f}%)\n")
            f.write(f"SSIM   : {m['before_ssim']:.4f} -> {m['after_ssim']:.4f} ({pct_change(m['before_ssim'],  m['after_ssim']):+.2f}%)\n")
            f.write(f"UCIQE  : {m['before_uciqe']:.4f} -> {m['after_uciqe']:.4f} ({pct_change(m['before_uciqe'], m['after_uciqe']):+.2f}%)\n")
            f.write(f"UIQM   : {m['before_uiqm']:.4f} -> {m['after_uiqm']:.4f}  ({pct_change(m['before_uiqm'],  m['after_uiqm']):+.2f}%)\n\n")
        if not df_ch.empty:
            m = df_ch.mean(numeric_only=True)
            f.write("UIEB-Challenging-60 (No Reference)\n===================================\n")
            f.write(f"Images : {len(df_ch)}\n")
            f.write(f"UCIQE  : {m['before_uciqe']:.4f} -> {m['after_uciqe']:.4f} ({pct_change(m['before_uciqe'], m['after_uciqe']):+.2f}%)\n")
            f.write(f"UIQM   : {m['before_uiqm']:.4f} -> {m['after_uiqm']:.4f}  ({pct_change(m['before_uiqm'],  m['after_uiqm']):+.2f}%)\n")

    print(f"\n  Summary saved -> {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
