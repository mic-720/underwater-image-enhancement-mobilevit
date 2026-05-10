"""
evaluation/clahe_baseline.py

CLAHE (Contrast Limited Adaptive Histogram Equalization) baseline evaluation.
Runs on UIEB-890 (paired) and RUIE — computes the same metrics as your model
so you get a direct apples-to-apples comparison row in your results table.

Usage:
    python evaluation/clahe_baseline.py
    python evaluation/clahe_baseline.py --dataset uieb
    python evaluation/clahe_baseline.py --dataset ruie
    python evaluation/clahe_baseline.py --dataset all

Output:
    results/clahe/clahe_uieb_results.csv
    results/clahe/clahe_ruie_results.csv
    results/clahe/clahe_summary.txt      ← paste this into your paper table
"""

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import (
    UIEB_RAW, UIEB_REF, UIEB_CHALLENGE,
    RUIE_ROOT, RESULTS_DIR, IMAGE_SIZE,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# CLAHE enhancement
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img_np: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    Apply CLAHE in LAB colour space — standard best practice for underwater images.
    img_np: uint8 RGB numpy array (H, W, 3)
    Returns: uint8 RGB numpy array (H, W, 3)
    """
    img_lab  = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe    = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (same formulas as your eval scripts)
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


def pct(b, a): return (a - b) / (abs(b) + 1e-9) * 100


def load_rgb(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


# ─────────────────────────────────────────────────────────────────────────────
# UIEB evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_uieb(out_dir: str, size: int):
    print("\nCLAHE baseline — UIEB-890 (paired) ...")

    if not os.path.isdir(UIEB_RAW):
        print(f"  [SKIP] UIEB_RAW not found: {UIEB_RAW}")
        return None

    raw_files = sorted(f for f in os.listdir(UIEB_RAW)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
    ref_set   = set(os.listdir(UIEB_REF)) if os.path.isdir(UIEB_REF) else set()
    pairs     = [f for f in raw_files if f in ref_set]

    if not pairs:
        print("  [SKIP] No matched pairs found.")
        return None

    records = []
    for fname in pairs:
        raw_np = load_rgb(os.path.join(UIEB_RAW, fname), size)
        ref_np = load_rgb(os.path.join(UIEB_REF, fname), size)
        enh_np = apply_clahe(raw_np)

        raw_f  = raw_np.astype(np.float32) / 255.0
        ref_f  = ref_np.astype(np.float32) / 255.0
        enh_f  = enh_np.astype(np.float32) / 255.0

        records.append({
            "image":        fname,
            "before_psnr":  peak_signal_noise_ratio(ref_f, raw_f, data_range=1.0),
            "after_psnr":   peak_signal_noise_ratio(ref_f, enh_f, data_range=1.0),
            "before_ssim":  structural_similarity(ref_f, raw_f, channel_axis=2, data_range=1.0),
            "after_ssim":   structural_similarity(ref_f, enh_f, channel_axis=2, data_range=1.0),
            "before_uciqe": compute_uciqe(raw_f),
            "after_uciqe":  compute_uciqe(enh_f),
            "before_uiqm":  compute_uiqm(raw_f),
            "after_uiqm":   compute_uiqm(enh_f),
        })

    means = {k: np.mean([r[k] for r in records]) for k in records[0] if k != "image"}

    print(f"  n={len(records)}")
    print(f"  PSNR  : {means['before_psnr']:.4f} → {means['after_psnr']:.4f}  ({pct(means['before_psnr'],  means['after_psnr']):+.2f}%)")
    print(f"  SSIM  : {means['before_ssim']:.4f} → {means['after_ssim']:.4f}  ({pct(means['before_ssim'],  means['after_ssim']):+.2f}%)")
    print(f"  UCIQE : {means['before_uciqe']:.4f} → {means['after_uciqe']:.4f}  ({pct(means['before_uciqe'], means['after_uciqe']):+.2f}%)")
    print(f"  UIQM  : {means['before_uiqm']:.4f} → {means['after_uiqm']:.4f}   ({pct(means['before_uiqm'],  means['after_uiqm']):+.2f}%)")

    # Save CSV
    csv_path = os.path.join(out_dir, "clahe_uieb_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"  Saved → {csv_path}")

    return means, len(records)


def eval_ruie(out_dir: str, size: int):
    print("\nCLAHE baseline — RUIE ...")

    data_root = Path(RUIE_ROOT)
    if not data_root.exists():
        print(f"  [SKIP] RUIE_ROOT not found: {data_root}")
        return None

    records = []
    for image_path in sorted(data_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMG_EXTS:
            continue

        relative = image_path.relative_to(data_root)
        parts    = relative.parts
        if not parts:
            continue

        subset = parts[0]
        if subset == "UTTS":
            if not (len(parts) >= 3 and parts[-2] == "JPEGImages"):
                continue
            label = f"UTTS/{parts[1]}"
        else:
            label = subset

        raw_np = load_rgb(str(image_path), size)
        enh_np = apply_clahe(raw_np)
        raw_f  = raw_np.astype(np.float32) / 255.0
        enh_f  = enh_np.astype(np.float32) / 255.0

        records.append({
            "subset":       label,
            "image":        relative.as_posix(),
            "before_uciqe": compute_uciqe(raw_f),
            "after_uciqe":  compute_uciqe(enh_f),
            "before_uiqm":  compute_uiqm(raw_f),
            "after_uiqm":   compute_uiqm(enh_f),
        })

    if not records:
        print("  [SKIP] No images found.")
        return None

    print(f"  Total images: {len(records)}")

    # Overall means
    b_u = np.mean([r["before_uciqe"] for r in records])
    a_u = np.mean([r["after_uciqe"]  for r in records])
    b_q = np.mean([r["before_uiqm"]  for r in records])
    a_q = np.mean([r["after_uiqm"]   for r in records])

    print(f"  UCIQE : {b_u:.4f} → {a_u:.4f}  ({pct(b_u, a_u):+.2f}%)")
    print(f"  UIQM  : {b_q:.4f} → {a_q:.4f}   ({pct(b_q, a_q):+.2f}%)")

    csv_path = os.path.join(out_dir, "clahe_ruie_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        w.writeheader()
        w.writerows(records)
    print(f"  Saved → {csv_path}")

    return {"before_uciqe": b_u, "after_uciqe": a_u,
            "before_uiqm":  b_q, "after_uiqm":  a_q}, len(records)


# ─────────────────────────────────────────────────────────────────────────────
# Summary for paper
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(out_dir, uieb_result, ruie_result):
    path = os.path.join(out_dir, "clahe_summary.txt")
    with open(path, "w") as f:
        f.write("CLAHE Baseline Summary (copy into paper Table 1)\n")
        f.write("=" * 52 + "\n\n")
        f.write("Method : CLAHE (clip=2.0, tile=8×8, LAB colour space)\n\n")

        if uieb_result:
            m, n = uieb_result
            f.write(f"UIEB-890 (n={n})\n")
            f.write(f"  PSNR  : {m['before_psnr']:.4f} -> {m['after_psnr']:.4f}  ({pct(m['before_psnr'],  m['after_psnr']):+.2f}%)\n")
            f.write(f"  SSIM  : {m['before_ssim']:.4f} -> {m['after_ssim']:.4f}  ({pct(m['before_ssim'],  m['after_ssim']):+.2f}%)\n")
            f.write(f"  UCIQE : {m['before_uciqe']:.4f} -> {m['after_uciqe']:.4f}  ({pct(m['before_uciqe'], m['after_uciqe']):+.2f}%)\n")
            f.write(f"  UIQM  : {m['before_uiqm']:.4f} -> {m['after_uiqm']:.4f}   ({pct(m['before_uiqm'],  m['after_uiqm']):+.2f}%)\n\n")

        if ruie_result:
            m, n = ruie_result
            f.write(f"RUIE Overall (n={n})\n")
            f.write(f"  UCIQE : {m['before_uciqe']:.4f} -> {m['after_uciqe']:.4f}  ({pct(m['before_uciqe'], m['after_uciqe']):+.2f}%)\n")
            f.write(f"  UIQM  : {m['before_uiqm']:.4f} -> {m['after_uiqm']:.4f}   ({pct(m['before_uiqm'],  m['after_uiqm']):+.2f}%)\n")

    print(f"\nSummary saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CLAHE baseline evaluation")
    p.add_argument("--dataset", choices=["all", "uieb", "ruie"], default="all")
    p.add_argument("--size",    type=int, default=IMAGE_SIZE)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = os.path.join(RESULTS_DIR, "clahe")
    os.makedirs(out_dir, exist_ok=True)

    print(f"CLAHE Baseline Evaluation")
    print(f"Image size : {args.size}×{args.size}")
    print(f"Output dir : {out_dir}")

    uieb_result = None
    ruie_result = None

    if args.dataset in ("all", "uieb"):
        uieb_result = eval_uieb(out_dir, args.size)

    if args.dataset in ("all", "ruie"):
        ruie_result = eval_ruie(out_dir, args.size)

    write_summary(out_dir, uieb_result, ruie_result)
    print("\nDone.")


if __name__ == "__main__":
    main()
