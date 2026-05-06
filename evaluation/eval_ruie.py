"""
evaluation/eval_ruie.py

Full RUIE evaluation across all subsets: UCCS, UIQS, UTTS.
Clean version of mam.py — same logic, all paths from config, proper CLI.

Usage:
    python main.py eval --dataset ruie
    python main.py eval --dataset ruie --checkpoint checkpoints/model_best.pth --save-images
    python evaluation/eval_ruie.py
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
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from config import RUIE_ROOT, CHECKPOINT_FINAL, RESULTS_DIR, IMAGE_SIZE
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
# RUIE image iterator — handles UCCS, UIQS, UTTS subfolder structures
# ─────────────────────────────────────────────────────────────────────────────

def iter_ruie_images(data_root: Path):
    """
    Yields (subset_label, relative_path, absolute_path) for all RUIE images.
    UTTS images sit under <task>/<scene>/JPEGImages/; others are flat.
    """
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
            # Only process images inside JPEGImages subfolders
            if len(parts) >= 3 and parts[-2] == "JPEGImages":
                subset_label = f"UTTS/{parts[1]}"
            else:
                continue
        else:
            subset_label = subset

        yield subset_label, relative, image_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate on RUIE dataset")
    p.add_argument("--checkpoint",  type=str,  default=CHECKPOINT_FINAL)
    p.add_argument("--size",        type=int,  default=IMAGE_SIZE)
    p.add_argument("--save-images", action="store_true")
    return p.parse_args()


def main():
    args      = parse_args()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(RUIE_ROOT)
    print(f"[RUIE] Device: {device}  Checkpoint: {args.checkpoint}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not data_root.exists():
        raise FileNotFoundError(f"RUIE dataset not found: {data_root}")

    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    tfm    = T.Compose([T.Resize((args.size, args.size), antialias=True), T.ToTensor()])
    to_pil = T.ToPILImage()

    out_root = Path(RESULTS_DIR) / "RUIE"
    out_root.mkdir(parents=True, exist_ok=True)

    entries = list(iter_ruie_images(data_root))
    if not entries:
        raise RuntimeError(f"No images found under {data_root}")

    print(f"Found {len(entries)} images across all RUIE subsets.")

    records = []
    with torch.no_grad():
        for subset_label, relative, image_path in entries:
            img   = Image.open(image_path).convert("RGB")
            inp_t = tfm(img).unsqueeze(0).to(device)
            out_t = model(inp_t).clamp(0, 1)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            if args.save_images:
                save_path = out_root / relative
                save_path.parent.mkdir(parents=True, exist_ok=True)
                to_pil(out_t.squeeze(0).cpu()).save(save_path)

            records.append({
                "subset":       subset_label,
                "image":        relative.as_posix(),
                "before_uciqe": compute_uciqe(inp_np),
                "after_uciqe":  compute_uciqe(out_np),
                "before_uiqm":  compute_uiqm(inp_np),
                "after_uiqm":   compute_uiqm(out_np),
            })

    df = pd.DataFrame(records)
    df["delta_uciqe"] = df["after_uciqe"] - df["before_uciqe"]
    df["delta_uiqm"]  = df["after_uiqm"]  - df["before_uiqm"]

    # Per-subset summary
    summary = (df.groupby("subset", as_index=False)
                 .agg(images=("image", "count"),
                      before_uciqe=("before_uciqe", "mean"),
                      after_uciqe=("after_uciqe",  "mean"),
                      delta_uciqe=("delta_uciqe",  "mean"),
                      before_uiqm=("before_uiqm",  "mean"),
                      after_uiqm=("after_uiqm",    "mean"),
                      delta_uiqm=("delta_uiqm",    "mean")))

    # Overall row
    overall = pd.DataFrame([{
        "subset":       "OVERALL",
        "images":       len(df),
        "before_uciqe": df["before_uciqe"].mean(),
        "after_uciqe":  df["after_uciqe"].mean(),
        "delta_uciqe":  df["delta_uciqe"].mean(),
        "before_uiqm":  df["before_uiqm"].mean(),
        "after_uiqm":   df["after_uiqm"].mean(),
        "delta_uiqm":   df["delta_uiqm"].mean(),
    }])
    summary = pd.concat([summary, overall], ignore_index=True)

    # Print
    print()
    for _, row in summary.iterrows():
        pu = pct_change(row["before_uciqe"], row["after_uciqe"])
        pq = pct_change(row["before_uiqm"],  row["after_uiqm"])
        print(
            f"  [{row['subset']}]  n={int(row['images'])}\n"
            f"    UCIQE : {row['before_uciqe']:.4f} -> {row['after_uciqe']:.4f}  ({pu:+.2f}%)\n"
            f"    UIQM  : {row['before_uiqm']:.4f} -> {row['after_uiqm']:.4f}   ({pq:+.2f}%)"
        )

    # Save
    df.to_csv(out_root / "ruie_per_image_metrics.csv", index=False)
    summary.to_csv(out_root / "ruie_summary_metrics.csv", index=False)

    txt_path = out_root / "ruie_summary_metrics.txt"
    with open(txt_path, "w") as f:
        f.write("RUIE Evaluation Summary\n=======================\n\n")
        for _, row in summary.iterrows():
            pu = pct_change(row["before_uciqe"], row["after_uciqe"])
            pq = pct_change(row["before_uiqm"],  row["after_uiqm"])
            f.write(f"Subset : {row['subset']}\n")
            f.write(f"Images : {int(row['images'])}\n")
            f.write(f"UCIQE  : {row['before_uciqe']:.4f} -> {row['after_uciqe']:.4f} ({pu:+.2f}%)\n")
            f.write(f"UIQM   : {row['before_uiqm']:.4f} -> {row['after_uiqm']:.4f}  ({pq:+.2f}%)\n\n")

    print(f"\n  Saved -> {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()
