"""
scripts/generate_preprocessed_sample_grid.py

Generate a 3 x 3 grid using the same dataset samples selected in
results/figures/dataset_sample_grid_manifest.csv, but after applying the
repo's actual preprocessing steps.

Actual preprocessing reflected here:
  - RGB conversion
  - resize to IMAGE_SIZE x IMAGE_SIZE
  - tensor-style scaling to [0, 1] and conversion back for visualization

Optional visualization mode:
  - deterministic flip + brightness/contrast/color adjustment preview
    to mimic training-time augmentation in a visible way

Usage:
    python scripts/generate_preprocessed_sample_grid.py
    python scripts/generate_preprocessed_sample_grid.py --mode augmented
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import IMAGE_SIZE, RESULTS_DIR


FIG_DIR = os.path.join(RESULTS_DIR, "figures")
MANIFEST_PATH = os.path.join(FIG_DIR, "dataset_sample_grid_manifest.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate preprocessed dataset sample grid")
    parser.add_argument(
        "--mode",
        choices=["standard", "augmented"],
        default="standard",
        help="Use standard preprocessing only or a visible augmentation preview",
    )
    return parser.parse_args()


def load_manifest():
    if not os.path.isfile(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}. Run generate_dataset_sample_grid.py first."
        )

    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(row)
    return rows


def preprocess_image(path: str, mode: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

    if mode == "augmented":
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img = ImageEnhance.Brightness(img).enhance(1.10)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.12)

    # Mirror the tensor conversion step used by the dataset pipeline.
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def build_lookup(rows):
    lookup = {}
    for row in rows:
        lookup[(row["dataset"], row["degradation_type"])] = row
    return lookup


def draw_grid(lookup, mode: str):
    row_labels = ["EUVP", "UIEB", "RUIE"]
    col_labels = ["Low-Light", "Greenish Cast", "Blur / Low Contrast"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor("white")

    title = "Dataset Sample Grid After Preprocessing"
    if mode == "augmented":
        title += " (Augmentation Preview)"
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for col_idx, col_label in enumerate(col_labels):
        axes[0, col_idx].set_title(col_label, fontsize=14, fontweight="bold", pad=10)

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            ax = axes[row_idx, col_idx]
            row = lookup[(row_label, col_label)]
            img = preprocess_image(row["path"], mode=mode)
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor("#222222")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=14, fontweight="bold", rotation=90)
            ax.text(
                0.02,
                0.96,
                Path(row["filename"]).name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = "dataset_sample_grid_preprocessed.png"
    if mode == "augmented":
        out_name = "dataset_sample_grid_augmented_preview.png"
    out_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    rows = load_manifest()
    lookup = build_lookup(rows)
    out_path = draw_grid(lookup, args.mode)
    print(f"Saved figure -> {out_path}")


if __name__ == "__main__":
    main()
