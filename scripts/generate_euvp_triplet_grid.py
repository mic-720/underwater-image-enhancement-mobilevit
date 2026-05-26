"""
scripts/generate_euvp_triplet_grid.py

Create a 3 x 3 figure for the three EUVP samples selected in
results/figures/dataset_sample_grid_manifest.csv.

Rows:
  - Low-Light
  - Greenish Cast
  - Blur / Low Contrast

Columns:
  - Image
  - Preprocessed
  - Enhanced

The enhancement step uses the specified checkpoint, defaulting to model1.pth.

Usage:
    python scripts/generate_euvp_triplet_grid.py
    python scripts/generate_euvp_triplet_grid.py --checkpoint checkpoints/model_best.pth
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
import torch
from PIL import Image
import torchvision.transforms as T

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import IMAGE_SIZE, RESULTS_DIR
from models.cnn_model import EnhancementCNN


MANIFEST_PATH = os.path.join(RESULTS_DIR, "figures", "dataset_sample_grid_manifest.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "figures", "euvp_image_preprocessed_enhanced_grid.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate EUVP image-preprocessed-enhanced grid")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "model1.pth"),
        help="Checkpoint to use for enhancement",
    )
    return parser.parse_args()


def load_manifest_rows():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    euvp_rows = [row for row in rows if row["dataset"] == "EUVP"]
    if len(euvp_rows) != 3:
        raise RuntimeError(f"Expected 3 EUVP rows in manifest, found {len(euvp_rows)}.")
    order = ["Low-Light", "Greenish Cast", "Blur / Low Contrast"]
    euvp_rows.sort(key=lambda row: order.index(row["degradation_type"]))
    return euvp_rows


def load_original(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_for_model(img: Image.Image):
    preprocess = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    tensor = preprocess(img)
    preprocessed_vis = T.ToPILImage()(tensor)
    return tensor, preprocessed_vis


def run_enhancement(model, tensor, device):
    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
    return T.ToPILImage()(output)


def main():
    args = parse_args()
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    rows = load_manifest_rows()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle("EUVP Samples: Image | Preprocessed | Enhanced", fontsize=16, fontweight="bold", y=0.98)

    col_labels = ["Image", "Preprocessed", "Enhanced"]
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=14, fontweight="bold", pad=10)

    for row_idx, row in enumerate(rows):
        original = load_original(row["path"])
        original_resized = original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        input_tensor, preprocessed_img = preprocess_for_model(original)
        enhanced_img = run_enhancement(model, input_tensor, device)

        panels = [original_resized, preprocessed_img, enhanced_img]
        row_label = row["degradation_type"]

        for col_idx, panel_img in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel_img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor("#222222")
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=13, fontweight="bold", rotation=90)
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
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {OUT_PATH}")


if __name__ == "__main__":
    main()
