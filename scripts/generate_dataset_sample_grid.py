"""
scripts/generate_dataset_sample_grid.py

Create a 3 x 3 qualitative sample grid using raw underwater images only.

Rows:
  - EUVP
  - UIEB
  - RUIE

Columns:
  - Low-Light
  - Greenish Cast
  - Blur / Low Contrast

The script automatically scores images per dataset and selects one sample
for each degradation type. It then writes:
  - results/figures/dataset_sample_grid.png
  - results/figures/dataset_sample_grid_manifest.csv

Usage:
    python scripts/generate_dataset_sample_grid.py
"""

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import EUVP_VAL, RESULTS_DIR, RUIE_ROOT, UIEB_CHALLENGE


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
OUT_IMAGE = os.path.join(FIG_DIR, "dataset_sample_grid.png")
OUT_CSV = os.path.join(FIG_DIR, "dataset_sample_grid_manifest.csv")


@dataclass
class SampleCandidate:
    path: str
    brightness: float
    green_bias: float
    blur_score: float
    contrast: float


def list_images(root: str, recursive: bool = False) -> List[str]:
    if recursive:
        iterator = Path(root).rglob("*")
    else:
        iterator = Path(root).glob("*")
    return sorted(
        str(path) for path in iterator
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_features(path: str) -> SampleCandidate:
    img = load_rgb(path)
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    rgb = resized.astype(np.float32) / 255.0
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)

    r_mean = float(rgb[:, :, 0].mean())
    g_mean = float(rgb[:, :, 1].mean())
    b_mean = float(rgb[:, :, 2].mean())
    green_bias = g_mean - 0.5 * (r_mean + b_mean)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = lap_var

    return SampleCandidate(
        path=path,
        brightness=brightness,
        green_bias=green_bias,
        blur_score=blur_score,
        contrast=contrast,
    )


def build_candidates(image_paths: List[str]) -> List[SampleCandidate]:
    candidates = []
    for path in image_paths:
        try:
            candidates.append(compute_features(path))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("No readable images found for candidate selection.")
    return candidates


def score_low_light(candidate: SampleCandidate) -> float:
    return (1.0 - candidate.brightness) + 0.25 * candidate.contrast


def score_greenish(candidate: SampleCandidate) -> float:
    return candidate.green_bias + 0.15 * (1.0 - candidate.brightness)


def score_blurry_low_contrast(candidate: SampleCandidate) -> float:
    blur_term = 1.0 / (candidate.blur_score + 1.0)
    return blur_term + 0.8 * (1.0 - candidate.contrast)


def choose_samples(candidates: List[SampleCandidate]) -> Dict[str, SampleCandidate]:
    ranked = {
        "Low-Light": sorted(candidates, key=score_low_light, reverse=True),
        "Greenish Cast": sorted(candidates, key=score_greenish, reverse=True),
        "Blur / Low Contrast": sorted(
            candidates, key=score_blurry_low_contrast, reverse=True
        ),
    }

    selected: Dict[str, SampleCandidate] = {}
    used_paths = set()

    for label, ordered in ranked.items():
        chosen = None
        for candidate in ordered:
            if candidate.path not in used_paths:
                chosen = candidate
                break
        if chosen is None:
            chosen = ordered[0]
        selected[label] = chosen
        used_paths.add(chosen.path)

    return selected


def draw_grid(dataset_samples: Dict[str, Dict[str, SampleCandidate]]) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)

    row_labels = ["EUVP", "UIEB", "RUIE"]
    col_labels = ["Low-Light", "Greenish Cast", "Blur / Low Contrast"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor("white")

    for col_idx, col_label in enumerate(col_labels):
        axes[0, col_idx].set_title(col_label, fontsize=14, fontweight="bold", pad=10)

    for row_idx, row_label in enumerate(row_labels):
        for col_idx, col_label in enumerate(col_labels):
            ax = axes[row_idx, col_idx]
            sample = dataset_samples[row_label][col_label]
            img = load_rgb(sample.path)
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
                Path(sample.path).name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.6),
            )

    plt.tight_layout()
    plt.savefig(OUT_IMAGE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_manifest(dataset_samples: Dict[str, Dict[str, SampleCandidate]]) -> None:
    rows = []
    for dataset_name, col_map in dataset_samples.items():
        for degradation_label, sample in col_map.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "degradation_type": degradation_label,
                    "filename": Path(sample.path).name,
                    "path": sample.path,
                    "brightness": f"{sample.brightness:.6f}",
                    "green_bias": f"{sample.green_bias:.6f}",
                    "blur_score": f"{sample.blur_score:.6f}",
                    "contrast": f"{sample.contrast:.6f}",
                }
            )

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    dataset_roots = {
        "EUVP": (EUVP_VAL, False),
        "UIEB": (UIEB_CHALLENGE, False),
        "RUIE": (RUIE_ROOT, True),
    }

    dataset_samples: Dict[str, Dict[str, SampleCandidate]] = {}

    for dataset_name, (root, recursive) in dataset_roots.items():
        image_paths = list_images(root, recursive=recursive)
        candidates = build_candidates(image_paths)
        dataset_samples[dataset_name] = choose_samples(candidates)

    draw_grid(dataset_samples)
    write_manifest(dataset_samples)

    print(f"Saved figure   -> {OUT_IMAGE}")
    print(f"Saved manifest -> {OUT_CSV}")


if __name__ == "__main__":
    main()
