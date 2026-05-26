"""
scripts/generate_euvp_triplet_grids_all_models.py

Create separate 3 x 3 EUVP qualitative grids for three models using the same
three EUVP samples from results/figures/dataset_sample_grid_manifest.csv.

Each grid has:
  Rows    : Low-Light, Greenish Cast, Blur / Low Contrast
  Columns : Input, Preprocessed, Enhanced

Model-checkpoint pairs:
  - Model 1 -> models.model1.EnhancementCNN + checkpoints/model1_training_curves/model1_final.pth
  - Model 2 -> models.model2.EnhancementCNN + checkpoints/model2_training_curves/model2_final.pth
  - Model 3 -> models.cnn_model.EnhancementCNN + checkpoints/model_best.pth

Outputs:
  - results/euvp_triplet_grids/model1_euvp_triplet_grid.png
  - results/euvp_triplet_grids/model2_euvp_triplet_grid.png
  - results/euvp_triplet_grids/model3_euvp_triplet_grid.png
  - results/euvp_triplet_grids/euvp_triplet_grids_manifest.csv
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import IMAGE_SIZE, RESULTS_DIR
from models.model1 import EnhancementCNN as Model1
from models.model2 import EnhancementCNN as Model2
from models.model3_final import EnhancementCNN as Model3


MANIFEST_PATH = os.path.join(RESULTS_DIR, "figures", "dataset_sample_grid_manifest.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "euvp_triplet_grids")
OUT_MANIFEST = os.path.join(OUT_DIR, "euvp_triplet_grids_manifest.csv")

MODEL_SPECS = [
    {
        "name": "Model 1",
        "tag": "model1",
        "model_cls": Model1,
        "checkpoint": os.path.join(PROJECT_ROOT, "checkpoints", "model1_training_curves", "model1_final.pth"),
    },
    {
        "name": "Model 2",
        "tag": "model2",
        "model_cls": Model2,
        "checkpoint": os.path.join(PROJECT_ROOT, "checkpoints", "model2_training_curves", "model2_final.pth"),
    },
    {
        "name": "Model 3",
        "tag": "model3",
        "model_cls": Model3,
        "checkpoint": os.path.join(PROJECT_ROOT, "checkpoints", "model_best.pth"),
    },
]


def load_euvp_rows():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as file_obj:
        rows = list(csv.DictReader(file_obj))
    euvp_rows = [row for row in rows if row["dataset"] == "EUVP"]
    order = ["Low-Light", "Greenish Cast", "Blur / Low Contrast"]
    euvp_rows.sort(key=lambda row: order.index(row["degradation_type"]))
    return euvp_rows


def preprocess_for_model(img: Image.Image):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    tensor = transform(img)
    preprocessed_img = T.ToPILImage()(tensor)
    return tensor, preprocessed_img


def run_model(model, tensor, device):
    with torch.no_grad():
        out = model(tensor.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
    return T.ToPILImage()(out)


def create_grid(model_spec, rows, device):
    checkpoint = model_spec["checkpoint"]
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = model_spec["model_cls"]().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"{model_spec['name']} - EUVP Samples: Input | Preprocessed | Enhanced",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    col_labels = ["Input", "Preprocessed", "Enhanced"]
    for col_idx, label in enumerate(col_labels):
        axes[0, col_idx].set_title(label, fontsize=14, fontweight="bold", pad=10)

    for row_idx, row in enumerate(rows):
        original = Image.open(row["path"]).convert("RGB")
        input_img = original.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        input_tensor, preprocessed_img = preprocess_for_model(original)
        enhanced_img = run_model(model, input_tensor, device)

        panels = [input_img, preprocessed_img, enhanced_img]
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
                ax.set_ylabel(row["degradation_type"], fontsize=13, fontweight="bold", rotation=90)
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
    out_path = os.path.join(OUT_DIR, f"{model_spec['tag']}_euvp_triplet_grid.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_manifest(rows, outputs):
    manifest_rows = []
    for model_spec, out_path in zip(MODEL_SPECS, outputs):
        for row in rows:
            manifest_rows.append(
                {
                    "model": model_spec["name"],
                    "model_tag": model_spec["tag"],
                    "checkpoint": model_spec["checkpoint"],
                    "output_grid": out_path,
                    "dataset": row["dataset"],
                    "degradation_type": row["degradation_type"],
                    "filename": row["filename"],
                    "path": row["path"],
                }
            )

    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_euvp_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs = [create_grid(model_spec, rows, device) for model_spec in MODEL_SPECS]
    write_manifest(rows, outputs)
    for out_path in outputs:
        print(f"Saved grid -> {out_path}")
    print(f"Saved manifest -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
