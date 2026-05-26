"""
scripts/generate_euvp_challenging_case_grids.py

Generate one challenging paired EUVP comparison grid per model:
    Input | Preprocessed | Ground Truth | Enhanced

The selected sample is a visually challenging EUVP paired image that shows:
  - dense coral texture
  - strong underwater blue cast
  - wide colour variation
  - contrast/detail restoration difficulty

Outputs:
  - results/euvp_challenging_case/model1_euvp_challenging_case.png
  - results/euvp_challenging_case/model2_euvp_challenging_case.png
  - results/euvp_challenging_case/model3_euvp_challenging_case.png
  - results/euvp_challenging_case/euvp_challenging_case_manifest.csv
  - results/euvp_challenging_case/euvp_challenging_case_notes.txt
"""

import csv
import os
import sys
from dataclasses import dataclass

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import EUVP_TRAIN_A, EUVP_TRAIN_B, IMAGE_SIZE, RESULTS_DIR
from models.model1 import EnhancementCNN as Model1
from models.model2 import EnhancementCNN as Model2
from models.model3_final import EnhancementCNN as Model3


OUT_DIR = os.path.join(RESULTS_DIR, "euvp_challenging_case")
MANIFEST_PATH = os.path.join(OUT_DIR, "euvp_challenging_case_manifest.csv")
NOTES_PATH = os.path.join(OUT_DIR, "euvp_challenging_case_notes.txt")
CHALLENGING_FILENAME = "im_f1210_.jpg"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    tag: str
    model_cls: type
    checkpoint: str


MODEL_SPECS = [
    ModelSpec(
        name="Model 1",
        tag="model1",
        model_cls=Model1,
        checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "model1_training_curves", "model1_final.pth"),
    ),
    ModelSpec(
        name="Model 2",
        tag="model2",
        model_cls=Model2,
        checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "model2_training_curves", "model2_final.pth"),
    ),
    ModelSpec(
        name="Model 3",
        tag="model3",
        model_cls=Model3,
        checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "model_best.pth"),
    ),
]


def load_rgb(path: str):
    return Image.open(path).convert("RGB")


def preprocess_for_display(img: Image.Image):
    tfm = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True), T.ToTensor()])
    tensor = tfm(img)
    return tensor, tensor.permute(1, 2, 0).numpy()


def load_model(spec: ModelSpec, device: torch.device):
    model = spec.model_cls().to(device)
    model.load_state_dict(torch.load(spec.checkpoint, map_location=device))
    model.eval()
    return model


def save_grid(spec: ModelSpec, input_np, pre_np, gt_np, enhanced_np, psnr, ssim):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
    panels = [
        ("Input", input_np),
        ("Preprocessed", pre_np),
        ("Ground Truth", gt_np),
        ("Enhanced", enhanced_np),
    ]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"{spec.name} - Challenging EUVP Case ({CHALLENGING_FILENAME})",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.03,
        f"PSNR: {psnr:.4f}    SSIM: {ssim:.4f}",
        ha="center",
        va="center",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])

    out_path = os.path.join(OUT_DIR, f"{spec.tag}_euvp_challenging_case.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_path = os.path.join(EUVP_TRAIN_A, CHALLENGING_FILENAME)
    gt_path = os.path.join(EUVP_TRAIN_B, CHALLENGING_FILENAME)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Missing challenging input image: {input_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"Missing challenging GT image: {gt_path}")

    input_img = load_rgb(input_path)
    gt_img = load_rgb(gt_path)

    input_t, pre_np = preprocess_for_display(input_img)
    gt_t, gt_np = preprocess_for_display(gt_img)
    input_np = input_t.permute(1, 2, 0).numpy()

    rows = []
    for spec in MODEL_SPECS:
        model = load_model(spec, device)
        with torch.no_grad():
            enhanced_t = model(input_t.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)
        enhanced_np = enhanced_t.permute(1, 2, 0).numpy()

        psnr = peak_signal_noise_ratio(gt_t.numpy().transpose(1, 2, 0), enhanced_np, data_range=1.0)
        ssim = structural_similarity(
            gt_t.numpy().transpose(1, 2, 0),
            enhanced_np,
            channel_axis=2,
            data_range=1.0,
        )

        output_grid = save_grid(spec, input_np, pre_np, gt_np, enhanced_np, psnr, ssim)
        rows.append(
            {
                "model": spec.name,
                "model_tag": spec.tag,
                "checkpoint": spec.checkpoint,
                "filename": CHALLENGING_FILENAME,
                "input_path": input_path,
                "gt_path": gt_path,
                "output_grid": output_grid,
                "psnr": f"{psnr:.6f}",
                "ssim": f"{ssim:.6f}",
            }
        )

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(NOTES_PATH, "w", encoding="utf-8") as file_obj:
        file_obj.write("Challenging EUVP Case Selection\n")
        file_obj.write("===============================\n\n")
        file_obj.write(f"Selected file: {CHALLENGING_FILENAME}\n")
        file_obj.write("Dataset split : EUVP paired trainA/trainB\n\n")
        file_obj.write("Why this image is challenging:\n")
        file_obj.write("- It contains dense coral texture and fine structures.\n")
        file_obj.write("- It has a strong blue underwater tone and wide colour variation.\n")
        file_obj.write("- It is useful for showing both robustness and limitations in colour restoration.\n")
        file_obj.write("- It allows comparison against paired ground truth.\n\n")
        file_obj.write("Interpretation focus:\n")
        file_obj.write("- colour restoration\n")
        file_obj.write("- contrast recovery\n")
        file_obj.write("- preservation of fine details\n")
        file_obj.write("- tendency toward oversaturation or remaining colour bias\n")

    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
