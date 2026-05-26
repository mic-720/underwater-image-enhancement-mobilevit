"""
evaluation/compare_euvp_models_table.py

Compare three model checkpoints on EUVP using:
  - No-reference EUVP validation set for UCIQE/UIQM before/after
  - A fixed paired split from EUVP trainA/trainB for PSNR/SSIM

Model-checkpoint pairs:
  - Model 1 -> models.model1.EnhancementCNN + checkpoints/model1_training_curves/model1_final.pth
  - Model 2 -> models.model2.EnhancementCNN + checkpoints/model2_training_curves/model2_final.pth
  - Model 3 -> models.model3_final.EnhancementCNN + checkpoints/model_best.pth

Outputs:
  - results/euvp_model_comparison/euvp_model_comparison.csv
  - results/euvp_model_comparison/euvp_model_comparison_table.txt
"""

import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import EUVP_TRAIN_A, EUVP_TRAIN_B, EUVP_VAL, IMAGE_SIZE, RESULTS_DIR
from datasets.euvp_dataset import EUVPDataset
from models.model1 import EnhancementCNN as Model1
from models.model2 import EnhancementCNN as Model2
from models.model3_final import EnhancementCNN as Model3


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_DIR = os.path.join(RESULTS_DIR, "euvp_model_comparison")
CSV_PATH = os.path.join(OUT_DIR, "euvp_model_comparison.csv")
TXT_PATH = os.path.join(OUT_DIR, "euvp_model_comparison_table.txt")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_cls: type
    checkpoint: str


MODEL_SPECS = [
    ModelSpec(
        name="Model 1",
        model_cls=Model1,
        checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "model1_training_curves", "model1_final.pth"),
    ),
    ModelSpec(
        name="Model 2",
        model_cls=Model2,
        checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "model2_training_curves", "model2_final.pth"),
    ),
    ModelSpec(
        name="Model 3",
        model_cls=Model3,
        checkpoint=os.path.join(PROJECT_ROOT, "checkpoints", "model_best.pth"),
    ),
]


def compute_uciqe(img: np.ndarray) -> float:
    img = np.clip(img, 0.0, 1.0)
    img_lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(img_lab)
    chroma = np.sqrt(a_chan ** 2 + b_chan ** 2)
    return (
        0.4680 * np.std(chroma)
        + 0.2745 * (np.max(l_chan) - np.min(l_chan)) / 255.0
        + 0.2576 * np.mean(chroma) / 255.0
    )


def compute_uiqm(img: np.ndarray) -> float:
    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255).astype(np.uint8)
    rg = img_u8[:, :, 0].astype(np.float32) - img_u8[:, :, 1].astype(np.float32)
    yb = (
        0.5 * (img_u8[:, :, 0].astype(np.float32) + img_u8[:, :, 1].astype(np.float32))
        - img_u8[:, :, 2].astype(np.float32)
    )
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    return (
        0.0282 * (np.std(rg) + np.std(yb))
        + 0.2953 * np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1)))
        + 3.5753 * np.std(gray)
    ) / 255.0


def pct_change(before: float, after: float) -> float:
    return (after - before) / (abs(before) + 1e-9) * 100.0


def load_model(spec: ModelSpec, device: torch.device):
    if not os.path.isfile(spec.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found for {spec.name}: {spec.checkpoint}")
    model = spec.model_cls().to(device)
    model.load_state_dict(torch.load(spec.checkpoint, map_location=device))
    model.eval()
    return model


def evaluate_no_reference(model, device: torch.device):
    tfm = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True), T.ToTensor()])
    files = sorted(
        f for f in os.listdir(EUVP_VAL)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )

    before_u, after_u, before_q, after_q = [], [], [], []

    with torch.no_grad():
        for fname in files:
            inp_t = tfm(Image.open(os.path.join(EUVP_VAL, fname)).convert("RGB")).unsqueeze(0).to(device)
            out_t = model(inp_t).clamp(0, 1)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            before_u.append(compute_uciqe(inp_np))
            after_u.append(compute_uciqe(out_np))
            before_q.append(compute_uiqm(inp_np))
            after_q.append(compute_uiqm(out_np))

    return {
        "images_no_ref": len(files),
        "uciqe_before": float(np.mean(before_u)),
        "uciqe_after": float(np.mean(after_u)),
        "uiqm_before": float(np.mean(before_q)),
        "uiqm_after": float(np.mean(after_q)),
    }


def build_paired_eval_pairs(seed: int = 42, val_ratio: float = 0.1):
    dataset = EUVPDataset(EUVP_TRAIN_A, EUVP_TRAIN_B, augment=False, image_size=IMAGE_SIZE)
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_indices = indices[:val_size]
    return dataset, val_indices


def evaluate_paired_metrics(model, device: torch.device, dataset, indices):
    psnr_before_scores = []
    psnr_after_scores = []
    ssim_before_scores = []
    ssim_after_scores = []

    with torch.no_grad():
        for idx in indices:
            inp_t, tar_t = dataset[idx]
            out_t = model(inp_t.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)

            inp_np = inp_t.numpy().transpose(1, 2, 0)
            tar_np = tar_t.numpy().transpose(1, 2, 0)
            out_np = out_t.numpy().transpose(1, 2, 0)

            psnr_before_scores.append(
                peak_signal_noise_ratio(tar_np, inp_np, data_range=1.0)
            )
            psnr_after_scores.append(
                peak_signal_noise_ratio(tar_np, out_np, data_range=1.0)
            )
            ssim_before_scores.append(
                structural_similarity(tar_np, inp_np, channel_axis=2, data_range=1.0)
            )
            ssim_after_scores.append(
                structural_similarity(tar_np, out_np, channel_axis=2, data_range=1.0)
            )

    return {
        "images_paired": len(indices),
        "psnr_before": float(np.mean(psnr_before_scores)),
        "psnr_after": float(np.mean(psnr_after_scores)),
        "ssim_before": float(np.mean(ssim_before_scores)),
        "ssim_after": float(np.mean(ssim_after_scores)),
    }


def format_table(results):
    lines = []
    lines.append("EUVP Performance Evaluation Table")
    lines.append("=" * 132)
    lines.append(
        f"{'Model':<10} {'No-Ref N':>8} {'Paired N':>8} "
        f"{'UCIQE Before':>14} {'UCIQE After':>13} "
        f"{'UIQM Before':>13} {'UIQM After':>12} "
        f"{'PSNR':>8} {'SSIM':>8}"
    )
    lines.append("-" * 132)
    for row in results:
        lines.append(
            f"{row['model']:<10} {row['images_no_ref']:>8} {row['images_paired']:>8} "
            f"{row['uciqe_before']:>14.4f} {row['uciqe_after']:>13.4f} "
            f"{row['uiqm_before']:>13.4f} {row['uiqm_after']:>12.4f} "
            f"{row['psnr_after']:>8.4f} {row['ssim_after']:>8.4f}"
        )
    lines.append("=" * 132)
    lines.append("PSNR/SSIM are computed on a fixed held-out paired split from EUVP trainA/trainB (seed=42, val_ratio=0.1).")
    lines.append("UCIQE/UIQM are computed on the EUVP validation folder as before/after no-reference metrics.")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    paired_dataset, paired_indices = build_paired_eval_pairs(seed=42, val_ratio=0.1)

    results = []
    for spec in MODEL_SPECS:
        print(f"Evaluating {spec.name} ...")
        model = load_model(spec, device)
        no_ref = evaluate_no_reference(model, device)
        paired = evaluate_paired_metrics(model, device, paired_dataset, paired_indices)

        results.append(
            {
                "model": spec.name,
                "checkpoint": spec.checkpoint,
                **no_ref,
                **paired,
                "uciqe_improve_pct": pct_change(no_ref["uciqe_before"], no_ref["uciqe_after"]),
                "uiqm_improve_pct": pct_change(no_ref["uiqm_before"], no_ref["uiqm_after"]),
                "psnr_improve_pct": pct_change(paired["psnr_before"], paired["psnr_after"]),
                "ssim_improve_pct": pct_change(paired["ssim_before"], paired["ssim_after"]),
            }
        )
        print(
            f"  UCIQE {no_ref['uciqe_before']:.4f} -> {no_ref['uciqe_after']:.4f} | "
            f"UIQM {no_ref['uiqm_before']:.4f} -> {no_ref['uiqm_after']:.4f} | "
            f"PSNR {paired['psnr_after']:.4f} | SSIM {paired['ssim_after']:.4f}"
        )

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    with open(TXT_PATH, "w", encoding="utf-8") as file_obj:
        file_obj.write(format_table(results))
        file_obj.write("\n")

    print(f"\nSaved CSV -> {CSV_PATH}")
    print(f"Saved TXT -> {TXT_PATH}")


if __name__ == "__main__":
    main()
