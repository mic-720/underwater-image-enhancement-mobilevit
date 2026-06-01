"""
evaluation/eval_model3_cross_dataset.py

Cross-dataset evaluation for the best Model 3 checkpoint trained with:
    0.70 * L1 + 0.25 * Edge + 0.05 * Color

Evaluates:
  - EUVP
      - No-reference EUVP validation: UCIQE, UIQM
      - Fixed held-out paired split from trainA/trainB: PSNR, SSIM
  - UIEB
      - raw-890 vs reference-890: PSNR, SSIM, UCIQE, UIQM
      - challenging-60: UCIQE, UIQM
  - RUIE
      - UCCS, UIQS, UTTS: UCIQE, UIQM

Outputs:
  - results/model3_final_cross_dataset_eval/
      - overall_summary.csv
      - overall_summary.txt
      - euvp/
      - uieb/
      - ruie/
"""

import argparse
import csv
import os
import random
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import (
    EUVP_TRAIN_A,
    EUVP_TRAIN_B,
    EUVP_VAL,
    IMAGE_SIZE,
    RESULTS_DIR,
    RUIE_ROOT,
    UIEB_CHALLENGE,
    UIEB_RAW,
    UIEB_REF,
)
from datasets.euvp_dataset import EUVPDataset
from models.model3_final import EnhancementCNN


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "model3_final_l1_700_edge_250_color_050",
    "model3_final_best.pth",
)
OUT_DIR = os.path.join(RESULTS_DIR, "model3_final_cross_dataset_eval")


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


def load_model(checkpoint: str, device: torch.device) -> EnhancementCNN:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def make_transform(image_size: int):
    return T.Compose([T.Resize((image_size, image_size), antialias=True), T.ToTensor()])


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def list_images(directory: str):
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def evaluate_euvp(model, device: torch.device, tfm, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    no_ref_records = []
    val_files = list_images(EUVP_VAL)
    with torch.no_grad():
        for fname in val_files:
            inp_t = tfm(Image.open(os.path.join(EUVP_VAL, fname)).convert("RGB")).unsqueeze(0).to(device)
            out_t = model(inp_t).clamp(0, 1)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            no_ref_records.append(
                {
                    "image": fname,
                    "before_uciqe": compute_uciqe(inp_np),
                    "after_uciqe": compute_uciqe(out_np),
                    "before_uiqm": compute_uiqm(inp_np),
                    "after_uiqm": compute_uiqm(out_np),
                }
            )

    no_ref_df = pd.DataFrame(no_ref_records)
    no_ref_df.to_csv(os.path.join(out_dir, "euvp_no_reference_per_image.csv"), index=False)

    dataset = EUVPDataset(EUVP_TRAIN_A, EUVP_TRAIN_B, augment=False, image_size=IMAGE_SIZE)
    indices = list(range(len(dataset)))
    rng = random.Random(42)
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * 0.1))
    held_out_indices = indices[:val_size]

    paired_records = []
    with torch.no_grad():
        for idx in held_out_indices:
            inp_t, tar_t = dataset[idx]
            out_t = model(inp_t.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)

            inp_np = inp_t.numpy().transpose(1, 2, 0)
            tar_np = tar_t.numpy().transpose(1, 2, 0)
            out_np = out_t.numpy().transpose(1, 2, 0)

            paired_records.append(
                {
                    "image": dataset.filenames[idx],
                    "before_psnr": peak_signal_noise_ratio(tar_np, inp_np, data_range=1.0),
                    "after_psnr": peak_signal_noise_ratio(tar_np, out_np, data_range=1.0),
                    "before_ssim": structural_similarity(tar_np, inp_np, channel_axis=2, data_range=1.0),
                    "after_ssim": structural_similarity(tar_np, out_np, channel_axis=2, data_range=1.0),
                }
            )

    paired_df = pd.DataFrame(paired_records)
    paired_df.to_csv(os.path.join(out_dir, "euvp_paired_split_per_image.csv"), index=False)

    summary_row = {
        "dataset": "EUVP",
        "subset": "validation_no_ref + held_out_paired_split",
        "images_no_ref": len(no_ref_df),
        "images_paired": len(paired_df),
        "before_uciqe": mean_or_nan(no_ref_df["before_uciqe"].tolist()),
        "after_uciqe": mean_or_nan(no_ref_df["after_uciqe"].tolist()),
        "before_uiqm": mean_or_nan(no_ref_df["before_uiqm"].tolist()),
        "after_uiqm": mean_or_nan(no_ref_df["after_uiqm"].tolist()),
        "before_psnr": mean_or_nan(paired_df["before_psnr"].tolist()),
        "after_psnr": mean_or_nan(paired_df["after_psnr"].tolist()),
        "before_ssim": mean_or_nan(paired_df["before_ssim"].tolist()),
        "after_ssim": mean_or_nan(paired_df["after_ssim"].tolist()),
        "notes": "PSNR/SSIM use fixed paired split from EUVP trainA/trainB with seed=42,val_ratio=0.1",
    }

    return no_ref_df, paired_df, [summary_row]


def evaluate_uieb(model, device: torch.device, tfm, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    raw_files = list_images(UIEB_RAW)
    ref_set = set(list_images(UIEB_REF))
    paired_files = [fname for fname in raw_files if fname in ref_set]

    paired_records = []
    with torch.no_grad():
        for fname in paired_files:
            raw_t = tfm(Image.open(os.path.join(UIEB_RAW, fname)).convert("RGB")).unsqueeze(0).to(device)
            ref_t = tfm(Image.open(os.path.join(UIEB_REF, fname)).convert("RGB"))
            out_t = model(raw_t).clamp(0, 1)

            raw_np = raw_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            ref_np = ref_t.numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            paired_records.append(
                {
                    "image": fname,
                    "before_psnr": peak_signal_noise_ratio(ref_np, raw_np, data_range=1.0),
                    "after_psnr": peak_signal_noise_ratio(ref_np, out_np, data_range=1.0),
                    "before_ssim": structural_similarity(ref_np, raw_np, channel_axis=2, data_range=1.0),
                    "after_ssim": structural_similarity(ref_np, out_np, channel_axis=2, data_range=1.0),
                    "before_uciqe": compute_uciqe(raw_np),
                    "after_uciqe": compute_uciqe(out_np),
                    "before_uiqm": compute_uiqm(raw_np),
                    "after_uiqm": compute_uiqm(out_np),
                }
            )

    paired_df = pd.DataFrame(paired_records)
    paired_df.to_csv(os.path.join(out_dir, "uieb_raw890_per_image.csv"), index=False)

    challenge_records = []
    challenge_files = list_images(UIEB_CHALLENGE)
    with torch.no_grad():
        for fname in challenge_files:
            inp_t = tfm(Image.open(os.path.join(UIEB_CHALLENGE, fname)).convert("RGB")).unsqueeze(0).to(device)
            out_t = model(inp_t).clamp(0, 1)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            challenge_records.append(
                {
                    "image": fname,
                    "before_uciqe": compute_uciqe(inp_np),
                    "after_uciqe": compute_uciqe(out_np),
                    "before_uiqm": compute_uiqm(inp_np),
                    "after_uiqm": compute_uiqm(out_np),
                }
            )

    challenge_df = pd.DataFrame(challenge_records)
    challenge_df.to_csv(os.path.join(out_dir, "uieb_challenging60_per_image.csv"), index=False)

    summary_rows = [
        {
            "dataset": "UIEB",
            "subset": "raw-890",
            "images_no_ref": len(paired_df),
            "images_paired": len(paired_df),
            "before_uciqe": mean_or_nan(paired_df["before_uciqe"].tolist()),
            "after_uciqe": mean_or_nan(paired_df["after_uciqe"].tolist()),
            "before_uiqm": mean_or_nan(paired_df["before_uiqm"].tolist()),
            "after_uiqm": mean_or_nan(paired_df["after_uiqm"].tolist()),
            "before_psnr": mean_or_nan(paired_df["before_psnr"].tolist()),
            "after_psnr": mean_or_nan(paired_df["after_psnr"].tolist()),
            "before_ssim": mean_or_nan(paired_df["before_ssim"].tolist()),
            "after_ssim": mean_or_nan(paired_df["after_ssim"].tolist()),
            "notes": "Full-reference and no-reference metrics on UIEB raw-890",
        },
        {
            "dataset": "UIEB",
            "subset": "challenging-60",
            "images_no_ref": len(challenge_df),
            "images_paired": 0,
            "before_uciqe": mean_or_nan(challenge_df["before_uciqe"].tolist()),
            "after_uciqe": mean_or_nan(challenge_df["after_uciqe"].tolist()),
            "before_uiqm": mean_or_nan(challenge_df["before_uiqm"].tolist()),
            "after_uiqm": mean_or_nan(challenge_df["after_uiqm"].tolist()),
            "before_psnr": float("nan"),
            "after_psnr": float("nan"),
            "before_ssim": float("nan"),
            "after_ssim": float("nan"),
            "notes": "No reference images available for PSNR/SSIM on challenging-60",
        },
    ]

    return paired_df, challenge_df, summary_rows


def iter_ruie_images(data_root: Path):
    for image_path in sorted(data_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMG_EXTS:
            continue

        relative = image_path.relative_to(data_root)
        parts = relative.parts
        if not parts:
            continue

        subset = parts[0]
        if subset == "UTTS":
            if len(parts) >= 3 and parts[-2] == "JPEGImages":
                subset_label = f"UTTS/{parts[1]}"
            else:
                continue
        else:
            subset_label = subset

        yield subset_label, relative.as_posix(), image_path


def evaluate_ruie(model, device: torch.device, tfm, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    records = []
    with torch.no_grad():
        for subset_label, relative_path, image_path in iter_ruie_images(Path(RUIE_ROOT)):
            inp_t = tfm(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            out_t = model(inp_t).clamp(0, 1)

            inp_np = inp_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            out_np = out_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            records.append(
                {
                    "subset": subset_label,
                    "image": relative_path,
                    "before_uciqe": compute_uciqe(inp_np),
                    "after_uciqe": compute_uciqe(out_np),
                    "before_uiqm": compute_uiqm(inp_np),
                    "after_uiqm": compute_uiqm(out_np),
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, "ruie_per_image.csv"), index=False)

    summary_rows = []
    grouped = df.groupby("subset", as_index=False).mean(numeric_only=True)
    counts = df.groupby("subset")["image"].count().to_dict()
    for _, row in grouped.iterrows():
        summary_rows.append(
            {
                "dataset": "RUIE",
                "subset": row["subset"],
                "images_no_ref": int(counts[row["subset"]]),
                "images_paired": 0,
                "before_uciqe": float(row["before_uciqe"]),
                "after_uciqe": float(row["after_uciqe"]),
                "before_uiqm": float(row["before_uiqm"]),
                "after_uiqm": float(row["after_uiqm"]),
                "before_psnr": float("nan"),
                "after_psnr": float("nan"),
                "before_ssim": float("nan"),
                "after_ssim": float("nan"),
                "notes": "No reference images available for PSNR/SSIM on RUIE",
            }
        )

    summary_rows.append(
        {
            "dataset": "RUIE",
            "subset": "OVERALL",
            "images_no_ref": len(df),
            "images_paired": 0,
            "before_uciqe": mean_or_nan(df["before_uciqe"].tolist()),
            "after_uciqe": mean_or_nan(df["after_uciqe"].tolist()),
            "before_uiqm": mean_or_nan(df["before_uiqm"].tolist()),
            "after_uiqm": mean_or_nan(df["after_uiqm"].tolist()),
            "before_psnr": float("nan"),
            "after_psnr": float("nan"),
            "before_ssim": float("nan"),
            "after_ssim": float("nan"),
            "notes": "No reference images available for PSNR/SSIM on RUIE",
        }
    )

    return df, summary_rows


def add_improvement_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["uciqe_change_pct"] = df.apply(
        lambda row: pct_change(row["before_uciqe"], row["after_uciqe"])
        if pd.notna(row["before_uciqe"]) and pd.notna(row["after_uciqe"])
        else float("nan"),
        axis=1,
    )
    df["uiqm_change_pct"] = df.apply(
        lambda row: pct_change(row["before_uiqm"], row["after_uiqm"])
        if pd.notna(row["before_uiqm"]) and pd.notna(row["after_uiqm"])
        else float("nan"),
        axis=1,
    )
    df["psnr_change_pct"] = df.apply(
        lambda row: pct_change(row["before_psnr"], row["after_psnr"])
        if pd.notna(row["before_psnr"]) and pd.notna(row["after_psnr"])
        else float("nan"),
        axis=1,
    )
    df["ssim_change_pct"] = df.apply(
        lambda row: pct_change(row["before_ssim"], row["after_ssim"])
        if pd.notna(row["before_ssim"]) and pd.notna(row["after_ssim"])
        else float("nan"),
        axis=1,
    )
    return df


def write_text_summary(summary_df: pd.DataFrame, checkpoint: str, out_path: str):
    lines = []
    lines.append("Model 3 Cross-Dataset Evaluation Summary")
    lines.append("========================================")
    lines.append(f"Checkpoint: {checkpoint}")
    lines.append("")

    for _, row in summary_df.iterrows():
        lines.append(f"Dataset : {row['dataset']}")
        lines.append(f"Subset  : {row['subset']}")
        lines.append(f"No-Ref N: {int(row['images_no_ref'])}")
        lines.append(f"Paired N: {int(row['images_paired'])}")
        lines.append(
            f"UCIQE   : {row['before_uciqe']:.4f} -> {row['after_uciqe']:.4f}"
            if pd.notna(row["before_uciqe"]) and pd.notna(row["after_uciqe"])
            else "UCIQE   : N/A"
        )
        lines.append(
            f"UIQM    : {row['before_uiqm']:.4f} -> {row['after_uiqm']:.4f}"
            if pd.notna(row["before_uiqm"]) and pd.notna(row["after_uiqm"])
            else "UIQM    : N/A"
        )
        lines.append(
            f"PSNR    : {row['before_psnr']:.4f} -> {row['after_psnr']:.4f}"
            if pd.notna(row["before_psnr"]) and pd.notna(row["after_psnr"])
            else "PSNR    : N/A"
        )
        lines.append(
            f"SSIM    : {row['before_ssim']:.4f} -> {row['after_ssim']:.4f}"
            if pd.notna(row["before_ssim"]) and pd.notna(row["after_ssim"])
            else "SSIM    : N/A"
        )
        lines.append(f"Notes   : {row['notes']}")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation for Model 3 best checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    os.makedirs(OUT_DIR, exist_ok=True)
    euvp_out = os.path.join(OUT_DIR, "euvp")
    uieb_out = os.path.join(OUT_DIR, "uieb")
    ruie_out = os.path.join(OUT_DIR, "ruie")
    for directory in (euvp_out, uieb_out, ruie_out):
        os.makedirs(directory, exist_ok=True)

    model = load_model(args.checkpoint, device)
    tfm = make_transform(args.size)

    print("Evaluating EUVP ...")
    _, _, euvp_summary_rows = evaluate_euvp(model, device, tfm, euvp_out)

    print("Evaluating UIEB ...")
    _, _, uieb_summary_rows = evaluate_uieb(model, device, tfm, uieb_out)

    print("Evaluating RUIE ...")
    _, ruie_summary_rows = evaluate_ruie(model, device, tfm, ruie_out)

    summary_df = pd.DataFrame(euvp_summary_rows + uieb_summary_rows + ruie_summary_rows)
    summary_df = add_improvement_columns(summary_df)

    csv_path = os.path.join(OUT_DIR, "overall_summary.csv")
    txt_path = os.path.join(OUT_DIR, "overall_summary.txt")
    summary_df.to_csv(csv_path, index=False)
    write_text_summary(summary_df, args.checkpoint, txt_path)

    print(f"Saved summary CSV -> {csv_path}")
    print(f"Saved summary TXT -> {txt_path}")


if __name__ == "__main__":
    main()
