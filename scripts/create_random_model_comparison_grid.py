"""
scripts/create_random_model_comparison_grid.py

Create a single 1 x 4 comparison grid using one random EUVP validation image:
    Input | Model 1 Enhanced | Model 2 Enhanced | Model 3 Enhanced

Outputs:
  - results/figures/random_model_comparison_grid.png
  - results/figures/random_model_comparison_grid_manifest.txt
"""

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from config import EUVP_VAL, IMAGE_SIZE, RESULTS_DIR
from models.model1 import EnhancementCNN as Model1
from models.model2 import EnhancementCNN as Model2
from models.model3_final import EnhancementCNN as Model3


OUT_DIR = os.path.join(RESULTS_DIR, "samples")
OUT_IMAGE = os.path.join(OUT_DIR, "random_model_comparison_grid_v2.png")
OUT_MANIFEST = os.path.join(OUT_DIR, "random_model_comparison_grid_v2_manifest.txt")

MODEL_SPECS = [
    {
        "label": "Input",
        "model_cls": None,
        "checkpoint": None,
    },
    {
        "label": "Model 1 Enhanced",
        "model_cls": Model1,
        "checkpoint": os.path.join(PROJECT_ROOT, "checkpoints", "model1_training_curves", "model1_final.pth"),
    },
    {
        "label": "Model 2 Enhanced",
        "model_cls": Model2,
        "checkpoint": os.path.join(PROJECT_ROOT, "checkpoints", "model2_training_curves", "model2_final.pth"),
    },
    {
        "label": "Model 3 Enhanced",
        "model_cls": Model3,
        "checkpoint": os.path.join(
            PROJECT_ROOT,
            "checkpoints",
            "model3_final_training_curves",
            "model3_final_best.pth",
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a random 4-panel comparison grid for Models 1, 2, and 3"
    )
    parser.add_argument("--input", type=str, default=None, help="Optional explicit image path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for image selection")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Panel image size")
    return parser.parse_args()


def get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def list_validation_images():
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        os.path.join(EUVP_VAL, name)
        for name in sorted(os.listdir(EUVP_VAL))
        if os.path.splitext(name)[1].lower() in valid_exts
    ]
    if not files:
        raise FileNotFoundError(f"No validation images found in: {EUVP_VAL}")
    return files


def choose_image(explicit_path: str | None, seed: int) -> str:
    if explicit_path:
        if not os.path.isfile(explicit_path):
            raise FileNotFoundError(f"Input image not found: {explicit_path}")
        return explicit_path

    files = list_validation_images()
    rng = random.Random(seed)
    return rng.choice(files)


def load_model(model_cls, checkpoint: str, device: torch.device):
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = model_cls().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def run_model(model, image: Image.Image, size: int, device: torch.device) -> Image.Image:
    tfm = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor()])
    tensor = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor).squeeze(0).cpu().clamp(0, 1)
    return T.ToPILImage()(output)


def overlay_label(img: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    labelled = img.copy()
    draw = ImageDraw.Draw(labelled)
    padding = 10
    left = padding
    top = padding
    text_bbox = draw.textbbox((left, top), text, font=font)
    box = (
        text_bbox[0] - 8,
        text_bbox[1] - 6,
        text_bbox[2] + 8,
        text_bbox[3] + 6,
    )
    draw.rounded_rectangle(box, radius=8, fill=(0, 0, 0, 150))
    draw.text((left, top), text, fill="white", font=font)
    return labelled


def build_grid(input_path: str, size: int, seed: int, device: torch.device):
    os.makedirs(OUT_DIR, exist_ok=True)
    title_font = get_font(36)
    label_font = get_font(20)
    meta_font = get_font(18)

    original = Image.open(input_path).convert("RGB")
    input_panel = original.resize((size, size), Image.Resampling.LANCZOS)

    panels = [overlay_label(input_panel, "Input", label_font)]
    for spec in MODEL_SPECS[1:]:
        model = load_model(spec["model_cls"], spec["checkpoint"], device)
        enhanced = run_model(model, original, size, device)
        panels.append(overlay_label(enhanced, spec["label"], label_font))

    gap = 24
    margin_x = 40
    title_h = 70
    footer_h = 52
    width = margin_x * 2 + size * len(panels) + gap * (len(panels) - 1)
    height = title_h + size + footer_h + 30

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = "Random EUVP Image Comparison Across Three Models"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (width - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, 16), title, fill=(88, 38, 21), font=title_font)

    y = title_h
    for idx, panel in enumerate(panels):
        x = margin_x + idx * (size + gap)
        canvas.paste(panel, (x, y))

    footer = f"Sample: {Path(input_path).name}"
    footer_bbox = draw.textbbox((0, 0), footer, font=meta_font)
    footer_x = (width - (footer_bbox[2] - footer_bbox[0])) // 2
    draw.text((footer_x, title_h + size + 14), footer, fill="black", font=meta_font)

    canvas.save(OUT_IMAGE)
    with open(OUT_MANIFEST, "w", encoding="utf-8") as file_obj:
        file_obj.write(f"selected_image={input_path}\n")
        file_obj.write(f"seed={seed}\n")
        for spec in MODEL_SPECS[1:]:
            file_obj.write(f"{spec['label']} checkpoint={spec['checkpoint']}\n")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chosen_image = choose_image(args.input, args.seed)
    build_grid(chosen_image, args.size, args.seed, device)
    print(f"Saved grid -> {OUT_IMAGE}")
    print(f"Saved manifest -> {OUT_MANIFEST}")
