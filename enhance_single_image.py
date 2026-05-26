"""
Enhance a single image with the final Model 3 checkpoint.

Example:
    python enhance_single_image.py --input path/to/photo.jpg
    python enhance_single_image.py --input path/to/photo.jpg --output-dir results/my_run
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from config import CHECKPOINT_BEST, IMAGE_SIZE
from models.model3_final import EnhancementCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance a single image using Model 3.")
    parser.add_argument("--input", required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint", default=CHECKPOINT_BEST, help="Checkpoint path.")
    parser.add_argument("--output-dir", default="resultsSingle", help="Folder for saved outputs.")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Inference image size.")
    return parser.parse_args()


def get_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def add_label(img: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    draw.text((11, 11), text, fill=(0, 0, 0), font=font)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font)
    return canvas


def load_model(checkpoint: str, device: torch.device) -> EnhancementCNN:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def main():
    args = parse_args()
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    transform = T.Compose([T.Resize((args.size, args.size), antialias=True), T.ToTensor()])
    to_pil = T.ToPILImage()
    font = get_font(18)

    original = Image.open(args.input).convert("RGB")
    input_tensor = transform(original).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu().clamp(0, 1)

    input_img = to_pil(input_tensor.squeeze(0).cpu())
    enhanced_img = to_pil(output)
    labelled_input = add_label(input_img, "Input", font)
    labelled_enhanced = add_label(enhanced_img, "Enhanced", font)

    comparison = Image.new("RGB", (args.size * 2, args.size), (255, 255, 255))
    comparison.paste(labelled_input, (0, 0))
    comparison.paste(labelled_enhanced, (args.size, 0))

    stem = Path(args.input).stem
    input_path = os.path.join(args.output_dir, f"{stem}_input.png")
    enhanced_path = os.path.join(args.output_dir, f"{stem}_enhanced.png")
    comparison_path = os.path.join(args.output_dir, f"{stem}_comparison.png")

    labelled_input.save(input_path)
    enhanced_img.save(enhanced_path)
    comparison.save(comparison_path)

    print(f"Saved input      -> {input_path}")
    print(f"Saved enhanced   -> {enhanced_path}")
    print(f"Saved comparison -> {comparison_path}")


if __name__ == "__main__":
    main()
