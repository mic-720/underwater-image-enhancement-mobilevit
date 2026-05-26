"""
Simple photo-to-enhanced-image pipeline for the final underwater enhancement model.

Examples:
    python enhance_photo_system.py --input path/to/photo.jpg
    python enhance_photo_system.py --input path/to/photo.jpg --output results/my_photo_enhanced.png
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from config import CHECKPOINT_BEST, IMAGE_SIZE
from models.model3_final import EnhancementCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Input a photo and save the enhanced output image."
    )
    parser.add_argument("--input", required=True, help="Path to the input photo.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output image path. Default: results/pipeline/<input_name>_enhanced.png",
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_BEST, help="Model checkpoint path.")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Inference image size.")
    return parser.parse_args()


def resolve_output_path(input_path: str, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    stem = Path(input_path).stem
    return Path("results") / "pipeline" / f"{stem}_enhanced.png"


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
        raise FileNotFoundError(f"Input photo not found: {args.input}")

    output_path = resolve_output_path(args.input, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    transform = T.Compose([T.Resize((args.size, args.size), antialias=True), T.ToTensor()])
    to_pil = T.ToPILImage()

    image = Image.open(args.input).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = model(tensor).squeeze(0).cpu().clamp(0, 1)

    to_pil(enhanced).save(output_path)
    print(f"Input  -> {args.input}")
    print(f"Output -> {output_path}")


if __name__ == "__main__":
    main()
