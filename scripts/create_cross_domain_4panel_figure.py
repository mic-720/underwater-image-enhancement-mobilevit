import argparse
import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_BEST, IMAGE_SIZE
from models.model3_final import EnhancementCNN


OUTPUT_PATH = PROJECT_ROOT / "results" / "cross_domain_4x2_grid.png"
PANEL_DIR = PROJECT_ROOT / "results" / "cross_domain_panels"

ITEMS = [
    {
        "source": PROJECT_ROOT / "data" / "UIEB" / "raw-890" / "100_img_.png",
        "label": "UIEB - fish scene",
        "panel_name": "uieb_fish_scene_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "UIEB" / "raw-890" / "10139.png",
        "label": "UIEB - reef scene",
        "panel_name": "uieb_reef_scene_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "UIEB" / "raw-890" / "10151.png",
        "label": "UIEB - diver scene",
        "panel_name": "uieb_diver_scene_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "UIEB" / "raw-890" / "102_img_.png",
        "label": "UIEB - cave opening scene",
        "panel_name": "uieb_cave_opening_scene_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "RUIE" / "UCCS" / "blue" / "blue_00.jpg",
        "label": "RUIE UCCS - blue cast",
        "panel_name": "ruie_uccs_blue_cast_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "RUIE" / "UIQS" / "A" / "A_00380.jpg",
        "label": "RUIE UIQS - quality sample",
        "panel_name": "ruie_uiqs_quality_sample_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "RUIE" / "UTTS" / "pic_A" / "JPEGImages" / "0316.JPG",
        "label": "RUIE UTTS - sea urchin scene",
        "panel_name": "ruie_utts_sea_urchin_scene_comparison.png",
    },
    {
        "source": PROJECT_ROOT / "data" / "RUIE" / "UIQS" / "E" / "E_722.jpg",
        "label": "RUIE UIQS - low-light scene",
        "panel_name": "ruie_uiqs_low_light_scene_comparison.png",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a 4 x 2 cross-domain qualitative figure.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_BEST), help="Checkpoint path.")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="Output figure path.")
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Inference image size.")
    return parser.parse_args()


def get_font(size: int):
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_model(checkpoint: str, device: torch.device) -> EnhancementCNN:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def enhance_image(model: EnhancementCNN, image_path: Path, device: torch.device, size: int):
    tfm = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor()])
    to_pil = T.ToPILImage()

    original = Image.open(image_path).convert("RGB")
    inp_t = tfm(original).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = model(inp_t).squeeze(0).cpu().clamp(0, 1)

    input_pil = to_pil(inp_t.squeeze(0).cpu())
    enhanced_pil = to_pil(out_t)
    return input_pil, enhanced_pil


def add_overlay_label(img: Image.Image, text: str, font: ImageFont.ImageFont) -> Image.Image:
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    draw.text((11, 6), text, fill=(0, 0, 0), font=font)
    draw.text((10, 5), text, fill=(255, 255, 255), font=font)
    return canvas


def make_comparison(input_img: Image.Image, enhanced_img: Image.Image, font: ImageFont.ImageFont) -> Image.Image:
    left = add_overlay_label(input_img, "Input", font)
    right = add_overlay_label(enhanced_img, "Enhanced", font)
    width, height = left.size
    canvas = Image.new("RGB", (width * 2, height), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    return canvas


def build_panel(model: EnhancementCNN, item: dict, device: torch.device, size: int, font: ImageFont.ImageFont) -> Path:
    source = item["source"]
    if not source.is_file():
        raise FileNotFoundError(f"Missing source image: {source}")

    input_img, enhanced_img = enhance_image(model, source, device, size)
    panel = make_comparison(input_img, enhanced_img, font)

    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = PANEL_DIR / item["panel_name"]
    panel.save(panel_path)
    return panel_path


def stitch_figure(items, panel_paths, output_path: Path):
    cell_width = 520
    cell_height = 260
    cols = 2
    rows = 4
    margin = 28
    label_height = 32

    canvas_w = cols * cell_width + (cols + 1) * margin
    canvas_h = rows * (cell_height + label_height) + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = get_font(20)

    for idx, (item, panel_path) in enumerate(zip(items, panel_paths)):
        panel = Image.open(panel_path).convert("RGB")
        panel = panel.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        row, col = divmod(idx, cols)
        x = margin + col * (cell_width + margin)
        y = margin + row * (cell_height + label_height + margin)
        canvas.paste(panel, (x, y))
        draw.text((x, y + cell_height + 4), item["label"], fill=(25, 25, 25), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    model = load_model(args.checkpoint, device)
    panel_font = get_font(18)
    panel_paths = [build_panel(model, item, device, args.size, panel_font) for item in ITEMS]
    stitch_figure(ITEMS, panel_paths, output_path)

    print(f"Saved figure -> {output_path}")
    print(f"Saved panels -> {PANEL_DIR}")


if __name__ == "__main__":
    main()
