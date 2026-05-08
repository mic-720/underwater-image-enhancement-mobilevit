"""
Create a publication-ready composite figure from saved comparison panels.

Edit EXAMPLE_ITEMS below or pass a JSON manifest with:
[
  {"path": "results/single/100_img__comparison.png", "label": "UIEB - fish scene"},
  ...
]

Usage:
    python scripts/make_publication_figure.py
    python scripts/make_publication_figure.py --manifest results/figures/figure_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "figures" / "publication_figure.png"

EXAMPLE_ITEMS = [
    {"path": "results/single/100_img__comparison.png", "label": "UIEB - fish scene"},
    {"path": "results/single/10139_comparison.png", "label": "UIEB - reef scene"},
    {"path": "results/single/10151_comparison.png", "label": "UIEB - diver scene"},
    {"path": "results/validation/im_u113__comparison.png", "label": "EUVP - red coral"},
    {"path": "results/validation/im_u881__comparison.png", "label": "EUVP - squid"},
    {"path": "results/validation/im_f2354__comparison.png", "label": "EUVP - crabs"},
    {"path": "results/single/blue_00_comparison.png", "label": "RUIE UCCS - blue cast"},
    {"path": "results/single/A_00380_comparison.png", "label": "RUIE UIQS - quality sample"},
    {"path": "results/single/0316_comparison.png", "label": "RUIE UTTS - failure case"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stitched figure from comparison images.")
    parser.add_argument("--manifest", type=str, default=None, help="Optional JSON manifest of {path, label} items.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--cols", type=int, default=2, help="Number of columns.")
    parser.add_argument("--cell-width", type=int, default=768, help="Width for each panel cell.")
    parser.add_argument("--cell-height", type=int, default=256, help="Height for each panel cell.")
    parser.add_argument("--margin", type=int, default=28, help="Outer and inner margin.")
    parser.add_argument("--label-height", type=int, default=28, help="Reserved height for the caption.")
    return parser.parse_args()


def load_items(manifest: str | None) -> list[dict[str, str]]:
    if manifest is None:
        return EXAMPLE_ITEMS

    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    items = load_items(args.manifest)
    if not items:
        raise ValueError("No figure items were provided.")

    cols = max(1, args.cols)
    rows = (len(items) + cols - 1) // cols
    canvas_w = cols * args.cell_width + (cols + 1) * args.margin
    canvas_h = rows * (args.cell_height + args.label_height) + (rows + 1) * args.margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = get_font(18)

    for idx, item in enumerate(items):
        path = resolve_path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Missing image: {path}")

        image = Image.open(path).convert("RGB").resize((args.cell_width, args.cell_height), Image.Resampling.LANCZOS)
        row, col = divmod(idx, cols)
        x = args.margin + col * (args.cell_width + args.margin)
        y = args.margin + row * (args.cell_height + args.label_height + args.margin)

        canvas.paste(image, (x, y))
        draw.text((x, y + args.cell_height + 4), item["label"], fill=(30, 30, 30), font=font)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved stitched figure -> {output_path}")


if __name__ == "__main__":
    main()
