"""
scripts/generate_model_block_diagrams.py

Generate separate block-diagram figures for:
  - models/model1.py
  - models/model2.py
  - models/model3_final.py

Outputs:
  - results/model_block_diagrams/model1_block_diagram.png
  - results/model_block_diagrams/model2_block_diagram.png
  - results/model_block_diagrams/model3_block_diagram.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "model_block_diagrams"
WIDTH = 1800
HEIGHT = 900
BACKGROUND = "#f6fbff"
OUTLINE = "#17324d"
TEXT = "#16314a"
FLOW = "#21435f"


@dataclass(frozen=True)
class Box:
    title: str
    lines: tuple[str, ...]
    rect: tuple[int, int, int, int]
    fill: str


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered_text(draw: ImageDraw.ImageDraw, box, text, font, fill) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    left, top, right, bottom = box.rect
    draw.rounded_rectangle(box.rect, radius=18, fill=box.fill, outline=OUTLINE, width=3)
    title_font = get_font(19, bold=True)
    body_font = get_font(15)
    centered_text(draw, (left + 10, top + 8, right - 10, top + 44), box.title, title_font, TEXT)
    y = top + 54
    for line in box.lines:
        centered_text(draw, (left + 12, y, right - 12, y + 22), line, body_font, TEXT)
        y += 22


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color=FLOW, width=5) -> None:
    draw.line((start, end), fill=color, width=width)
    x2, y2 = end
    direction = 1 if end[0] >= start[0] else -1
    draw.polygon(
        [(x2, y2), (x2 - 16 * direction, y2 - 8), (x2 - 16 * direction, y2 + 8)],
        fill=color,
    )


def draw_polyline(draw: ImageDraw.ImageDraw, points: Iterable[tuple[int, int]], color: str, width: int = 4) -> None:
    points = list(points)
    draw.line(points, fill=color, width=width)
    x2, y2 = points[-1]
    prev_x, prev_y = points[-2]
    direction = 1 if x2 >= prev_x else -1
    draw.polygon(
        [(x2, y2), (x2 - 14 * direction, y2 - 7), (x2 - 14 * direction, y2 + 7)],
        fill=color,
    )


def render_diagram(title: str, subtitle: str, boxes: list[Box], arrows: list[tuple[tuple[int, int], tuple[int, int]]], extras=None, out_name: str = "diagram.png") -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    centered_text(draw, (0, 24, WIDTH, 64), title, get_font(34, bold=True), TEXT)
    centered_text(draw, (0, 68, WIDTH, 100), subtitle, get_font(18), "#35516a")

    for box in boxes:
        draw_box(draw, box)
    for start, end in arrows:
        draw_arrow(draw, start, end)
    if extras:
        extras(draw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUT_DIR / out_name)


def model1_diagram():
    boxes = [
        Box("Input", ("3 x 256 x 256",), (60, 320, 220, 420), "#dfeffc"),
        Box("Encoder", ("Conv 3->32", "ReLU", "MaxPool", "Conv 32->64", "ReLU", "MaxPool"), (300, 250, 560, 490), "#9ed8e6"),
        Box("Decoder", ("Upsample", "Conv 64->64", "ReLU", "Upsample", "Conv 64->32", "ReLU", "Conv 32->3", "Sigmoid"), (660, 210, 980, 530), "#ffd58f"),
        Box("Output", ("3 x 256 x 256",), (1080, 320, 1240, 420), "#d9f0d2"),
    ]
    arrows = [((220, 370), (300, 370)), ((560, 370), (660, 370)), ((980, 370), (1080, 370))]
    render_diagram(
        "Model 1 Block Diagram",
        "Simple encoder-decoder CNN without bottleneck or skip connections",
        boxes,
        arrows,
        out_name="model1_block_diagram.png",
    )


def model2_diagram():
    boxes = [
        Box("Input", ("3 x 256 x 256",), (60, 320, 220, 420), "#dfeffc"),
        Box("Encoder", ("Conv 3->32", "ReLU", "MaxPool", "Conv 32->64", "ReLU", "MaxPool"), (280, 250, 540, 490), "#9ed8e6"),
        Box("Bottleneck", ("Conv 64->128", "ReLU", "Conv 128->128", "ReLU"), (620, 290, 880, 450), "#f4b7c7"),
        Box("Decoder", ("Upsample", "Conv 128->64", "ReLU", "Upsample", "Conv 64->32", "ReLU", "Conv 32->3", "Sigmoid"), (960, 210, 1280, 530), "#ffd58f"),
        Box("Output", ("3 x 256 x 256",), (1360, 320, 1520, 420), "#d9f0d2"),
    ]
    arrows = [((220, 370), (280, 370)), ((540, 370), (620, 370)), ((880, 370), (960, 370)), ((1280, 370), (1360, 370))]
    render_diagram(
        "Model 2 Block Diagram",
        "Encoder-bottleneck-decoder CNN with convolutional bottleneck",
        boxes,
        arrows,
        out_name="model2_block_diagram.png",
    )


def model3_diagram():
    boxes = [
        Box("Input / Residual", ("3 x 256 x 256", "Saved for final add"), (40, 330, 230, 430), "#dfeffc"),
        Box("enc1", ("Double Conv 3->32", "BatchNorm + ReLU", "Feature e1"), (280, 170, 540, 350), "#9ed8e6"),
        Box("enc2", ("MaxPool", "Double Conv 32->64", "BatchNorm + ReLU", "Feature e2"), (620, 170, 900, 370), "#9ed8e6"),
        Box("MobileViT Bottleneck", ("MaxPool", "Local Conv", "Transformer x depth", "Proj Conv + Residual"), (980, 180, 1300, 380), "#f7d58c"),
        Box("dec1", ("Upsample", "Concat with e2", "Double Conv 128->64"), (1380, 160, 1680, 360), "#f2c2cf"),
        Box("dec2", ("Upsample", "Concat with e1", "Double Conv 96->32"), (1380, 470, 1680, 670), "#f2c2cf"),
        Box("Final Conv", ("1 x 1 Conv 32->3", "Add residual", "Sigmoid"), (980, 520, 1260, 680), "#d9f0d2"),
    ]
    arrows = [
        ((230, 380), (280, 260)),
        ((540, 260), (620, 260)),
        ((900, 280), (980, 280)),
        ((1300, 280), (1380, 260)),
        ((1530, 360), (1530, 470)),
        ((1380, 570), (1260, 600)),
    ]

    def extras(draw: ImageDraw.ImageDraw) -> None:
        skip_color = "#c85c5c"
        res_color = "#7a3db8"
        draw_polyline(draw, [(540, 300), (560, 300), (560, 620), (1380, 620)], skip_color)
        draw.text((760, 630), "Skip 1: e1 -> dec2 concat", font=get_font(15, bold=True), fill=skip_color)
        draw_polyline(draw, [(900, 330), (930, 330), (930, 250), (1380, 250)], skip_color)
        draw.text((1030, 220), "Skip 2: e2 -> dec1 concat", font=get_font(15, bold=True), fill=skip_color)
        draw_polyline(draw, [(230, 380), (230, 720), (1080, 720), (1080, 680)], res_color)
        draw.text((610, 735), "Residual: input added before sigmoid", font=get_font(15, bold=True), fill=res_color)

    render_diagram(
        "Model 3 Block Diagram",
        "CNN encoder-decoder with MobileViT-style bottleneck, skip connections, and residual output",
        boxes,
        arrows,
        extras=extras,
        out_name="model3_block_diagram.png",
    )


def main():
    model1_diagram()
    model2_diagram()
    model3_diagram()
    print(f"Saved diagrams to {OUT_DIR}")


if __name__ == "__main__":
    main()
