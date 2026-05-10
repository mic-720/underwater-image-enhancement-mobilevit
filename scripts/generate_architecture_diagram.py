"""
Generate an exact architecture diagram for ``models/cnn_model.py``.

Outputs:
    results/figures/architecture_diagram_detailed.svg
    results/figures/architecture_diagram_detailed.png
    results/figures/architecture_diagram.svg
    results/figures/architecture_diagram.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

DETAILED_SVG_PATH = FIGURES_DIR / "architecture_diagram_detailed.svg"
DETAILED_PNG_PATH = FIGURES_DIR / "architecture_diagram_detailed.png"
SVG_PATH = FIGURES_DIR / "architecture_diagram.svg"
PNG_PATH = FIGURES_DIR / "architecture_diagram.png"

WIDTH = 1880
HEIGHT = 980
BACKGROUND = "#eef7fb"
OUTLINE = "#17324d"
TEXT = "#183147"
SUBTEXT = "#35516a"
FLOW = "#21435f"
SKIP = "#c85c5c"
RESIDUAL = "#7a3db8"


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


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_box(draw: ImageDraw.ImageDraw, box: Box) -> None:
    left, top, right, bottom = box.rect
    draw.rounded_rectangle(box.rect, radius=18, fill=box.fill, outline=OUTLINE, width=2)
    title_font = get_font(15, bold=True)
    body_font = get_font(12)
    centered_text(draw, (left + 8, top + 8, right - 8, top + 42), box.title, title_font, TEXT)
    y = top + 46
    for line in box.lines:
        centered_text(draw, (left + 10, y, right - 10, y + 20), line, body_font, TEXT)
        y += 18


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str | None = None,
    color: str = FLOW,
) -> None:
    draw.line((start, end), fill=color, width=4)
    x2, y2 = end
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        direction = 1 if end[0] >= start[0] else -1
        draw.polygon(
            [(x2, y2), (x2 - 14 * direction, y2 - 7), (x2 - 14 * direction, y2 + 7)],
            fill=color,
        )
        if label:
            mid_x = (start[0] + end[0]) / 2
            centered_text(draw, (int(mid_x - 140), min(start[1], end[1]) - 28, int(mid_x + 140), min(start[1], end[1]) - 8), label, get_font(12), SUBTEXT)
    else:
        direction = 1 if end[1] >= start[1] else -1
        draw.polygon(
            [(x2, y2), (x2 - 7, y2 - 14 * direction), (x2 + 7, y2 - 14 * direction)],
            fill=color,
        )
        if label:
            mid_y = (start[1] + end[1]) / 2
            centered_text(draw, (x2 - 120, int(mid_y - 12), x2 + 120, int(mid_y + 12)), label, get_font(12), SUBTEXT)


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int,
    fill: str = TEXT,
    bold: bool = False,
    anchor: str = "middle",
) -> str:
    weight = "700" if bold else "400"
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escape_xml(text)}</text>'
    )


def svg_box(box: Box) -> str:
    left, top, right, bottom = box.rect
    center_x = (left + right) / 2
    parts = [
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" '
        f'rx="18" ry="18" fill="{box.fill}" stroke="{OUTLINE}" stroke-width="2"/>',
        svg_text(center_x, top + 28, box.title, 15, bold=True),
    ]
    y = top + 56
    for line in box.lines:
        parts.append(svg_text(center_x, y, line, 12))
        y += 18
    return "\n".join(parts)


def svg_arrow(
    start: tuple[int, int],
    end: tuple[int, int],
    label: str | None = None,
    color: str = FLOW,
) -> str:
    x1, y1 = start
    x2, y2 = end
    parts = [f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="4"/>']
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = f"{x2},{y2} {x2 - 14 * direction},{y2 - 7} {x2 - 14 * direction},{y2 + 7}"
        parts.append(f'<polygon points="{points}" fill="{color}"/>')
        if label:
            parts.append(svg_text((x1 + x2) / 2, min(y1, y2) - 12, label, 12, fill=SUBTEXT))
    else:
        direction = 1 if y2 >= y1 else -1
        points = f"{x2},{y2} {x2 - 7},{y2 - 14 * direction} {x2 + 7},{y2 - 14 * direction}"
        parts.append(f'<polygon points="{points}" fill="{color}"/>')
        if label:
            parts.append(svg_text(x2, (y1 + y2) / 2 + 4, label, 12, fill=SUBTEXT))
    return "\n".join(parts)


def svg_polyline(points: Iterable[tuple[int, int]], color: str, label: str | None = None, label_x: int | None = None, label_y: int | None = None) -> str:
    point_str = " ".join(f"{x},{y}" for x, y in points)
    parts = [f'<polyline points="{point_str}" fill="none" stroke="{color}" stroke-width="4"/>']
    x2, y2 = list(points)[-1]
    prev_x, prev_y = list(points)[-2]
    if abs(x2 - prev_x) >= abs(y2 - prev_y):
        direction = 1 if x2 >= prev_x else -1
        tip = f"{x2},{y2} {x2 - 14 * direction},{y2 - 7} {x2 - 14 * direction},{y2 + 7}"
    else:
        direction = 1 if y2 >= prev_y else -1
        tip = f"{x2},{y2} {x2 - 7},{y2 - 14 * direction} {x2 + 7},{y2 - 14 * direction}"
    parts.append(f'<polygon points="{tip}" fill="{color}"/>')
    if label and label_x is not None and label_y is not None:
        parts.append(svg_text(label_x, label_y, label, 13, fill=color))
    return "\n".join(parts)


def make_boxes() -> list[Box]:
    return [
        Box(
            "Input x / residual",
            (
                "Tensor: 3 x 256 x 256",
                "Saved as residual = x",
                "Feeds enc1 and final add",
            ),
            (40, 150, 230, 270),
            "#dfeffc",
        ),
        Box(
            "self.enc1",
            (
                "conv_block(3 -> 32)",
                "enc1[0]: Conv2d 3 -> 32, k=3, p=1",
                "enc1[1]: BatchNorm2d(32), ReLU",
                "enc1[3]: Conv2d 32 -> 32, k=3, p=1",
                "enc1[4]: BatchNorm2d(32), ReLU",
                "e1: 32 x 256 x 256",
            ),
            (290, 120, 580, 300),
            "#9ed8e6",
        ),
        Box(
            "self.pool",
            (
                "MaxPool2d",
                "kernel_size=2, stride=2",
                "pool(e1): 32 x 128 x 128",
            ),
            (640, 155, 800, 265),
            "#dcecf3",
        ),
        Box(
            "self.enc2",
            (
                "conv_block(32 -> 64)",
                "enc2[0]: Conv2d 32 -> 64, k=3, p=1",
                "enc2[1]: BatchNorm2d(64), ReLU",
                "enc2[3]: Conv2d 64 -> 64, k=3, p=1",
                "enc2[4]: BatchNorm2d(64), ReLU",
                "e2: 64 x 128 x 128",
            ),
            (860, 120, 1150, 300),
            "#8ed0dc",
        ),
        Box(
            "self.pool",
            (
                "MaxPool2d",
                "kernel_size=2, stride=2",
                "pool(e2): 64 x 64 x 64",
            ),
            (1210, 155, 1370, 265),
            "#dcecf3",
        ),
        Box(
            "self.bottleneck",
            (
                "MobileViTBlock(dim=64, depth=2, heads=2)",
                "local_conv: Conv2d 64 -> 64, k=3, p=1",
                "flatten: (B,64,64,64) -> (B,4096,64)",
                "transformer: TransformerEncoder x 2",
                "each layer: nhead=2, FFN 64->128->64",
                "drop=0.1, batch_first=T, norm_first=T",
                "proj_conv: Conv2d 64 -> 64, k=1",
                "internal residual: return x + identity",
                "b: 64 x 64 x 64",
            ),
            (1430, 90, 1830, 330),
            "#ffd58f",
        ),
        Box(
            "F.interpolate",
            (
                "d1 upsample",
                "scale_factor=2",
                'mode="bilinear"',
                "align_corners=False",
                "64 x 128 x 128",
            ),
            (1440, 410, 1620, 540),
            "#f7ead0",
        ),
        Box(
            "torch.cat + self.dec1",
            (
                "cat([d1, e2], dim=1)",
                "64 + 64 = 128 channels",
                "dec1 = conv_block(128 -> 64)",
                "Conv2d 128 -> 64, BN, ReLU",
                "Conv2d 64 -> 64, BN, ReLU",
                "d1: 64 x 128 x 128",
            ),
            (1080, 390, 1380, 560),
            "#b8e3b0",
        ),
        Box(
            "F.interpolate",
            (
                "d2 upsample",
                "scale_factor=2",
                'mode="bilinear"',
                "align_corners=False",
                "64 x 256 x 256",
            ),
            (970, 650, 1150, 780),
            "#f7ead0",
        ),
        Box(
            "torch.cat + self.dec2",
            (
                "cat([d2, e1], dim=1)",
                "64 + 32 = 96 channels",
                "dec2 = conv_block(96 -> 32)",
                "Conv2d 96 -> 32, BN, ReLU",
                "Conv2d 32 -> 32, BN, ReLU",
                "d2: 32 x 256 x 256",
            ),
            (610, 625, 910, 800),
            "#c9ebb8",
        ),
        Box(
            "self.final_conv",
            (
                "Conv2d 32 -> 3",
                "kernel_size=1",
                "out: 3 x 256 x 256",
            ),
            (370, 660, 540, 765),
            "#efe3f8",
        ),
        Box(
            "Residual Add",
            (
                "out + residual",
                "global output skip",
                "3 x 256 x 256",
            ),
            (240, 660, 340, 765),
            "#f7d7d7",
        ),
        Box(
            "torch.sigmoid",
            (
                "applied after add",
                "range [0, 1]",
                "3 x 256 x 256",
            ),
            (90, 660, 200, 765),
            "#e6eefc",
        ),
        Box(
            "Enhanced Output",
            (
                "network prediction",
                "3 x 256 x 256",
            ),
            (40, 820, 230, 915),
            "#d8f1de",
        ),
    ]


def draw_legend(draw: ImageDraw.ImageDraw) -> None:
    legend = (280, 835, 1830, 940)
    draw.rounded_rectangle(legend, radius=16, fill="#ffffff", outline="#bfd3df", width=2)
    centered_text(
        draw,
        (310, 848, 1800, 875),
        "Forward: x -> enc1 -> pool -> enc2 -> pool -> bottleneck -> upsample -> cat with e2 -> dec1 -> upsample -> cat with e1 -> dec2 -> final_conv -> out + residual -> sigmoid",
        get_font(15, bold=True),
        TEXT,
    )
    centered_text(
        draw,
        (310, 878, 1800, 902),
        "Red paths are U-Net-style skip connections. Purple path is the separate global residual used by EnhancementCNN.forward().",
        get_font(14),
        SUBTEXT,
    )
    centered_text(
        draw,
        (310, 904, 1800, 928),
        "Bottleneck also has its own internal residual: MobileViTBlock returns x + identity after proj_conv.",
        get_font(14),
        SUBTEXT,
    )


def draw_base(draw: ImageDraw.ImageDraw, boxes: list[Box]) -> None:
    centered_text(draw, (0, 20, WIDTH, 60), "Exact EnhancementCNN Architecture", get_font(30, bold=True), "#13304d")
    centered_text(
        draw,
        (0, 60, WIDTH, 88),
        "Source: models/cnn_model.py   |   IMAGE_SIZE=256   |   MOBILEVIT_DIM=64   |   MOBILEVIT_DEPTH=2   |   MOBILEVIT_HEADS=2   |   Params=321,859",
        get_font(15),
        SUBTEXT,
    )
    for box in boxes:
        draw_box(draw, box)

    draw_arrow(draw, (230, 210), (290, 210), "3 x 256 x 256")
    draw_arrow(draw, (580, 210), (640, 210), "32 x 256 x 256")
    draw_arrow(draw, (800, 210), (860, 210), "32 x 128 x 128")
    draw_arrow(draw, (1150, 210), (1210, 210), "64 x 128 x 128")
    draw_arrow(draw, (1370, 210), (1430, 210), "64 x 64 x 64")
    draw_arrow(draw, (1630, 330), (1530, 410), None)
    draw_arrow(draw, (1440, 475), (1380, 475), "64 x 128 x 128")
    draw_arrow(draw, (1230, 560), (1060, 650), None)
    draw_arrow(draw, (970, 715), (910, 715), "64 x 256 x 256")
    draw_arrow(draw, (610, 715), (540, 715), "32 x 256 x 256")
    draw_arrow(draw, (370, 715), (340, 715), "3 x 256 x 256")
    draw_arrow(draw, (240, 715), (200, 715), None)
    draw_arrow(draw, (145, 765), (145, 820), None)

    draw.line((1150, 290, 1530, 290), fill=SKIP, width=4)
    draw.line((1530, 290, 1530, 410), fill=SKIP, width=4)
    draw.polygon([(1530, 410), (1523, 396), (1537, 396)], fill=SKIP)
    centered_text(draw, (1180, 298, 1510, 322), "Skip: e2 -> cat([d1, e2], dim=1)", get_font(13), SKIP)

    draw.line((580, 285, 1060, 285), fill=SKIP, width=4)
    draw.line((1060, 285, 1060, 650), fill=SKIP, width=4)
    draw.polygon([(1060, 650), (1053, 636), (1067, 636)], fill=SKIP)
    centered_text(draw, (670, 293, 1035, 317), "Skip: e1 -> cat([d2, e1], dim=1)", get_font(13), SKIP)

    draw.line((80, 270, 80, 715), fill=RESIDUAL, width=4)
    draw.line((80, 715, 240, 715), fill=RESIDUAL, width=4)
    draw.polygon([(240, 715), (226, 708), (226, 722)], fill=RESIDUAL)
    centered_text(draw, (40, 610, 220, 635), "Global residual", get_font(13), RESIDUAL)
    centered_text(draw, (40, 632, 240, 655), "residual = x", get_font(13), RESIDUAL)

    draw_legend(draw)


def export_png() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw_base(draw, make_boxes())
    DETAILED_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(DETAILED_PNG_PATH)
    image.save(PNG_PATH)


def export_svg() -> None:
    boxes = make_boxes()
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BACKGROUND}"/>',
        svg_text(WIDTH / 2, 50, "Exact EnhancementCNN Architecture", 30, fill="#13304d", bold=True),
        svg_text(
            WIDTH / 2,
            80,
            "Source: models/cnn_model.py   |   IMAGE_SIZE=256   |   MOBILEVIT_DIM=64   |   MOBILEVIT_DEPTH=2   |   MOBILEVIT_HEADS=2   |   Params=321,859",
            15,
            fill=SUBTEXT,
        ),
    ]
    parts.extend(svg_box(box) for box in boxes)
    parts.extend(
        [
            svg_arrow((230, 210), (290, 210), "3 x 256 x 256"),
            svg_arrow((580, 210), (640, 210), "32 x 256 x 256"),
            svg_arrow((800, 210), (860, 210), "32 x 128 x 128"),
            svg_arrow((1150, 210), (1210, 210), "64 x 128 x 128"),
            svg_arrow((1370, 210), (1430, 210), "64 x 64 x 64"),
            svg_arrow((1630, 330), (1530, 410)),
            svg_arrow((1440, 475), (1380, 475), "64 x 128 x 128"),
            svg_arrow((1230, 560), (1060, 650)),
            svg_arrow((970, 715), (910, 715), "64 x 256 x 256"),
            svg_arrow((610, 715), (540, 715), "32 x 256 x 256"),
            svg_arrow((370, 715), (340, 715), "3 x 256 x 256"),
            svg_arrow((240, 715), (200, 715)),
            svg_arrow((145, 765), (145, 820)),
            svg_polyline([(1150, 290), (1530, 290), (1530, 410)], SKIP, "Skip: e2 -> cat([d1, e2], dim=1)", 1340, 317),
            svg_polyline([(580, 285), (1060, 285), (1060, 650)], SKIP, "Skip: e1 -> cat([d2, e1], dim=1)", 845, 312),
            svg_polyline([(80, 270), (80, 715), (240, 715)], RESIDUAL, "Global residual: residual = x", 150, 633),
            '<rect x="280" y="835" width="1550" height="105" rx="16" ry="16" fill="#ffffff" stroke="#bfd3df" stroke-width="2"/>',
            svg_text(
                1055,
                864,
                "Forward: x -> enc1 -> pool -> enc2 -> pool -> bottleneck -> upsample -> cat with e2 -> dec1 -> upsample -> cat with e1 -> dec2 -> final_conv -> out + residual -> sigmoid",
                15,
                bold=True,
            ),
            svg_text(
                1055,
                894,
                "Red paths are U-Net-style skip connections. Purple path is the separate global residual used by EnhancementCNN.forward().",
                14,
                fill=SUBTEXT,
            ),
            svg_text(
                1055,
                920,
                "Bottleneck also has its own internal residual: MobileViTBlock returns x + identity after proj_conv.",
                14,
                fill=SUBTEXT,
            ),
            "</svg>",
        ]
    )
    svg = "\n".join(parts)
    DETAILED_SVG_PATH.parent.mkdir(parents=True, exist_ok=True)
    DETAILED_SVG_PATH.write_text(svg, encoding="utf-8")
    SVG_PATH.write_text(svg, encoding="utf-8")


def main() -> None:
    export_png()
    export_svg()
    print(f"Saved detailed architecture diagram -> {DETAILED_PNG_PATH}")
    print(f"Saved detailed architecture diagram -> {DETAILED_SVG_PATH}")
    print(f"Saved architecture diagram -> {PNG_PATH}")
    print(f"Saved architecture diagram -> {SVG_PATH}")


if __name__ == "__main__":
    main()
