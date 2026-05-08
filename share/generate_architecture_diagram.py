"""
Generate a simple architecture diagram for the enhancement network.

Outputs:
    results/figures/architecture_diagram.svg
    results/figures/architecture_diagram.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "architecture_diagram.svg"
OUTPUT_PNG_PATH = PROJECT_ROOT / "results" / "figures" / "architecture_diagram.png"


def block(x: int, y: int, w: int, h: int, fill: str, title: str, body: list[str]) -> str:
    lines = [
        f'<rect x="{x}" y="{y}" rx="18" ry="18" width="{w}" height="{h}" fill="{fill}" stroke="#17324d" stroke-width="2"/>',
        f'<text x="{x + w/2}" y="{y + 34}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#102030">{title}</text>',
    ]
    for i, text in enumerate(body):
        lines.append(
            f'<text x="{x + w/2}" y="{y + 72 + i*24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#1e2f3f">{text}</text>'
        )
    return "\n".join(lines)


def arrow(x1: int, y1: int, x2: int, y2: int, label: str = "") -> str:
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#17324d" stroke-width="4" marker-end="url(#arrowhead)"/>'
    ]
    if label:
        parts.append(
            f'<text x="{(x1 + x2) / 2}" y="{y1 - 12}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#35516a">{label}</text>'
        )
    return "\n".join(parts)


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    x = left + (right - left - (bbox[2] - bbox[0])) / 2
    y = top + (bottom - top - (bbox[3] - bbox[1])) / 2
    draw.text((x, y), text, font=font, fill=fill)


def export_png() -> None:
    width, height = 1380, 500
    image = Image.new("RGB", (width, height), "#eef8fb")
    draw = ImageDraw.Draw(image)

    title_font = get_font(30, bold=True)
    block_title_font = get_font(22, bold=True)
    body_font = get_font(16, bold=False)
    note_font = get_font(18, bold=True)
    subnote_font = get_font(15, bold=False)

    draw_centered(draw, (0, 18, width, 72), "Underwater Enhancement Network", title_font, "#0f2740")

    blocks = [
        (70, 140, 370, 320, "#9ed8e6", "Encoder", ["Conv block x 2", "Downsampling", "Feature extraction"]),
        (540, 110, 840, 350, "#ffd58f", "MobileViT Bottleneck", ["Local convolutions", "Lightweight self-attention", "Global context fusion", "Compact latent features"]),
        (1010, 140, 1310, 320, "#b8e3b0", "Decoder", ["Upsampling", "Reconstruction layers", "Enhanced RGB output"]),
    ]
    for left, top, right, bottom, fill, title, lines in blocks:
        draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=fill, outline="#17324d", width=2)
        draw_centered(draw, (left, top + 8, right, top + 54), title, block_title_font, "#102030")
        for idx, line in enumerate(lines):
            y = top + 72 + idx * 24
            draw_centered(draw, (left, y, right, y + 18), line, body_font, "#1e2f3f")

    draw.line((370, 230, 540, 230), fill="#17324d", width=4)
    draw.polygon([(540, 230), (528, 224), (528, 236)], fill="#17324d")
    draw_centered(draw, (390, 190, 520, 215), "Encoded feature maps", body_font, "#35516a")

    draw.line((840, 230, 1010, 230), fill="#17324d", width=4)
    draw.polygon([(1010, 230), (998, 224), (998, 236)], fill="#17324d")
    draw_centered(draw, (860, 190, 990, 215), "Context-enriched features", body_font, "#35516a")

    draw.rounded_rectangle((65, 380, 1310, 450), radius=14, fill="#ffffff", outline="#bfd3df", width=2)
    draw_centered(draw, (95, 390, 1280, 418), "Flow: Input underwater image -> Encoder -> MobileViT bottleneck -> Decoder -> Enhanced image", note_font, "#16314a")
    draw_centered(draw, (95, 418, 1280, 442), "PNG and SVG versions are both saved for easy reuse in reports and slides.", subnote_font, "#35516a")

    image.save(OUTPUT_PNG_PATH)


def main() -> None:
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1380" height="500" viewBox="0 0 1380 500">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#e8f6fb"/>
      <stop offset="100%" stop-color="#f9fcfd"/>
    </linearGradient>
    <marker id="arrowhead" markerWidth="12" markerHeight="9" refX="10" refY="4.5" orient="auto">
      <polygon points="0 0, 12 4.5, 0 9" fill="#17324d"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1380" height="500" fill="url(#bg)"/>
  <text x="690" y="58" text-anchor="middle" font-family="Arial, sans-serif" font-size="30" font-weight="700" fill="#0f2740">
    Underwater Enhancement Network
  </text>

  {block(70, 140, 300, 180, "#9ed8e6", "Encoder", ["Conv block x 2", "Downsampling", "Feature extraction"])}
  {block(540, 110, 300, 240, "#ffd58f", "MobileViT Bottleneck", ["Local convolutions", "Lightweight self-attention", "Global context fusion", "Compact latent features"])}
  {block(1010, 140, 300, 180, "#b8e3b0", "Decoder", ["Upsampling", "Reconstruction layers", "Enhanced RGB output"])}

  {arrow(370, 230, 540, 230, "Encoded feature maps")}
  {arrow(840, 230, 1010, 230, "Context-enriched features")}

  <rect x="65" y="380" rx="14" ry="14" width="1245" height="70" fill="#ffffff" stroke="#bfd3df" stroke-width="1.5"/>
  <text x="690" y="410" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#16314a">
    Flow: Input underwater image -> Encoder -> MobileViT bottleneck -> Decoder -> Enhanced image
  </text>
  <text x="690" y="435" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#35516a">
    Use this SVG directly in slides, reports, or convert it to PNG in any browser if needed.
  </text>
</svg>
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(svg, encoding="utf-8")
    export_png()
    print(f"Saved architecture diagram -> {OUTPUT_PATH}")
    print(f"Saved architecture diagram -> {OUTPUT_PNG_PATH}")


if __name__ == "__main__":
    main()
