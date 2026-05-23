from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "cross_domain_4panel.png"

ITEMS = [
    {
        "path": PROJECT_ROOT / "results" / "single" / "100_img__comparison.png",
        "label": "UIEB - fish scene",
        "crop_mode": "input_enhanced_from_triplet",
    },
    {
        "path": PROJECT_ROOT / "results" / "single" / "10139_comparison.png",
        "label": "UIEB - reef scene",
        "crop_mode": "input_enhanced_from_triplet",
    },
    {
        "path": PROJECT_ROOT / "results" / "single" / "blue_00_comparison.png",
        "label": "RUIE UCCS - blue cast",
        "crop_mode": "keep",
    },
    {
        "path": PROJECT_ROOT / "results" / "single" / "0316_comparison.png",
        "label": "RUIE UTTS - sea urchin scene",
        "crop_mode": "keep",
    },
]


def get_font(size: int):
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def prepare_panel(path: Path, crop_mode: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size

    if crop_mode == "input_enhanced_from_triplet":
        third = w // 3
        left = image.crop((0, 0, third, h))
        right = image.crop((w - third, 0, w, h))
        panel = Image.new("RGB", (third * 2, h), (255, 255, 255))
        panel.paste(left, (0, 0))
        panel.paste(right, (third, 0))
        return panel

    return image


def main():
    panels = [prepare_panel(item["path"], item["crop_mode"]) for item in ITEMS]

    cell_width = 520
    cell_height = 260
    cols = 2
    rows = 2
    margin = 28
    label_height = 32

    canvas_w = cols * cell_width + (cols + 1) * margin
    canvas_h = rows * (cell_height + label_height) + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = get_font(20)

    for idx, (item, panel) in enumerate(zip(ITEMS, panels)):
        panel = panel.resize((cell_width, cell_height), Image.Resampling.LANCZOS)
        row, col = divmod(idx, cols)
        x = margin + col * (cell_width + margin)
        y = margin + row * (cell_height + label_height + margin)
        canvas.paste(panel, (x, y))
        draw.text((x, y + cell_height + 4), item["label"], fill=(25, 25, 25), font=font)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
