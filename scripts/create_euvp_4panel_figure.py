from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "euvp_4panel.png"

ITEMS = [
    {
        "path": PROJECT_ROOT / "results" / "validation" / "im_u113__comparison.png",
        "label": "EUVP - red coral",
    },
    {
        "path": PROJECT_ROOT / "results" / "validation" / "im_u881__comparison.png",
        "label": "EUVP - squid",
    },
    {
        "path": PROJECT_ROOT / "results" / "validation" / "im_f2354__comparison.png",
        "label": "EUVP - crabs",
    },
    {
        "path": PROJECT_ROOT / "results" / "validation" / "im_f1010__comparison.png",
        "label": "EUVP - coral fish",
    },
]


def get_font(size: int):
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    cell_width = 420
    cell_height = 220
    cols = 2
    rows = 2
    margin = 28
    label_height = 30

    canvas_w = cols * cell_width + (cols + 1) * margin
    canvas_h = rows * (cell_height + label_height) + (rows + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = get_font(20)

    for idx, item in enumerate(ITEMS):
        image = Image.open(item["path"]).convert("RGB")
        image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)

        row, col = divmod(idx, cols)
        x = margin + col * (cell_width + margin)
        y = margin + row * (cell_height + label_height + margin)

        canvas.paste(image, (x, y))
        draw.text((x, y + cell_height + 4), item["label"], fill=(25, 25, 25), font=font)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
