from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "results" / "figures" / "euvp_3panel.png"

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
    cols = 3
    margin = 24
    label_height = 30

    canvas_w = cols * cell_width + (cols + 1) * margin
    canvas_h = cell_height + label_height + 2 * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = get_font(19)

    for idx, item in enumerate(ITEMS):
        image = Image.open(item["path"]).convert("RGB")
        image = image.resize((cell_width, cell_height), Image.Resampling.LANCZOS)

        x = margin + idx * (cell_width + margin)
        y = margin
        canvas.paste(image, (x, y))
        draw.text((x, y + cell_height + 4), item["label"], fill=(25, 25, 25), font=font)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
