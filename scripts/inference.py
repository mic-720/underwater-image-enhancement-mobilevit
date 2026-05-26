"""
scripts/inference.py

Unified inference — replaces inference.py, inference3.py,
inferenceSingle.py, and inferenceValidation.py.

Modes:
    batch       : Enhance N images from a folder, save 2-panel comparisons
    single      : Enhance one image; optionally show 3-panel with ground truth
    validation  : Enhance all images in EUVP validation folder

Usage (via main.py):
    python main.py infer --mode batch
    python main.py infer --mode batch      --input data/EUVP/.../trainA --limit 20
    python main.py infer --mode single     --input path/to/image.jpg
    python main.py infer --mode single     --input path/to/image.jpg --target path/to/gt.jpg
    python main.py infer --mode validation
    python main.py infer --mode batch      --checkpoint checkpoints/model_best.pth

Direct usage:
    python scripts/inference.py --mode single --input path/to/image.jpg
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from config import (
    EUVP_TRAIN_A, EUVP_VAL,
    CHECKPOINT_BEST, RESULTS_DIR, IMAGE_SIZE,
)
from models.model3_final import EnhancementCNN

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device: torch.device) -> EnhancementCNN:
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run  python main.py train  first, or pass --checkpoint <path>."
        )
    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def enhance(model, img_path: str, device: torch.device, size: int):
    """Return (input_pil, enhanced_pil) both resized to (size, size)."""
    tfm    = T.Compose([T.Resize((size, size), antialias=True), T.ToTensor()])
    to_pil = T.ToPILImage()

    original = Image.open(img_path).convert("RGB")
    inp_t    = tfm(original).unsqueeze(0).to(device)

    with torch.no_grad():
        out_t = model(inp_t)

    return to_pil(inp_t.squeeze(0).cpu()), to_pil(out_t.squeeze(0).clamp(0, 1).cpu())


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Overlay a white label with dark shadow on the top-left corner."""
    img  = img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((11, 6),  text, fill=(0, 0, 0), font=font)    # shadow
    draw.text((10, 5),  text, fill=(255, 255, 255), font=font)
    return img


def make_comparison(*panels) -> Image.Image:
    """Stitch N labelled panels side by side into one image."""
    w, h  = panels[0].size
    comp  = Image.new("RGB", (w * len(panels), h))
    for i, p in enumerate(panels):
        comp.paste(p, (i * w, 0))
    return comp


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(args, model, device):
    input_dir = args.input or EUVP_TRAIN_A
    out_dir   = os.path.join(RESULTS_DIR, "batch")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in sorted(os.listdir(input_dir))
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if args.limit:
        files = files[:args.limit]

    print(f"Enhancing {len(files)} images from {input_dir} ...")

    for fname in files:
        path     = os.path.join(input_dir, fname)
        inp, enh = enhance(model, path, device, args.size)
        stem     = os.path.splitext(fname)[0]

        comp = make_comparison(
            add_label(inp, "Input"),
            add_label(enh, "Enhanced"),
        )
        comp.save(os.path.join(out_dir, f"{stem}_comparison.png"))
        enh.save(os.path.join(out_dir,  f"{stem}_enhanced.png"))
        print(f"  Saved {stem}")

    print(f"\nDone. Results in {out_dir}")


def run_single(args, model, device):
    if not args.input or not os.path.isfile(args.input):
        raise ValueError("--mode single requires --input <path/to/image.jpg>")

    out_dir = os.path.join(RESULTS_DIR, "single")
    os.makedirs(out_dir, exist_ok=True)

    inp, enh = enhance(model, args.input, device, args.size)
    stem     = os.path.splitext(os.path.basename(args.input))[0]

    if args.target and os.path.isfile(args.target):
        gt  = T.ToPILImage()(
            T.Compose([T.Resize((args.size, args.size), antialias=True), T.ToTensor()])(
                Image.open(args.target).convert("RGB")
            )
        )
        comp = make_comparison(
            add_label(inp, "Input"),
            add_label(gt,  "Ground Truth"),
            add_label(enh, "Enhanced"),
        )
    else:
        comp = make_comparison(
            add_label(inp, "Input"),
            add_label(enh, "Enhanced"),
        )

    save_path = os.path.join(out_dir, f"{stem}_comparison.png")
    comp.save(save_path)
    print(f"Saved -> {save_path}")


def run_validation(args, model, device):
    val_dir = args.input or EUVP_VAL
    out_dir = os.path.join(RESULTS_DIR, "validation")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in sorted(os.listdir(val_dir))
             if os.path.splitext(f)[1].lower() in IMG_EXTS]

    print(f"Running validation inference on {len(files)} images ...")

    for fname in files:
        path     = os.path.join(val_dir, fname)
        inp, enh = enhance(model, path, device, args.size)
        stem     = os.path.splitext(fname)[0]

        comp = make_comparison(
            add_label(inp, "Input"),
            add_label(enh, "Enhanced"),
        )
        comp.save(os.path.join(out_dir, f"{stem}_comparison.png"))
        enh.save(os.path.join(out_dir,  f"{stem}_enhanced.png"))

    print(f"Saved {len(files)} comparisons -> {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Underwater image enhancement — inference")
    p.add_argument("--mode",       choices=["batch", "single", "validation"],
                   default="batch")
    p.add_argument("--input",      type=str, default=None,
                   help="Image file (single) or folder (batch/validation)")
    p.add_argument("--target",     type=str, default=None,
                   help="(single) Ground truth image for 3-panel comparison")
    p.add_argument("--checkpoint", type=str, default=CHECKPOINT_BEST)
    p.add_argument("--size",       type=int, default=IMAGE_SIZE)
    p.add_argument("--limit",      type=int, default=8,
                   help="(batch) Max images to process. 0 = all")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Mode       : {args.mode}")

    model = load_model(args.checkpoint, device)

    if args.mode == "batch":
        run_batch(args, model, device)
    elif args.mode == "single":
        run_single(args, model, device)
    elif args.mode == "validation":
        run_validation(args, model, device)


if __name__ == "__main__":
    main()
