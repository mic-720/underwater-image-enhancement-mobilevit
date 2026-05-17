# enhance_single_image.py
"""
Script to enhance a single underwater image using the trained EnhancementCNN model.
It loads the model checkpoint (model_best.pth), processes the specified image, and saves a side‑by‑side
comparison (input | enhanced) with textual labels.

Usage (run from the repository root)::
    python enhance_single_image.py
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

# -----------------------------
# Configuration
# -----------------------------
# Absolute path to the input image (modify if needed)
INPUT_IMAGE_PATH = r"C:\Users\ASUS\Downloads\Underwater-Image-Enhancement-clean (1)\clean_repo\data\EUVP\Paired\underwater_scenes\validation\im_f1010_.jpg"

# Path to the model checkpoint (relative to repository root)
MODEL_CHECKPOINT_PATH = os.path.join("checkpoints", "model_best.pth")

# Output directory for the result images
OUTPUT_DIR = "resultsSingle"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Model setup
# -----------------------------
# Import the model definition (relative import based on repository structure)
from models.cnn_model import EnhancementCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancementCNN().to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT_PATH, map_location=device))
model.eval()

# -----------------------------
# Image preprocessing / post‑processing utilities
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

# Font handling – fall back to default if arial is unavailable
try:
    font_small = ImageFont.truetype("arial.ttf", 18)
    font_big = ImageFont.truetype("arial.ttf", 22)
except Exception:
    font_small = ImageFont.load_default()
    font_big = ImageFont.load_default()


def add_label(img: Image.Image, text: str) -> Image.Image:
    """Add a small label at the top‑left corner of *img*.
    The label colour is white for visibility on most images.
    """
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255), font=font_small)
    return img


def add_main_label(img: Image.Image, text: str) -> Image.Image:
    """Add a larger label centred near the bottom of the combined image."""
    draw = ImageDraw.Draw(img)
    draw.text((10, img.height - 30), text, fill=(255, 255, 255), font=font_big)
    return img

# -----------------------------
# Processing pipeline
# -----------------------------
print(f"Processing image: {INPUT_IMAGE_PATH}")

# Load and prepare the input image
original_img = Image.open(INPUT_IMAGE_PATH).convert("RGB")
input_tensor = transform(original_img).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Convert model output back to PIL image
output_tensor = output.squeeze(0).cpu().clamp(0, 1)
enhanced_img = to_pil(output_tensor)

# Resize original to match the model input size (256x256)
input_resized = original_img.resize((256, 256))

# Add per‑image labels
input_resized = add_label(input_resized, "Input")
enhanced_img = add_label(enhanced_img, "Enhanced")

# Create side‑by‑side canvas (width = 512, height = 256)
comparison = Image.new("RGB", (512, 256))
comparison.paste(input_resized, (0, 0))
comparison.paste(enhanced_img, (256, 0))

# Add a main caption (you can customise the text)
comparison = add_main_label(comparison, "Validation Sample")

# Save individual and combined results
base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
input_resized.save(os.path.join(OUTPUT_DIR, f"{base_name}_input.png"))
enhanced_img.save(os.path.join(OUTPUT_DIR, f"{base_name}_enhanced.png"))
comparison.save(os.path.join(OUTPUT_DIR, f"{base_name}_comparison.png"))

print(f"Saved results to '{OUTPUT_DIR}' folder.")
