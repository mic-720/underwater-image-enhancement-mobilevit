"""
datasets/euvp_dataset.py

Paired and unpaired image dataset loader for EUVP (and compatible datasets).

Supports:
  - Paired mode  : returns (input_tensor, target_tensor)
  - Unpaired mode: returns input_tensor only (inference / no-ref eval)
  - Optional data augmentation (random flips + colour jitter on input only)
  - Graceful skip of unreadable images with a logged warning
"""

import os
import logging
import random
from typing import Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import IMAGE_SIZE

logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(directory: str) -> list:
    return sorted(
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


class EUVPDataset(Dataset):
    """
    Args:
        input_dir  : Path to degraded images (trainA / validation).
        target_dir : Path to clean ground truth (trainB). If None, returns
                     input tensors only (unpaired / inference mode).
        augment    : Apply random flips and colour jitter (training only).
        image_size : Resize spatial resolution (default from config).
    """

    def __init__(
        self,
        input_dir: str,
        target_dir: Optional[str] = None,
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
    ):
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if target_dir and not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target directory not found: {target_dir}")

        self.input_dir  = input_dir
        self.target_dir = target_dir
        self.augment    = augment
        self.image_size = image_size

        all_files = _list_images(input_dir)

        if target_dir:
            target_files  = set(_list_images(target_dir))
            self.filenames = [f for f in all_files if f in target_files]
            missing = len(all_files) - len(self.filenames)
            if missing:
                logger.warning(f"{missing} input image(s) have no matching target — skipped.")
        else:
            self.filenames = all_files

        if not self.filenames:
            raise RuntimeError(f"No valid images found in {input_dir}")

        self.resize    = T.Resize((image_size, image_size), antialias=True)
        self.to_tensor = T.ToTensor()
        self.jitter    = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def _load(self, path: str) -> Optional[Image.Image]:
        try:
            return self.resize(Image.open(path).convert("RGB"))
        except Exception as e:
            logger.warning(f"Could not load image {path}: {e}")
            return None

    def _augment_pair(
        self, inp: Image.Image, tar: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() > 0.5:
            inp, tar = TF.hflip(inp), TF.hflip(tar)
        if random.random() > 0.5:
            inp, tar = TF.vflip(inp), TF.vflip(tar)
        inp = self.jitter(inp)   # colour jitter on input only
        return inp, tar

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        fname   = self.filenames[idx]
        inp_img = self._load(os.path.join(self.input_dir, fname))

        zero = torch.zeros(3, self.image_size, self.image_size)

        if inp_img is None:
            return (zero, zero) if self.target_dir else zero

        if self.target_dir:
            tar_img = self._load(os.path.join(self.target_dir, fname))
            if tar_img is None:
                return self.to_tensor(inp_img), zero
            if self.augment:
                inp_img, tar_img = self._augment_pair(inp_img, tar_img)
            return self.to_tensor(inp_img), self.to_tensor(tar_img)

        return self.to_tensor(inp_img)
