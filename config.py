"""
config.py — Central configuration for all paths, hyperparameters, and settings.
Edit this file instead of touching individual scripts.
"""

import os

# ─────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

EUVP_TRAIN_A = os.path.join(DATA_ROOT, "EUVP", "Paired", "underwater_scenes", "trainA")
EUVP_TRAIN_B = os.path.join(DATA_ROOT, "EUVP", "Paired", "underwater_scenes", "trainB")
EUVP_VAL     = os.path.join(DATA_ROOT, "EUVP", "Paired", "underwater_scenes", "validation")

UIEB_RAW       = os.path.join(DATA_ROOT, "UIEB", "raw-890")
UIEB_REF       = os.path.join(DATA_ROOT, "UIEB", "reference-890")
UIEB_CHALLENGE = os.path.join(DATA_ROOT, "UIEB", "challenging-60")

RUIE_ROOT = os.path.join(DATA_ROOT, "RUIE")

LSUI_INPUT = os.path.join(DATA_ROOT, "LSUI", "input")
LSUI_GT    = os.path.join(DATA_ROOT, "LSUI", "GT")

# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
CHECKPOINTS_DIR  = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
LOGS_DIR         = os.path.join(PROJECT_ROOT, "logs")

CHECKPOINT_FINAL = os.path.join(CHECKPOINTS_DIR, "model_final.pth")
CHECKPOINT_BEST  = os.path.join(CHECKPOINTS_DIR, "model_best.pth")

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS     = 100
LR         = 0.0005
SAVE_EVERY = 10

# Loss weights (L1 + Edge + Color)
W_L1    = 0.6
W_EDGE  = 0.3
W_COLOR = 0.1

# MobileViT bottleneck
MOBILEVIT_DIM   = 64
MOBILEVIT_DEPTH = 2
MOBILEVIT_HEADS = 2
