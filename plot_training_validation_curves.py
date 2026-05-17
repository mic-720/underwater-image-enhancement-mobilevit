# plot_training_validation_curves.py
"""
Utility script to plot training and validation loss curves from log files.
Expected files:
- training_loss.txt : each line = loss value per epoch (or iteration)
- validation_loss.txt : each line = validation loss per epoch
The script reads both files, aligns epochs, and saves a combined plot.
"""
import os
import matplotlib.pyplot as plt

# Adjust these paths if needed
LOG_DIR = "resultsTraining"  # folder where loss logs are stored
TRAIN_LOG = os.path.join(LOG_DIR, "training_loss.txt")
VAL_LOG = os.path.join(LOG_DIR, "validation_loss.txt")
OUTPUT_PLOT = os.path.join(LOG_DIR, "loss_curve.png")

def read_losses(path):
    if not os.path.isfile(path):
        print(f"[WARN] File not found: {path}")
        return []
    with open(path, "r") as f:
        lines = f.readlines()
    # Convert to float, ignore empty lines / comments
    losses = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            losses.append(float(line))
        except ValueError:
            continue
    return losses

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    train_losses = read_losses(TRAIN_LOG)
    val_losses = read_losses(VAL_LOG)
    if not train_losses:
        print("[ERROR] No training loss data found.")
        return
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o")
    if val_losses:
        val_epochs = list(range(1, len(val_losses) + 1))
        plt.plot(val_epochs, val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss Curve")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300)
    plt.close()
    print(f"[INFO] Loss curve saved to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
