"""
main.py — Single entry point for the entire project.

Usage:
    python main.py train      [--epochs 100] [--batch 8] [--lr 0.0005] [--resume checkpoints/model_final.pth]
    python main.py infer      [--mode batch|single|validation] [--input <path>] [--checkpoint <ckpt>]
    python main.py eval       [--dataset all|euvp|uieb|ruie] [--checkpoint <ckpt>] [--save-images]
    python main.py check-env

Examples:
    python main.py train --epochs 100 --batch 8
    python main.py infer --mode single --input path/to/image.jpg
    python main.py eval  --dataset uieb --save-images
    python main.py check-env
"""

import argparse
import sys


def cmd_train(remaining):
    from training.train import main as train_main
    sys.argv = [sys.argv[0]] + remaining
    train_main()


def cmd_infer(remaining):
    from scripts.inference import main as infer_main
    sys.argv = [sys.argv[0]] + remaining
    infer_main()


def cmd_eval(remaining):
    dataset = "all"
    forwarded = []
    skip_next = False

    for i, arg in enumerate(remaining):
        if skip_next:
            skip_next = False
            continue
        if arg == "--dataset":
            if i + 1 < len(remaining):
                dataset = remaining[i + 1]
                skip_next = True
            continue
        forwarded.append(arg)

    sys.argv = [sys.argv[0]] + forwarded

    if dataset in ("all", "euvp"):
        from evaluation.eval_euvp import main as euvp_main
        euvp_main()
    if dataset in ("all", "uieb"):
        from evaluation.eval_uieb import main as uieb_main
        uieb_main()
    if dataset in ("all", "ruie"):
        from evaluation.eval_ruie import main as ruie_main
        ruie_main()


def cmd_check_env(_):
    import torch, torchvision, cv2, PIL, numpy, skimage, matplotlib
    print("-" * 42)
    print("Environment Check")
    print("-" * 42)
    print(f"Python        : {sys.version.split()[0]}")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Torchvision   : {torchvision.__version__}")
    print(f"OpenCV        : {cv2.__version__}")
    print(f"Pillow        : {PIL.__version__}")
    print(f"NumPy         : {numpy.__version__}")
    print(f"scikit-image  : {skimage.__version__}")
    print(f"Matplotlib    : {matplotlib.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print("-" * 42)
    print("All dependencies OK.")


COMMANDS = {
    "train":     cmd_train,
    "infer":     cmd_infer,
    "eval":      cmd_eval,
    "check-env": cmd_check_env,
}


def main():
    parser = argparse.ArgumentParser(
        description="Underwater Image Enhancement — main entry point",
        usage="python main.py <command> [options]",
    )
    parser.add_argument(
        "command",
        choices=list(COMMANDS.keys()),
        help="Sub-command to run",
    )
    args, remaining = parser.parse_known_args()
    COMMANDS[args.command](remaining)


if __name__ == "__main__":
    main()
