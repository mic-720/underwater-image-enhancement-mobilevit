# Underwater Image Enhancement

PyTorch project for underwater image enhancement using a lightweight CNN encoder-decoder with a MobileViT-style bottleneck. The repository supports training on paired EUVP data, running inference on new images, and evaluating results on EUVP, UIEB, and RUIE.

## Features

- Single entry point through `main.py`
- GPU-aware training and inference with PyTorch
- EUVP paired training pipeline
- Batch, single-image, and validation inference modes
- Evaluation scripts for EUVP, UIEB, and RUIE
- Periodic checkpoints, CSV logs, and training curves
- Local Jupyter notebook for end-to-end training

## Project Structure

```text
clean_repo/
├── main.py
├── config.py
├── requirements.txt
├── Underwater_Enhancement_Training_local.ipynb
├── datasets/
│   └── euvp_dataset.py
├── evaluation/
│   ├── eval_euvp.py
│   ├── eval_ruie.py
│   └── eval_uieb.py
├── models/
│   └── cnn_model.py
├── scripts/
│   └── inference.py
├── training/
│   └── train.py
├── checkpoints/
├── logs/
└── results/
```

## Environment Setup

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux or macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
python main.py check-env
```

For NVIDIA GPU support, install a CUDA-enabled PyTorch build. Example for CUDA 11.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Dataset Layout

Expected structure under `data/`:

```text
data/
├── EUVP/
│   └── Paired/
│       └── underwater_scenes/
│           ├── trainA/
│           ├── trainB/
│           └── validation/
├── UIEB/
│   ├── raw-890/
│   ├── reference-890/
│   └── challenging-60/
└── RUIE/
    ├── UCCS/
    ├── UIQS/
    └── UTTS/
```

You can adjust paths in `config.py` if your local layout differs.

## Training

Start training with default settings:

```bash
python main.py train
```

Example with custom hyperparameters:

```bash
python main.py train --epochs 100 --batch 8 --lr 0.0005 --workers 2
```

Resume from a checkpoint:

```bash
python main.py train --resume checkpoints/model_final.pth
```

Training saves:

- `checkpoints/model_best.pth`
- `checkpoints/model_final.pth`
- `checkpoints/model_epoch_<N>.pth`
- `logs/train_log.csv`
- `results/training_curves.png`

## Inference

Run batch inference on a few images:

```bash
python main.py infer --mode batch --limit 8
```

Run on a single image:

```bash
python main.py infer --mode single --input path/to/image.jpg
```

Run on the validation folder:

```bash
python main.py infer --mode validation
```

Batch inference outputs are saved to `results/batch/`.

## Evaluation

Evaluate on EUVP:

```bash
python main.py eval --dataset euvp
```

Evaluate on UIEB and save enhanced outputs:

```bash
python main.py eval --dataset uieb --save-images
```

Evaluate on RUIE and save enhanced outputs:

```bash
python main.py eval --dataset ruie --save-images
```

Notes:

- `UIEB` evaluation requires `data/UIEB/`
- `RUIE` evaluation requires `data/RUIE/`
- `EUVP` evaluation requires `data/EUVP/Paired/underwater_scenes/validation/`

## Model Summary

- Encoder-decoder CNN backbone
- MobileViT-style transformer bottleneck
- Residual output formulation
- Image size controlled through `config.py`

## Notebook

Use `Underwater_Enhancement_Training_local.ipynb` for local training in VS Code or Jupyter. Make sure the notebook kernel points to this repo's virtual environment.

## Outputs

Common generated outputs:

- `checkpoints/` for trained weights
- `logs/` for training logs
- `results/` for inference images, evaluation reports, and plots

## License

Add your preferred license here before publishing the repository.
