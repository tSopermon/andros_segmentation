# Andros Segmentation Project

## Overview
This repository provides a reproducible pipeline for multiclass image segmentation using PyTorch. It supports multiple architectures (DeepLabV3, UNet, UNet++, DeepLabV3+) and includes tools for training, evaluation, metrics, and visualization.

For a detailed explanation of the data loading and training process, see [DATA_AND_TRAINING.md](DATA_AND_TRAINING.md).

## Features
- **Modular Codebase:** Organized into config, data, models, training, evaluation, and utils.
- **Supported Models:**
    - Standard (SMP-backed): DeepLabV3, UNet, UNet++, DeepLabV3+
    - Original Implementations: `UNet_original`, `DeepLabV1_original`, `DeepLabV2_original`, `DeepLabV3_original`
- **Backbone:** Configurable via `BACKBONE` (default: `resnet101`).
- **Data Loading:** Efficient loading with caching and optimized workers.
- **Augmentation:** Comprehensive pipeline (Geometric, Pixel-level, Noise) using Albumentations.
- **Training:** Mixed precision (AMP), global reproducibility, early stopping, and checkpointing.
- **Transfer Learning:** Fine-tune from existing checkpoints with automatic shape mismatch handling (useful for varying class counts) and encoder freezing via `TRANSFER_LEARNING` config.
- **Self-Supervised Learning:** Built-in Masked Autoencoder (MAE) style pre-training for U-Net architectures to improve performance on small datasets without requiring labeled ground truth.
- **Self-Training (Pseudo-Labeling)**: Leverages unlabeled data using a Teacher-Student framework. Confidence-based filtering and dual-stream loaders allow the model to learn from unannotated satellite imagery using reliable automated predictions. 
- **Evaluation:** Automated generation of metrics, confusion matrices, and visualizations.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support recommended

### Local Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation (Recommended)
1. **Configure Environment:**
    ```bash
    cp .env.example .env
    # Edit .env to set DATASET_DIR=/absolute/path/to/dataset
    ```
2. **Run Training:**
    ```bash
    docker-compose up --build andros-segmentation
    ```
3. **Run Evaluation/Mask Generation:**
    ```bash
    docker-compose run --rm evaluate
    docker-compose run --rm generate-masks
    ```

## Usage

All scripts support a `--config` argument to specify the configuration file (default: `config/config.yaml`).

### Training
```bash
python train.py --config config/config.yaml
```
- **K-Fold Cross-Validation:** Set `K_FOLDS > 1` in config.
- **Ensembling:** Set `ENSEMBLE: true` in config.
- **Model Selection:** Controlled via `MODEL_SET` in config (`standard`, `originals`, `all`).
- **Standard Model Subset:** Optionally set `STANDARD_MODELS` (list) to run only selected standard models, e.g. `['DeepLabV3', 'UNet']`.
- **Transfer Learning:** Settings in `config.yaml`:
    - `TRANSFER_LEARNING: true`
    - `PRETRAINED_CHECKPOINT_DIR: 'checkpoints/'`
    - `PRETRAINED_WEIGHT_SUFFIX: '_pretrained.pth'` (or `_best.pth`)
    - `FREEZE_ENCODER: true/false`

### Self-Supervised Pre-training
```bash
python pretrain.py --config config/config.yaml
```
- Trains the model to reconstruct patch-masked inputs, building rich feature representations without labeled data.
- Generates both random masks and object-centric masks (using Sobel edge-density) to force high-level structural learning.
- Controlled via `PRETRAIN_EPOCHS`, `MASK_RATIO`, `PATCH_SIZE`, and `OBJECT_CENTRIC_EPOCH`.
- Seamlessly integrates with downstream training using `TRANSFER_LEARNING: true` and `PRETRAINED_WEIGHT_SUFFIX: '_pretrained.pth'`.
- See [SELF_SUPERVISED_LEARNING.md](SELF_SUPERVISED_LEARNING.md) for dedicated SSL instructions.

### Self-Training (Semi-Supervised)
```yaml
SELF_TRAINING: true
UNLABELED_IMG_PATH: "/path/to/unlabeled"
```
- Simultaneously learns from labeled data and confident Teacher-generated pseudo-labels.
- Automatically handles dual-stream batching (50% labeled, 50% unlabeled).
- Configurable `PSEUDO_LABEL_THRESHOLD` and `IGNORE_INDEX`.
- See [SELF_TRAINING.md](SELF_TRAINING.md) for technical details and implementation logic.

### Evaluation
```bash
python evaluate.py --config config/config.yaml
```
- Generates metrics and plots in `outputs/`.

### Mask Generation
```bash
python generate_masks.py --config config/config.yaml
```
- Generates segmentation masks for test images in `outputs/<ModelName>/masks/`.

### Training History Visualization
```bash
python evaluation/visualize_history.py
```
- Visualizes loss and metric trends from `outputs/training_history.npy`.

### Augmentation Visualization
```bash
python evaluation/visualize_augmentation.py
```
- Generates a grid of original vs. augmented image-mask pairs in `outputs/debug/augmentation_validation.png` to assess augmentation methods.

## Project Structure
- `config/`: Configuration files (YAML).
- `utils/`: Data loading and transformations.
- `models/`: Model architectures.
- `training/`: Training logic and loss functions.
- `evaluation/`: Metrics and visualization tools.
- `train.py`, `evaluate.py`, `generate_masks.py`: Entry points.

## Dataset Structure
```text
Andros Dataset/
    train/
        image/
        mask/
    val/     (Optional, used if PRE_SPLIT_DATASET: true)
        image/
        mask/
    test/
        image/
        mask/
```
Update paths in `config/config.yaml`. Set `PRE_SPLIT_DATASET: true` if your dataset already includes a `val/` partition (this will force `K_FOLDS=1`).

## Testing
Run unit tests:
```bash
pytest tests --maxfail=3 -v -rw
```

## License
[License Information]