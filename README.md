# Andros Segmentation Project

## Overview
This repository provides a modular pipeline for multiclass image segmentation using PyTorch. It supports multiple architectures (DeepLabV3, UNet, UNet++, DeepLabV3+) and includes tools for training, evaluation, metrics, and visualization.

## Features
- Modular codebase: config, data, models, training, evaluation, utils, tests
- Supports DeepLabV3, UNet, UNet++, DeepLabV3+ (via segmentation-models-pytorch)
- Supports original-paper implementations:
    - `UNet_original` — native PyTorch implementation of the original U-Net architecture
    - `DeepLabV1_original` — DeepLabV1 with Large Field-of-View (LargeFOV)
    - `DeepLabV2_original` — DeepLabV2 with Dilated ResNet backbone and ASPP
    - `DeepLabV3_original` — DeepLabV3 via `torch.hub` (`deeplabv3_resnet50`)
- Model selection controlled via `MODEL_SET` config option:
    - `standard` (default): SMP-backed models (DeepLabV3, UNet, UNet++, DeepLabV3+)
    - `originals`: original-paper implementations only
    - `all`: train/evaluate both standard and original implementations
- Efficient data loading and augmentation
- Mixed precision training (automatic GPU acceleration with torch.amp)
- Full reproducibility: global random seeds for Python, NumPy, and PyTorch
- Training with early stopping and checkpointing
- Comprehensive metrics: Precision, Recall, F1, IoU
- Automated evaluation and visualization
- Unit tests for core modules and reproducibility/mixed precision

## Evaluation & Visualization
- Automated generation of:
   - Per-class and per-model metric plots (Precision, Recall, F1, IoU)
   - Confusion matrix heatmaps for each model
   - Per-image metric distribution boxplots
   - Class frequency vs. metric scatter plots (robust to remapped mask values)
   - Metric correlation matrices
   - All plots saved in `outputs/` for easy review
   - Plots, CSVs, visualizations and saved masks use human-readable class names (configurable in the evaluation entry point).
   - Saved overlay images include input/mask filenames in panel titles and a legend mapping colors → class names.
   - The system embeds a custom color palette for classes in saved PNG masks so colors are consistent between saved masks and overlay visualizations.

## Installation

### Option 1: Local Installation
1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   # Ensure YAML parser
   Ensure a YAML parser is available (PyYAML recommended):

   ```bash
   pip install pyyaml
   ```

Note: `requirements.txt` now includes `torchvision`. The `encoding` package is optional — it's used only to enable `SyncBatchNorm` if available. Loading `DeepLabV3` via `torch.hub` may download pretrained weights and requires network access.

### Option 2: Docker Installation (Recommended for Production)

Docker provides a containerized environment with all dependencies pre-configured, ensuring consistency across different systems.

#### Prerequisites
- Docker installed ([installation guide](https://docs.docker.com/get-docker/))
- NVIDIA Docker runtime for GPU support ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- At least 10GB free disk space for the Docker image

#### Quick Start with Docker

1. **Build the Docker image:**
   ```bash
   ./docker-run.sh build
   ```
   Or manually:
   ```bash
   docker build -t andros-segmentation:latest .
   ```

2. **Run training:**
   ```bash
   ./docker-run.sh train
   ```
   This mounts your dataset, config, checkpoints, and outputs directories automatically.

3. **Run evaluation:**
   ```bash
   ./docker-run.sh evaluate
   ```

4. **Generate masks:**
   ```bash
   ./docker-run.sh generate
   ```

5. **Interactive shell (for debugging):**
   ```bash
   ./docker-run.sh shell
   ```

6. **Run tests:**
   ```bash
   ./docker-run.sh test
   ```

#### Using Docker Compose

Alternatively, use docker-compose for more complex workflows:

```bash
# Train with GPU support
docker-compose up andros-segmentation

# Run evaluation (uses profile)
docker-compose --profile evaluate up evaluate

# Generate masks (uses profile)
docker-compose --profile generate up generate-masks
```

#### Docker Configuration

- **Dataset path:** Update the dataset path in [docker-compose.yml](docker-compose.yml#L22) or [docker-run.sh](docker-run.sh#L12) to match your local dataset location.
- **Config file:** The Docker setup uses [config/config.docker.yaml](config/config.docker.yaml) by default (with `DATASET_PATH: /data`), but you can edit the mounted [config/config.yaml](config/config.yaml) for custom runs.
- **GPU settings:** Control GPU visibility with `CUDA_VISIBLE_DEVICES` environment variable in docker-compose.yml.
- **CPU-only mode:** Use `./docker-run.sh cpu` or remove the `--gpus all` flag from docker run commands.

#### Volume Mounts
Docker containers mount the following directories:
- `/data` → Your dataset (read-only)
- `/app/config` → Configuration files (allows live editing)
- `/app/checkpoints` → Model checkpoints (persisted)
- `/app/outputs` → Evaluation results and masks (persisted)


## Usage
- **Training:**
   ```bash
   python train.py --config config/config.yaml
   ```
   - Mixed precision is enabled automatically on GPU (no extra flag needed).
   - Training is fully reproducible (global seeds set in train.py).
   - K-fold training: set `K_FOLDS` in `config/config.yaml` (or via env `K_FOLDS`) to run stratified k-fold cross-validation; set `ENSEMBLE: true` to ensemble fold checkpoints for final evaluation.
   - When `K_FOLDS > 1`, the training script performs per-fold training and saves fold-specific checkpoints as `checkpoints/<ModelName>_fold{n}_best.pth`. After completing all folds the script will select the best-performing fold (by validation mIoU), optionally initialize a model from that fold's weights, and retrain a single model on the full training set. The consolidated model is saved to `checkpoints/<ModelName>_best.pth` for deployment or mask generation.
   - Model selection is controlled via the `MODEL_SET` config option in `config/config.yaml`:
      - `MODEL_SET: standard` (default) — train/evaluate standard SMP-backed models (DeepLabV3, UNet, UNet++, DeepLabV3+)
      - `MODEL_SET: originals` — train/evaluate only original-paper implementations (`UNet_original`, `DeepLabV1_original`, `DeepLabV2_original`, `DeepLabV3_original`)
      - `MODEL_SET: all` — train/evaluate both standard and original implementations
   - Legacy per-model flags are still supported when `MODEL_SET: standard` for backward compatibility:
      - `USE_UNET_ORIGINAL: true` — include only `UNet_original` in standard set
      - Individual DeepLab flags (`USE_DEEPLABV1_ORIGINAL`, `USE_DEEPLABV2_ORIGINAL`, `USE_DEEPLABV3_ORIGINAL`) work similarly
   - The `--config` flag selects which YAML to use; model selection follows that YAML exclusively.
   - Encoder pretrained weights: control via `ENCODER_WEIGHTS` in `config/config.yaml` (e.g., `imagenet`) or via env `ENCODER_WEIGHTS` (set to `None` to disable downloads during smoke tests).
   - You can also override `BATCH_SIZE`, `NUM_WORKERS`, and `MAX_EPOCHS` via environment variables for quick smoke runs.
   - Optimizer / scheduler: the training code uses `Adam` by default with an `ExponentialLR` scheduler.
   - Loss Function: controlled via `LOSS_FUNCTION` in `config/config.yaml`. Options: `CrossEntropy` (default), `Dice`, `DiceBCE` (Dice + CrossEntropy).


- **Evaluation:**
   ```bash
   python evaluate.py --config config/config.yaml
   ```
   - Generates and saves all evaluation plots in `outputs/`.
   - Handles remapped mask values for robust class frequency analysis.
   - If no valid points are available for a plot, the script will skip plotting those points.

- **Training History Visualization:**
   To visualize training metrics (Loss, mIoU, F1) over epochs from `outputs/training_history.npy`:
   ```bash
   python evaluation/visualize_history.py
   ```
   - Generates:
     - `outputs/model_comparison_metrics.png`: Comparisons of all models for Validation Loss, mIoU, and F1.
     - `outputs/history_<ModelName>.png`: Individual training vs validation plots for each model.

- **Generate Segmentation Masks for Test Data:**
   This script runs all trained models on the test set and saves predicted masks in model-specific folders:
   ```bash
   python generate_masks.py
   ```
   - Output masks are saved in `outputs/<ModelName>/masks/` (e.g., `outputs/DeepLabV3/masks/`).
    - Each mask is named after the input image with `_mask.png` suffix by default.
    - New config options (in `config/config.yaml`):
       - `GENERATE_FOR_ALL_SETS` (`false` by default) — when `true`, run inference over `train`, `val`, and `test` subsets (not only `test`).
       - `INCLUDE_LOWRES` (`false` by default) — when `true`, include images from the dataset `lowres/` folder as well.
   - When `GENERATE_FOR_ALL_SETS` or `INCLUDE_LOWRES` are enabled the script organizes outputs into subset subfolders under `outputs/<ModelName>/masks/` (for example `outputs/DeepLabV3/masks/train/`, `.../masks/test/`, `.../masks/lowres/`). Filenames keep the original image basename with the `_mask.png` suffix (for example `0000_mask.png`).
   - Model selection follows the `MODEL_SET` config setting:
      - `standard`: processes DeepLabV3, DeepLabV3Plus, UNet, UNetPlusPlus
      - `originals`: processes UNet_original, DeepLabV1_original, DeepLabV2_original, DeepLabV3_original
      - `all`: processes all standard and original models
      - Saved PNG masks use a fixed, project-wide palette so each class maps to a specific RGB color. Current mapping (index → name → RGB / HEX):
         - 0: Water — (0, 0, 255) — #0000FF
         - 1: Woodland — (60, 16, 152) — #3C1098
         - 2: Arable land — (132, 41, 246) — #8429F6
         - 3: Frygana — (0, 255, 0) — #00FF00
         - 4: Other — (155, 155, 155) — #9B9B9B
         - 5: Artificial land — (226, 169, 41) — #E2A929
         - 6: Perm. Cult — (255, 255, 0) — #FFFF00
         - 7: Bareland — (255, 255, 255) — #FFFFFF


## Project Structure
- `config/` - Configuration files (YAML)
- `utils/` - Dataset and transforms
- `models/` - Model architectures
- `training/` - Training and metrics
- `evaluation/` - Evaluation, visualization, export
- `tests/` - Unit tests
- `train.py` - Main training script
- `evaluate.py` - Main evaluation script
- `generate_masks.py` - Script to generate and save segmentation masks for all test images and models
- `evaluation/visualize_history.py` - Script to visualize training history curves


## Debugging & Testing
- Extensive debug and validation utilities ensure correct mask value handling and class frequency computation.
- Comprehensive unit tests cover all major modules: dataset, transforms, model zoo, training utilities, metrics, mask generation, CLI scripts, and plotting functions.
- Mixed precision and reproducibility are enforced and tested by default.
- The test suite is warning-free and fully passing as of January 2026.

## Logging
- Centralized logging is provided by `utils/logging_config.py` and used by entry scripts (`train.py`, `evaluate.py`, `generate_masks.py`).
- Configure the runtime logging level using the `LOGGING_LEVEL` key in `config/config.yaml` (e.g. `INFO`, `DEBUG`). Entry scripts call `configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))` to initialize logging.
- This keeps log formatting and handlers consistent across the codebase and makes it easy to change verbosity from a single place.

## Notes
- The `outputs/` directory is automatically created for saving training history and results.
- Mixed precision and reproducibility are enforced by default in the training pipeline.

## Dataset
Organize your dataset as follows:
```
Andros Dataset/
    train/
        image/
        mask/
    test/
        image/
        mask/
```
Update paths in `config/config.yaml` as needed.



## Testing
Run all unit tests (including mask generation and CLI error handling):
```bash
pytest tests --maxfail=3 -v -rw
```
This verifies dataset, metrics, transforms, model zoo, training utilities, mask generation, CLI scripts, plotting, mixed precision, and reproducibility. The suite is robust to edge cases and system errors.

Note: The repository's unit tests passed successfully in the current environment after the recent training changes.

## Citation

---

## License

---