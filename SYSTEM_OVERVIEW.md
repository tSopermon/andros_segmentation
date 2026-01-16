# Andros Segmentation System Overview

## Introduction
The Andros Segmentation project is a modular, reproducible pipeline for multiclass image segmentation, designed for research and production workflows. It leverages PyTorch for deep learning, with a focus on extensibility, reproducibility, and best practices in model development and evaluation.

---

## Project Structure
- **config/**: YAML configuration files for hyperparameters, paths, and model settings.
- **utils/**: Data loading (`dataset.py`), augmentations (`transforms.py`), and configuration parsing (`config_loader.py`).
- **models/**: Model architectures in `model_zoo.py`:
  - Standard SMP-backed models: DeepLabV3, UNet, UNet++, DeepLabV3+
  - Original-paper implementations: `UNet_original`, `DeepLabV1_original`, `DeepLabV2_original`, `DeepLabV3_original`
  - Model selection controlled via `MODEL_SET` config option (`standard`/`originals`/`all`)
- **training/**: Training logic (`train_utils.py`), metrics computation (`metrics.py`).
- **evaluation/**: Evaluation, visualization, and metrics export utilities.
- **outputs/**: Auto-generated directory for results, metrics, and training history.
- **checkpoints/**: Stores model checkpoints for each architecture.
- **tests/**: Unit tests for all major modules and reproducibility.
- **Entry Points**: `train.py`, `evaluate.py`, `generate_masks.py`, `evaluation/visualize_history.py` for main workflows.

---

## Key Workflows
### Training

  - **Mixed Precision (automatic if GPU available):**
    - Reduces memory usage and speeds up training by leveraging GPU tensor cores, with negligible loss in accuracy. This enables larger batch sizes and faster experimentation.
  - **Global Seed for Reproducibility:**
    - Ensures that experiments are repeatable, which is critical for scientific research and fair model comparison.
  - **Early Stopping and Checkpointing:**
    - Prevents overfitting by halting training when validation performance plateaus, and ensures best models are saved for later evaluation or deployment.
  - **Model Selection via Config:**
    - Enables modular experimentation and hyperparameter tuning without code changes, supporting reproducible and scalable research workflows.
    - **K-fold & Ensemble support:**
      - The training script now supports stratified `K_FOLDS` (config: `K_FOLDS`, default `1`). When `K_FOLDS > 1` the code computes a primary per-image label to perform stratified splits, recomputes per-fold class weights, and saves fold-specific checkpoints as `checkpoints/{ModelName}_fold{fold}_best.pth`.
      - To ensemble fold models for final evaluation set `ENSEMBLE: true` in the config file (or via env `ENSEMBLE=true`). The ensemble averages logits across fold models for final predictions.
      - After K-fold training the script now performs an optional final step: it selects the best-performing fold (by validation mIoU), initializes a model from that fold's checkpoint (if available), and retrains on the entire training set to produce a single final checkpoint saved as `checkpoints/{ModelName}_best.pth`. This retrain uses the same hyperparameters (and any env overrides) from the config and is intended to consolidate k-fold tuning into one deployable model.
      - Use `ENCODER_WEIGHTS` config (e.g., `imagenet`) or set `ENCODER_WEIGHTS: null` to disable pretrained backbone downloads for smoke tests.

## Full Training Procedure

### 1. Dataset Loading
- **Config-Driven Paths:** Dataset locations are specified in the YAML config file, ensuring reproducibility and flexibility.
- **Custom Dataset Class:** The `utils/dataset.py` module defines a PyTorch `Dataset` class that:
  - Loads images and masks from the specified directory structure.
  - Supports multiclass segmentation with proper label encoding.
  - Applies data augmentations and preprocessing using `utils/transforms.py`.
- **Data Augmentation:**
  - Random flips, rotations, scaling, and color jitter are applied to increase data diversity and improve generalization.
- **Efficient DataLoader:**
  - PyTorch `DataLoader` is used with multi-worker loading and pinned memory for fast batch preparation.

### 2. Model Initialization
- **Model Selection:** Architecture and parameters are chosen via the config file and instantiated using `models/model_zoo.py`.
- **Unified Model Selection via `MODEL_SET`:** The pipeline supports both standard SMP-backed models and original-paper implementations. Selection is controlled via the `MODEL_SET` config option:
  - `MODEL_SET: standard` (default) — use standard models (DeepLabV3, UNet, UNet++, DeepLabV3+)
  - `MODEL_SET: originals` — use only original-paper implementations (`UNet_original`, `DeepLabV1_original`, `DeepLabV2_original`, `DeepLabV3_original`)
  - `MODEL_SET: all` — train/evaluate both standard and original implementations
- **Original-Paper Implementations:**
  - `UNet_original` — native PyTorch implementation of the original U-Net architecture
  - `DeepLabV1_original` — DeepLabV1 with Large Field-of-View (LargeFOV) and bilinear upsampling
  - `DeepLabV2_original` — DeepLabV2 with Dilated ResNet backbone, ASPP, and bilinear upsampling
  - `DeepLabV3_original` — DeepLabV3 via `torch.hub` (deeplabv3_resnet50) with output wrapper for consistent interface
- **Legacy Flags:** Individual per-model flags (`USE_UNET_ORIGINAL`, `USE_DEEPLABV1_ORIGINAL`, etc.) are still supported for backward compatibility when `MODEL_SET: standard`.
- **Weight Initialization:** Standard PyTorch initializations are used, with pretrained backbones if specified.

### 3. Training Loop
- **Mixed Precision Training:**
  - Enabled automatically if a GPU is available, using `torch.cuda.amp` for faster computation and reduced memory usage.
- **Optimizer & Scheduler:**
  - The current implementation uses the `Adam` optimizer by default together with an `ExponentialLR` scheduler. Other optimizers or learning-rate schedulers mentioned in prior drafts are not exposed via the configuration by default.
- **Loss Function:**
  - Controlled via `LOSS_FUNCTION` in config. Options: `CrossEntropy` (default), `Dice`, `DiceBCE` (Dice + CrossEntropy).
  - The implementation handles these choices in `train.py` and `training/losses.py`.
  - When using k-fold, class weights are recomputed per-fold and passed to the loss function (if applicable) to account for class imbalance per split.
**Gradient Clipping & Accumulation:**
  - Gradient clipping and gradient accumulation are not implemented in the current training loop. They can be added if required for large-batch or stability use cases.
- **Early Stopping:**
  - Monitors validation loss/IoU and halts training if no improvement is seen for a set number of epochs.
- **Checkpointing:**
  - Best model weights are saved automatically based on validation performance.
- **Logging:**
  - Training/validation metrics and losses are logged and saved to `outputs/` for analysis.

### 4. Validation
- **On-the-fly Evaluation:**
  - After each epoch, the model is evaluated on the validation set using the same metrics as final evaluation (IoU, F1, etc.).
- **Metric Tracking:**
  - All metrics are tracked per epoch for later visualization and analysis.

### 5. Optimization Techniques
- **Mixed Precision:**
  - Reduces memory and speeds up training when a GPU is available (implemented via `torch.amp`).
- **Data Augmentation:**
  - Increases effective dataset size and robustness.
- **Learning Rate Scheduling:**
  - The implementation uses `ExponentialLR` to decay the learning rate; other scheduling strategies are not currently exposed via configuration.
- **Early Stopping:**
  - Prevents overfitting and saves compute resources.

Note: gradient clipping/accumulation and multi-GPU training (DataParallel/DDP) are not implemented in the current codebase.


### Evaluation

- Run: `python evaluate.py --config config/config.yaml`
- **Design Choices & Reasoning:**
  - **Metrics (Precision, Recall, F1, IoU):**
    - These metrics provide a comprehensive view of segmentation performance, especially for imbalanced classes. IoU is the standard for segmentation, while Precision/Recall/F1 offer insight into class-specific errors.
  - **Results saved in `outputs/`:**
    - Ensures all evaluation outputs are organized and accessible for analysis, comparison, and reporting.
  - Plots, CSVs and visualizations use human-readable class names by default and saved overlays include filenames and a legend mapping colors to class names.

 - **Automated Plots & Diagnostics:**
   - Per-class and per-model metric plots (Precision, Recall, F1, IoU)
   - Confusion matrix heatmaps for each model
   - Per-image metric distribution boxplots
   - Class frequency vs. metric scatter plots (robust to remapped mask values)
   - Metric correlation matrices
   - All plots saved in `outputs/` for easy review
   - Saved PNG masks and overlays use a fixed project palette so colors are consistent between saved masks and visualizations.
   - Current class-color mapping (index → name → RGB / HEX):
     - 0: Water — (0, 0, 255) — #0000FF
     - 1: Woodland — (60, 16, 152) — #3C1098
     - 2: Arable land — (132, 41, 246) — #8429F6
     - 3: Frygana — (0, 255, 0) — #00FF00
     - 4: Other — (155, 155, 155) — #9B9B9B
     - 5: Artificial land — (226, 169, 41) — #E2A929
     - 6: Perm. Cult — (255, 255, 0) — #FFFF00
     - 7: Bareland — (255, 255, 255) — #FFFFFF
   - Robust handling of mask value remapping: class frequency and metrics are computed correctly even if mask values are relabeled to 0,1,...,n-1
   - If no valid points are available for a plot, the script will skip plotting those points
   - Extensive debug and validation utilities were used during development to ensure correct mask value handling and class frequency computation
   - Unit tests for plotting functions (see `tests/test_plots.py`)

### Training History Visualization
- **Run:** `python evaluation/visualize_history.py`
- **Purpose:** Visualize training metrics (Loss, mIoU, F1) over epochs to monitor convergence and compare models.
- **Outputs:**
  - `outputs/model_comparison_metrics.png`: Comparison of all models.
  - `outputs/history_<ModelName>.png`: Individual metric plots for each model.

### Mask Generation

- Run: `python generate_masks.py`
 - New config options (in `config/config.yaml`):
   - `GENERATE_FOR_ALL_SETS` (`false` by default) — when `true`, run inference over `train`, `val`, and `test` subsets instead of only `test`.
   - `INCLUDE_LOWRES` (`false` by default) — when `true`, also include images from the dataset `lowres/` folder.
  - When enabled, outputs are organized into subset subfolders under `outputs/<ModelName>/masks/` (for example `outputs/DeepLabV3/masks/train/`, `.../masks/test/`, `.../masks/lowres/`). Filenames retain the original image basename with `_mask.png` appended (for example `0000_mask.png`).
- **Design Choices & Reasoning:**
  - **Batch Mask Generation for All Models:**
    - Facilitates direct visual and quantitative comparison between models. Enables downstream analysis and qualitative assessment of segmentation quality.
  - **Output Organization:**
    - Masks are saved in model-specific directories, supporting traceability and reproducibility of results.
  - Saved masks are written using the project palette above so each class index maps to a stable RGB color.


### Testing
- Run: `pytest tests --maxfail=3 -v -rw`
- Tests cover datasets, metrics, transforms, model zoo, training utils, mask generation, CLI scripts, plotting, mixed precision, and reproducibility.
- The test suite is comprehensive, warning-free, and fully passing as of January 2026.

---

## Rationale for System Design Choices

### Training
- **Mixed Precision:** Chosen for its ability to accelerate training and reduce memory footprint, which is especially beneficial for large segmentation models and high-resolution images.
- **Reproducibility:** Essential for scientific rigor and fair benchmarking; all random seeds are set globally.
- **Early Stopping/Checkpointing:** Prevents overfitting and ensures the best model is always available for evaluation.
- **Config-Driven Workflows:** Decouples code from experiment settings, making the system flexible and user-friendly.

### Metrics
- **IoU (Intersection over Union):** The gold standard for segmentation, directly measures overlap between predicted and ground truth masks.
- **Precision/Recall/F1:** Address class imbalance and provide a nuanced view of model performance, especially for rare classes.

### Evaluation
- **Automated, Scripted Evaluation:** Guarantees consistency across experiments and simplifies batch analysis.
- **Output Logging:** All metrics and results are saved for traceability, comparison, and publication.

### Mask Generation
- **Automated Mask Export:** Enables visual inspection and further quantitative analysis (e.g., post-processing, error analysis).
- **Model-wise Organization:** Ensures clarity when comparing outputs from different architectures or training runs.

---

---

## Configuration & Conventions
- All settings are managed via YAML files in `config/`.
- Dataset directory structure:
  ```
  Andros Dataset/
      train/
          image/
          mask/
      test/
          image/
          mask/
  ```
- Use `utils/config_loader.py` for config parsing.
- Model instantiation is handled in `models/model_zoo.py`.
- Outputs (metrics, history) are always written to `outputs/`.
- Checkpoints are saved in `checkpoints/`.

### Logging
- The system uses a centralized logging configuration via `utils/logging_config.py` to ensure consistent formatting and handler setup across modules.
- Runtime verbosity is configurable with the `LOGGING_LEVEL` key in `config/config.yaml` (for example `LOGGING_LEVEL: INFO`). Entry scripts call `configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))` during startup.

---

## Dependencies
- torch
- numpy
- opencv-python
- albumentations
- segmentation-models-pytorch
- scikit-learn
- matplotlib
- tqdm
- pytest
- seaborn

Note: a YAML parser (PyYAML) is required for config parsing; install with `pip install pyyaml` if not already available. The `requirements.txt` now includes `torchvision`; `encoding` is optional (used only to enable `SyncBatchNorm` if installed). Loading `DeepLabV3` via `torch.hub` may download pretrained weights and requires network access. The precise runtime requirements are listed in `requirements.txt`.

---


## Best Practices
- Modular code structure for easy extension.
- Mixed precision and reproducibility enforced and tested by default.
- Early stopping, checkpointing, and experiment tracking built-in.
- All major modules and workflows are unit tested, including error handling and edge cases.

---

## Extending the System
- **Add a new model:**
  1. Implement in `models/model_zoo.py`.
  2. Update config YAML with new model parameters.
  3. Add unit tests in `tests/test_model_zoo.py`.
- **Generate masks for all models:**
  - Run `python generate_masks.py`.
  - Masks saved in `outputs/<ModelName>/masks/`.

---

## References
- See the official documentation for PyTorch, YAML, and pytest for further details.
- For unclear conventions, review the README and config files.
