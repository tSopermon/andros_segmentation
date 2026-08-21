# Andros Segmentation System Overview

## Introduction
The Andros Segmentation project is a reproducible pipeline for multiclass image segmentation, designed for research and production workflows. It leverages PyTorch for deep learning, focusing on reproducibility, and best practices.

## System Architecture

### Core Components
- **Configuration:** YAML-based configuration (`config/config.yaml`) decoupled from code.
- **Data Pipeline:** Custom `Dataset` class handling image loading, class mapping, and augmentations (Albumentations). Supports on-the-fly caching for efficiency and `DualStreamDataset` for concurrent labeled/unlabeled loading.
- **Model Zoo:** Lazy-loaded models including SMP-backed architectures (DeepLabV3, UNet, etc.) and native paper implementations.
- **Training Engine:** 
    - Loop with mixed precision (AMP) support.
    - Global random seeding for reproducibility.
    - Class Weight Clipping (`CLASS_WEIGHT_CLIP`) to stabilize extreme dataset imbalance.
    - Early stopping and checkpointing (best fold/best overall).
    - K-Fold Cross-Validation and Ensemble support.
    - **Teacher-Student Framework**: Active Student model learns from Ground Truth and frozen Teacher-generated Pseudo-Labels with confidence-based filtering (`IGNORE_INDEX`).
    - **Transfer Learning:** Supports fine-tuning from previously trained checkpoints with automatic shape mismatch handling and optional encoder freezing.
- **Evaluation:** Automated metric computation (IoU, F1, Precision, Recall) and visualization generation.
- **Explainability (XAI):** Post-hoc Grad-CAM class-activation visualization (`gradcam.py` + `evaluation/gradcam_utils.py`) producing per-class heatmaps and overlays for the test set, enabling qualitative interpretation of model behavior and laying the groundwork for quantitative XAI metrics.

### Workflows

#### 1. Training (`train.py`)
- **Initialization:** Loads config, sets seeds, prepares datasets.
- **Data Splitting/Validation:** Can accept datasets pre-split into train/val (`PRE_SPLIT_DATASET: true`) or perform automated Stratified K-Fold / random splits on a monolithic training folder.
- **K-Fold:** If enabled, splits data, trains per fold, and saves fold checkpoints.
- **Self-Training:** Optionally enables semi-supervised learning via `SELF_TRAINING`. Loads an existing frozen Teacher to generate labels for massive unannotated sets.
- **Ensembling:** Optionally ensembles fold models for evaluation.
- **Final Model:** Selects best fold configuration and retrains on the full training set for a production-ready model.

#### 2. Evaluation (`evaluate.py`)
- Loads the best checkpoint.
- Computes metrics on the test set.
- Generates plots: Confusion Matrices, Metric distributions, Predictions.

#### 3. Mask Generation (`generate_masks.py`)
- Runs inference on specified datasets.
- Exports color-coded segmentation masks for visual inspection.

#### 4. Inference / Prediction (`predict.py`)
- Provides a flexible CLI for on-demand inference on external images or directories.
- Uses **Patch-based Sliding Window Inference** with customizable overlapping (`--patch-size`, `--overlap`) to seamlessly process arbitrarily large images without OOM errors.
- Automatically detects class counts from checkpoints and outputs masked images with color overlays.

#### 5. Grad-CAM / XAI Visualization (`gradcam.py`)
- Generates Class Activation Maps for every evaluated model on the test set using the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library.
- **Method registry** (`evaluation/gradcam_utils.py::METHODS`) maps five CAM variants to their construction metadata: `GradCAM`, `SegEigenCAM`, `LayerCAM`, `EigenCAM`, and `HiResCAM`. `EigenCAM` is class-agnostic (one map per image); the remaining four are class-discriminative (one map per class present in the ground-truth mask).
- **Class targeting** reuses the segmentation-aware `SemanticSegmentationTarget(category, binary_mask)` from the library: for a class `c`, the binary GT mask `(gt == c)` gates the per-pixel class score, so the CAM localizes the evidence the model uses for that specific class.
- **Target-layer resolution** (`resolve_target_layers`) follows a three-tier strategy: an optional `GRADCAM_TARGET_LAYERS` config override → a per-model default table (e.g. `encoder.layer4[-1]` for ResNet-family SMP encoders, `encoder.model.stages_3.blocks[-1]` for `tu-` timm encoders, `middle_conv.second` for `UNet_original`) → a fallback that selects the deepest module under the encoder whose class name contains `Conv`/`Block`/`Stage`. `--print-target-layers` resolves and prints the choice without running inference.
- **Preprocessing alignment:** the input tensor is ImageNet-normalized (as in training/evaluation); the overlay is composited on the *de-normalized* post-transform RGB so the heatmap grid aligns exactly with the 512×512 CAM. A `SemanticSegmentationTarget` is built from the label-mapped GT mask at the same spatial size.
- **Outputs** are written under `outputs/gradcam/<ModelName>__<backbone>/<method>/test/` as raw grayscale heatmaps (`*_raw.png`, 0–255) and JET overlays (`*_overlay.png`), with a shared `config_summary.json` recording the configuration and resolved target layers.
- **Memory management:** for large backbones (e.g. `tu-maxvit_large_tf_512` at 512×512) the CAM forward/backward runs under `torch.autocast(dtype=float16)`; the SVD-based methods (`EigenCAM`, `SegEigenCAM`) upcast activations/gradients to float32 (NumPy `linalg.svd` rejects float16), and each CAM's retained autograd graph is explicitly released after use to avoid GPU OOM.

## Directory Structure
- `config/`: Configuration files.
- `utils/`: Utilities for data, transforms, and logging.
- `models/`: Model definitions (`model_zoo.py`).
- `training/`: Training loop and loss functions.
- `evaluation/`: plotting and metric export.
- `outputs/`: Generated artifacts (plots, logs, masks).
- `checkpoints/`: Model weights.
- `gradcam.py`: Grad-CAM / XAI entry point.
- `evaluation/gradcam_utils.py`: CAM method registry, target-layer resolver, and metadata helpers.

## Design Decisions

### Mixed Precision
We use `torch.amp` to reduce memory variance and improve training speed without sacrificing accuracy.

### Reproducibility
All random seeds (Python, NumPy, PyTorch, CUDA) are fixed to a global `SEED`. This ensures that experiments are deterministic and comparable.

### Config-Driven
All hyperparameters (learning rate, batch size, model selection) are defined in `config.yaml`. This allows for rapid experimentation without code changes.

### Automated Output
The system is designed to "run and report". Entry scripts automatically save all relevant metrics and plots to the `outputs/` directory, facilitating offline analysis.

### Explainability by Construction
Grad-CAM outputs are generated from the *same* preprocessing transform and label mapping used in `evaluate.py`, guaranteeing the heatmaps are directly comparable to the reported metrics and confusion matrices. Raw heatmaps are persisted separately from overlays so that quantitative faithfulness metrics (e.g. ROAD, ARCC) can be computed later without re-running inference.

## Dependencies
- PyTorch & Torchvision
- NumPy, OpenCV, Scikit-learn
- Albumentations (Augmentations)
- Segmentation Models PyTorch (SMP)
- Matplotlib, Seaborn (Visualization)
- PyYAML (Config)
- `grad-cam` ([pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)) — Grad-CAM / XAI visualization

## Extending the System
- **Adding Models:** Register new architectures in `models/model_zoo.py`.
- **New Metrics:** Add computations to `training/metrics.py`.
- **Custom Losses:** Implement in `training/losses.py` and register in `train.py`. (Available: `CrossEntropy`, `Dice`, `Focal`, `DiceBCE`, `DiceFocal`).
