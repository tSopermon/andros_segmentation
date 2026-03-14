# Andros Segmentation System Overview

## Introduction
The Andros Segmentation project is a reproducible pipeline for multiclass image segmentation, designed for research and production workflows. It leverages PyTorch for deep learning, focusing on reproducibility, and best practices.

## System Architecture

### Core Components
- **Configuration:** YAML-based configuration (`config/config.yaml`) decoupled from code.
- **Data Pipeline:** Custom `Dataset` class handling image loading, class mapping, and augmentations (Albumentations). Supports on-the-fly caching for efficiency.
- **Model Zoo:** Lazy-loaded models including SMP-backed architectures (DeepLabV3, UNet, etc.) and native paper implementations.
- **Training Engine:** 
    - Loop with mixed precision (AMP) support.
    - Global random seeding for reproducibility.
    - Early stopping and checkpointing (best fold/best overall).
    - K-Fold Cross-Validation and Ensemble support.
    - **Transfer Learning:** Supports fine-tuning from previously trained checkpoints with automatic shape mismatch handling and optional encoder freezing.
- **Evaluation:** Automated metric computation (IoU, F1, Precision, Recall) and visualization generation.

### Workflows

#### 1. Training (`train.py`)
- **Initialization:** Loads config, sets seeds, prepares datasets.
- **Data Splitting/Validation:** Can accept datasets pre-split into train/val (`PRE_SPLIT_DATASET: true`) or perform automated Stratified K-Fold / random splits on a monolithic training folder.
- **K-Fold:** If enabled, splits data, trains per fold, and saves fold checkpoints.
- **Ensembling:** Optionally ensembles fold models for evaluation.
- **Final Model:** Selects best fold configuration and retrains on the full training set for a production-ready model.

#### 2. Evaluation (`evaluate.py`)
- Loads the best checkpoint.
- Computes metrics on the test set.
- Generates plots: Confusion Matrices, Metric distributions, Predictions.

#### 3. Mask Generation (`generate_masks.py`)
- Runs inference on specified datasets.
- Exports color-coded segmentation masks for visual inspection.

## Directory Structure
- `config/`: Configuration files.
- `utils/`: Utilities for data, transforms, and logging.
- `models/`: Model definitions (`model_zoo.py`).
- `training/`: Training loop and loss functions.
- `evaluation/`: plotting and metric export.
- `outputs/`: Generated artifacts (plots, logs, masks).
- `checkpoints/`: Model weights.

## Design Decisions

### Mixed Precision
We use `torch.amp` to reduce memory variance and improve training speed without sacrificing accuracy.

### Reproducibility
All random seeds (Python, NumPy, PyTorch, CUDA) are fixed to a global `SEED`. This ensures that experiments are deterministic and comparable.

### Config-Driven
All hyperparameters (learning rate, batch size, model selection) are defined in `config.yaml`. This allows for rapid experimentation without code changes.

### Automated Output
The system is designed to "run and report". Entry scripts automatically save all relevant metrics and plots to the `outputs/` directory, facilitating offline analysis.

## Dependencies
- PyTorch & Torchvision
- NumPy, OpenCV, Scikit-learn
- Albumentations (Augmentations)
- Segmentation Models PyTorch (SMP)
- Matplotlib, Seaborn (Visualization)
- PyYAML (Config)

## Extending the System
- **Adding Models:** Register new architectures in `models/model_zoo.py`.
- **New Metrics:** Add computations to `training/metrics.py`.
- **Custom Losses:** Implement in `training/losses.py` and register in `train.py`.
