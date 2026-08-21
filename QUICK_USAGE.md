# Quick Usage Guide

This guide provides the fastest path to training a model and running inference with the Andros Segmentation system. For detailed information, see the [README.md](README.md) and [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md).

## 1. Setup Environment

Clone the repository and install dependencies using `pip`:

```bash
git clone <repo_url>
cd <repo_dir>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Before proceeding, create your configuration file from the provided template:

```bash
cp config/config.yaml.template config/config.yaml
```
Open `config/config.yaml` and set your desired parameters (such as `DATASET_PATH`).

## 2. Prepare Data

Ensure your dataset follows this structure:

```text
Dataset/
    train/
        image/
        mask/
    test/
        image/
        mask/
```

Update the `DATASET_DIR` in `config/config.yaml` to point to your `Dataset/` folder.

## 3. Train a Model

Run the training pipeline. By default, this uses the settings in `config/config.yaml`.

```bash
python train.py --config config/config.yaml
```

Checkpoints will be saved in `checkpoints/` and training metrics/plots in `outputs/`.

## 4. Evaluate the Model

Evaluate the best trained model on your test set:

```bash
python evaluate.py --config config/config.yaml
```

Metrics (IoU, F1) and confusion matrices will be saved in `outputs/`.

## 5. Generate Test Masks

To generate and save predicted masks for all test images:

```bash
python generate_masks.py --config config/config.yaml
```

Masks will be saved in `outputs/<ModelName>/masks/`.

## 6. Standalone Inference (Prediction)

To run inference on any new image or folder of images (without needing a dataset structure), use the `predict.py` script:

```bash
python predict.py --input /path/to/my_image.png --model UNetPlusPlus --patch-size 512 --overlap 0.5
```

This will automatically use the default best checkpoint (`checkpoints/<ModelName>_best.pth`) for the specified model.

### Using a Custom Trained Checkpoint

If you have a specific trained checkpoint file that you want to use for testing or prediction, you can provide the path to it using the `--checkpoint` argument:

```bash
python predict.py --input /path/to/test_images_folder/ --model UNetPlusPlus --checkpoint /path/to/my_custom_checkpoint.pth --patch-size 512 --overlap 0.5
```

The script will load your trained weights and save the resulting predictions with color overlays in the `predictions/` directory.

## 7. Grad-CAM / XAI Visualization

Generate Class Activation Maps (CAMs) for your trained model on the test set using the [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library:

```bash
# Full test set, all methods (GradCAM, SegEigenCAM, LayerCAM, EigenCAM, HiResCAM)
python gradcam.py --config config/config.yaml

# Restrict to one model and the first N images (fast sanity run)
python gradcam.py --config config/config.yaml --models UNetPlusPlus --limit 2

# Restrict to specific methods
python gradcam.py --config config/config.yaml --methods GradCAM,EigenCAM --limit 5

# Debug: resolve and print each model's target layer, then exit
python gradcam.py --config config/config.yaml --print-target-layers
```

Outputs are written to `outputs/gradcam/<ModelName>__<backbone>/<method>/test/`:

- `<base>__<ClassName>__raw.png` — normalized grayscale heatmap (0–255).
- `<base>__<ClassName>__overlay.png` — JET-colormap heatmap composited on the image.
- `EigenCAM` is class-agnostic and emits one map per image (`<base>__raw.png`); the
  other methods are class-discriminative and emit one map per class present in the
  ground-truth mask.
- A shared `outputs/gradcam/config_summary.json` records the training configuration
  and resolved target layers for reproducibility.

See `scientific_docs/GRADCAM_DOCUMENTATION.md` for the underlying theory and
interpretation guidance.
