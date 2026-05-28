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

## 6. Standalone Inference

To run inference on any new image or folder of images (without needing a dataset structure), use the `predict.py` script:

```bash
python predict.py --input /path/to/my_image.png --model UNetPlusPlus --patch-size 512 --overlap 0.5
```

This will automatically use the best checkpoint for the specified model and save the predictions with color overlays in the `predictions/` directory.
