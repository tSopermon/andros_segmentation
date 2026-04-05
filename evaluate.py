"""
Evaluation script for segmentation models.

Loads trained model checkpoints, computes evaluation metrics, generates plots, and exports results.

Usage:
    python evaluate.py --config config/config.yaml

Outputs:
    - Metrics CSVs in outputs/
    - Plots in outputs/
    - Predicted masks in outputs/<ModelName>/masks/
"""
import torch
import numpy as np
from pathlib import Path
from utils.config_loader import load_config
from utils.model_selection import get_selected_model_names
from utils.dataset import SegmentationDataset
from utils.transforms import get_val_transform
from models.model_zoo import get_models
from training.metrics import SegmentationMetrics
from training.train_utils import evaluate_model
from evaluation.visualization import visualize_predictions
from evaluation.export_metrics import export_metrics
import os
from evaluation.plots import (
    plot_confusion_matrices,
    plot_metric_vs_class_frequency,
    plot_per_image_metric_distribution,
    plot_metric_correlation_matrix,
    plot_metric_per_class,
    plot_mean_metrics,
    plot_metric_per_model_per_class,
    plot_all_averages
)
import logging
from utils.logging_config import configure_logging

import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate segmentation models.')
parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML file')
args = parser.parse_args()

# Load config
config = load_config(args.config)
DATASET_PATH = Path(config['DATASET_PATH'])
TEST_IMG_PATH = DATASET_PATH / 'test' / ('Image' if (DATASET_PATH / 'test' / 'Image').exists() else 'image')
TEST_MASK_PATH = DATASET_PATH / 'test' / ('Mask' if (DATASET_PATH / 'test' / 'Mask').exists() else 'mask')
IMAGE_SIZE = config['IMAGE_SIZE']
BATCH_SIZE = config['BATCH_SIZE']
NUM_WORKERS = config['NUM_WORKERS']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)
logger.info(f"Using device: {device}")

# Get image and mask files
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
import cv2

test_images = sorted([f for f in os.listdir(TEST_IMG_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
test_masks = sorted([f for f in os.listdir(TEST_MASK_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])

# Scan ALL masks in dataset to define consistent NUM_CLASSES and label_mapping
all_classes = set()
for subset in ['train', 'val', 'test', 'lowres']:
    mask_dir = DATASET_PATH / subset / ('Mask' if (DATASET_PATH / subset / 'Mask').exists() else 'mask')
    if mask_dir.exists():
        for mask_file in os.listdir(mask_dir):
            if mask_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                mask = cv2.imread(str(mask_dir / mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    all_classes.update(np.unique(mask).tolist())

NUM_CLASSES = config.get('NUM_CLASSES', len(all_classes))
if NUM_CLASSES == 0:
    raise RuntimeError("No masks found to infer NUM_CLASSES and not provided in config.")

class_labels = sorted(all_classes)
label_mapping = {original: idx for idx, original in enumerate(class_labels)}

# Human-readable class names (index -> name)
if NUM_CLASSES == 8:
    # Classic Andros Dataset
    class_names = [
        'Water',
        'Woodland',
        'Arable land',
        'Frygana',
        'Other',
        'Artificial land',
        'Perm. Cult',
        'Bareland'
    ]
else:
    # Modify these for your other dataset
    class_names = [f'Class_{i}' for i in range(NUM_CLASSES)]
# Ensure class_names length matches detected number of classes
if len(class_names) != len(class_labels):
    logger = logging.getLogger(__name__)
    logger.warning('Provided class names length (%d) does not match detected classes (%d). Truncating or padding names.',
                   len(class_names), len(class_labels))
    if len(class_names) > len(class_labels):
        class_names = class_names[:len(class_labels)]
    else:
        class_names = class_names + [f'Class_{i}' for i in range(len(class_names), len(class_labels))]

# Datasets and loaders
val_transform = get_val_transform(IMAGE_SIZE)
test_dataset = SegmentationDataset(TEST_IMG_PATH, TEST_MASK_PATH, test_images, test_masks, val_transform, label_mapping)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Models
# Ensure model_zoo will register requested original DeepLab variants when configured
import os
# Temporarily set env vars so get_models will register originals when required,
# but avoid leaving these env vars set globally during test collection.
old_env = {k: os.environ.get(k) for k in ('USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL', 'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
try:
    os.environ['USE_UNET_ORIGINAL'] = str(config.get('USE_UNET_ORIGINAL', False)).lower()
    os.environ['USE_DEEPLABV1_ORIGINAL'] = str(config.get('USE_DEEPLABV1_ORIGINAL', False)).lower()
    os.environ['USE_DEEPLABV2_ORIGINAL'] = str(config.get('USE_DEEPLABV2_ORIGINAL', False)).lower()
    os.environ['USE_DEEPLABV3_ORIGINAL'] = str(config.get('USE_DEEPLABV3_ORIGINAL', False)).lower()
    os.environ['USE_MAXVIT_UNET'] = str(config.get('USE_MAXVIT_UNET', False)).lower()

    models_dict = get_models(NUM_CLASSES, backbone=config.get('BACKBONE', 'resnet101'), encoder_weights=config.get('ENCODER_WEIGHTS', 'imagenet'))
finally:
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

# Select which models to evaluate based on MODEL_SET + STANDARD_MODELS config
selected_models = get_selected_model_names(config)

models_dict = {k: v for k, v in models_dict.items() if k in selected_models}

# Ensure missing checkpoints exit properly
missing = []
for model_name in list(models_dict.keys()):
    path = f"checkpoints/{model_name}_best.pth"
    if not os.path.exists(path):
        logger.error('No such file: %s', path)
        missing.append(path)

if len(missing) > 0:
    raise SystemExit(2)

for model_name in list(models_dict.keys()):
    models_dict[model_name] = models_dict[model_name].to(device)
    models_dict[model_name].load_state_dict(torch.load(f"checkpoints/{model_name}_best.pth"))
    models_dict[model_name].eval()

# Load training history
training_history = np.load('outputs/training_history.npy', allow_pickle=True).item()

# Evaluate
all_test_results = {}
all_preds_dict = {}
all_targets_list = []
targets_collected = False

from tqdm import tqdm

for model_name, model in models_dict.items():
    test_metrics = SegmentationMetrics(NUM_CLASSES)
    model.eval()
    test_metrics.reset()
    all_preds_dict[model_name] = []
    
    logger.info(f"Evaluating {model_name} on test set...")
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc=f"Eval {model_name}"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            test_metrics.update(outputs, masks)
            
            # Cache predictions for plotting later
            preds = torch.argmax(outputs, dim=1).cpu().numpy().astype(np.uint8)
            all_preds_dict[model_name].append(preds)
            
            if not targets_collected:
                all_targets_list.append(masks.cpu().numpy().astype(np.uint8))
                
    targets_collected = True
    all_preds_dict[model_name] = np.concatenate(all_preds_dict[model_name], axis=0)
    
    test_metrics_dict = test_metrics.compute_metrics()
    all_test_results[model_name] = test_metrics_dict
    logger.info("%s", "\n" + model_name.upper())
    logger.info('%s', '-' * 80)
    logger.info("%s", f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
    logger.info('%s', '-' * 80)
    for c in range(NUM_CLASSES):
        cname = class_names[c] if c < len(class_names) else f'Class_{c}'
        logger.info(f"{cname:<20} {test_metrics_dict['precision'][c]:<12.4f} {test_metrics_dict['recall'][c]:<12.4f} "
                    f"{test_metrics_dict['f1'][c]:<12.4f} {test_metrics_dict['iou'][c]:<12.4f}")
    logger.info('%s', '-' * 80)
    logger.info(f"{'MEAN (Macro)':<20} {test_metrics_dict['precision_mean']:<12.4f} {test_metrics_dict['recall_mean']:<12.4f} "
                f"{test_metrics_dict['f1_mean']:<12.4f} {test_metrics_dict['iou_mean']:<12.4f}")
    logger.info(f"{'Weighted':<20} {test_metrics_dict['precision_weighted']:<12.4f} {test_metrics_dict['recall_weighted']:<12.4f} "
                f"{test_metrics_dict['f1_weighted']:<12.4f} {test_metrics_dict['iou_weighted']:<12.4f}")
    logger.info(f"{'Micro':<20} {test_metrics_dict['precision_micro']:<12.4f} {test_metrics_dict['recall_micro']:<12.4f} "
                f"{test_metrics_dict['f1_micro']:<12.4f} {test_metrics_dict['iou_micro']:<12.4f}")
    logger.info('%s', '=' * 80)

# Consolidate targets
all_targets = np.concatenate(all_targets_list, axis=0)

# Visualizations
visualize_predictions(models_dict, test_dataset, test_images, test_masks, TEST_IMG_PATH, TEST_MASK_PATH, label_mapping, device, class_names=class_names)

# --- Generate Plots ---
os.makedirs("outputs", exist_ok=True)
plot_metric_per_class(all_test_results, 'precision', class_names)
plot_metric_per_class(all_test_results, 'recall', class_names)
plot_metric_per_class(all_test_results, 'f1', class_names)
plot_metric_per_class(all_test_results, 'iou', class_names)
plot_mean_metrics(all_test_results)
plot_all_averages(all_test_results)
plot_metric_per_model_per_class(all_test_results, class_names)
plot_confusion_matrices(all_preds_dict, all_targets, label_mapping, class_names=class_names, output_dir="outputs")
plot_metric_vs_class_frequency(all_test_results, all_targets, label_mapping=label_mapping, class_names=class_names, output_dir="outputs")
plot_per_image_metric_distribution(all_preds_dict, all_targets, class_names=class_names, output_dir="outputs")
plot_metric_correlation_matrix(all_test_results, output_dir="outputs")

# Export metrics
export_metrics(models_dict, all_test_results, training_history, NUM_CLASSES, class_names=class_names, output_dir="outputs")
