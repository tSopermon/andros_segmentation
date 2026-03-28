"""
Training script for segmentation models.

Loads configuration from config/config.yaml, trains the model, saves checkpoints, and logs training history.

Usage:
    python train.py --config config/config.yaml

Outputs:
    - Model checkpoints in checkpoints/
    - Training history in outputs/training_history.npy
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import torch
import numpy as np
from pathlib import Path
from utils.config_loader import load_config
from utils.model_selection import get_active_original_models, get_selected_model_names
from utils.dataset import SegmentationDataset, count_pixels, get_pixel_counts_cache
from utils.transforms import get_train_transform, get_val_transform
from models.model_zoo import get_models
from training.metrics import SegmentationMetrics
from training.train_utils import train_epoch, validate, apply_transfer_learning, freeze_encoder_if_requested
from training.losses import DiceLoss, DiceBCELoss, FocalLoss, DiceFocalLoss
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import cv2
import random
import logging
import datetime
from utils.logging_config import configure_logging, add_file_handler
from sklearn.model_selection import StratifiedKFold
import warnings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML')
args = parser.parse_args()

# Load config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
config = load_config(args.config)
DATASET_PATH = Path(config['DATASET_PATH'])
TRAIN_IMG_PATH = DATASET_PATH / 'train' / ('Image' if (DATASET_PATH / 'train' / 'Image').exists() else 'image')
TRAIN_MASK_PATH = DATASET_PATH / 'train' / ('Mask' if (DATASET_PATH / 'train' / 'Mask').exists() else 'mask')
TEST_IMG_PATH = DATASET_PATH / 'test' / ('Image' if (DATASET_PATH / 'test' / 'Image').exists() else 'image')
TEST_MASK_PATH = DATASET_PATH / 'test' / ('Mask' if (DATASET_PATH / 'test' / 'Mask').exists() else 'mask')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', config['BATCH_SIZE']))
# Allow short smoke runs via env override
MAX_EPOCHS = int(os.environ.get('MAX_EPOCHS', config['MAX_EPOCHS']))
LEARNING_RATE = config['LEARNING_RATE']
LR_DECAY_GAMMA = config['LR_DECAY_GAMMA']
IMAGE_SIZE = config['IMAGE_SIZE']
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', config['NUM_WORKERS']))
USE_AUGMENTATION = config['USE_AUGMENTATION']
PATIENCE = config['PATIENCE']
MIN_EPOCHS = config['MIN_EPOCHS']
MIN_DELTA = config['MIN_DELTA']
LR_PATIENCE_THRESHOLD = config['LR_PATIENCE_THRESHOLD']
LR_PATIENCE_SCALE = config['LR_PATIENCE_SCALE']
K_FOLDS = int(os.environ.get('K_FOLDS', config.get('K_FOLDS', 1)))
ENSEMBLE = str(os.environ.get('ENSEMBLE', config.get('ENSEMBLE', False))).lower() in ('1', 'true', 'yes')
# Allow overriding encoder weights to avoid downloads in smoke tests
ENCODER_WEIGHTS = os.environ.get('ENCODER_WEIGHTS', config.get('ENCODER_WEIGHTS', None))
if ENCODER_WEIGHTS in ('', 'None'):
    ENCODER_WEIGHTS = None

# Transfer Learning config
TRANSFER_LEARNING = str(os.environ.get('TRANSFER_LEARNING', config.get('TRANSFER_LEARNING', False))).lower() in ('1', 'true', 'yes')
PRETRAINED_CHECKPOINT_DIR = os.environ.get('PRETRAINED_CHECKPOINT_DIR', config.get('PRETRAINED_CHECKPOINT_DIR', 'checkpoints/'))
FREEZE_ENCODER = str(os.environ.get('FREEZE_ENCODER', config.get('FREEZE_ENCODER', False))).lower() in ('1', 'true', 'yes')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

# ── File logging ──────────────────────────────────────────────────────────────
_run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = os.path.join("logs", f"train_{_run_ts}.log")
add_file_handler(_log_file, level=config.get('LOGGING_LEVEL', 'INFO'))
logger.info("Log file: %s", os.path.abspath(_log_file))

# ── Configuration summary ─────────────────────────────────────────────────────
logger.info("=" * 70)
logger.info("TRAINING CONFIGURATION")
logger.info("=" * 70)
logger.info("Config file : %s", args.config)
logger.info("Run started : %s", datetime.datetime.now().isoformat(sep=' ', timespec='seconds'))
logger.info("Seed        : %d", SEED)
logger.info("Device      : %s", device)
logger.info("-" * 70)
_max_key = max(len(k) for k in config) if config else 0
for _key, _val in sorted(config.items()):
    logger.info("  %-*s : %s", _max_key, _key, _val)
logger.info("=" * 70)

logger.info("Using device: %s", device)
if TRANSFER_LEARNING:
    logger.info("Transfer Learning ENABLED. Checkpoints dir: %s. Freeze Encoder: %s", PRETRAINED_CHECKPOINT_DIR, FREEZE_ENCODER)

# Pre-split configurations
PRE_SPLIT_DATASET = config.get('PRE_SPLIT_DATASET', False)
if PRE_SPLIT_DATASET:
    logger.info("PRE_SPLIT_DATASET is true: forcing K_FOLDS=1 and ENSEMBLE=False")
    K_FOLDS = 1
    ENSEMBLE = False

# Get image and mask files
image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
train_images = sorted([f for f in os.listdir(TRAIN_IMG_PATH) if f.lower().endswith(image_extensions)])
train_masks = sorted([f for f in os.listdir(TRAIN_MASK_PATH) if f.lower().endswith(image_extensions)])

if PRE_SPLIT_DATASET:
    VAL_IMG_PATH = DATASET_PATH / 'val' / ('Image' if (DATASET_PATH / 'val' / 'Image').exists() else 'image')
    VAL_MASK_PATH = DATASET_PATH / 'val' / ('Mask' if (DATASET_PATH / 'val' / 'Mask').exists() else 'mask')
    val_images = sorted([f for f in os.listdir(VAL_IMG_PATH) if f.lower().endswith(image_extensions)])
    val_masks = sorted([f for f in os.listdir(VAL_MASK_PATH) if f.lower().endswith(image_extensions)])
else:
    VAL_IMG_PATH = None
    VAL_MASK_PATH = None
    val_images = []
    val_masks = []

test_images = sorted([f for f in os.listdir(TEST_IMG_PATH) if f.lower().endswith(image_extensions)])
test_masks = sorted([f for f in os.listdir(TEST_MASK_PATH) if f.lower().endswith(image_extensions)])

# Scan masks for class labels
all_classes = set()
for mask_file in train_masks:
    mask = cv2.imread(str(TRAIN_MASK_PATH / mask_file), cv2.IMREAD_GRAYSCALE)
    all_classes.update(np.unique(mask).tolist())
if PRE_SPLIT_DATASET:
    for mask_file in val_masks:
        mask = cv2.imread(str(VAL_MASK_PATH / mask_file), cv2.IMREAD_GRAYSCALE)
        all_classes.update(np.unique(mask).tolist())
for mask_file in test_masks:
    mask = cv2.imread(str(TEST_MASK_PATH / mask_file), cv2.IMREAD_GRAYSCALE)
    all_classes.update(np.unique(mask).tolist())
NUM_CLASSES = len(all_classes)
class_labels = sorted(all_classes)
label_mapping = {original: idx for idx, original in enumerate(class_labels)}

# Train/val split
if K_FOLDS == 1:
    if PRE_SPLIT_DATASET:
        train_images_split = train_images
        train_masks_split = train_masks
        # val_images and val_masks are already loaded
    else:
        train_images_split, val_images = train_test_split(train_images, test_size=0.2, random_state=42)
        train_masks_split, val_masks = train_test_split(train_masks, test_size=0.2, random_state=42)
else:
    # For stratified k-fold we compute a primary label per image (most common mapped label)
    primary_labels = []
    for mname in train_masks:
        m = cv2.imread(str(TRAIN_MASK_PATH / mname), cv2.IMREAD_GRAYSCALE)
        mapped = np.zeros_like(m)
        for orig, idx in label_mapping.items():
            mapped[m == orig] = idx
        vals, freqs = np.unique(mapped, return_counts=True)
        if len(vals) == 0:
            primary = 0
        else:
            primary = int(vals[np.argmax(freqs)])
        primary_labels.append(primary)
    # If there is only one class present, fallback to simple KFold splitting
    try:
        skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
        folds = list(skf.split(np.arange(len(train_masks)), np.array(primary_labels)))
    except Exception:
        warnings.warn('StratifiedKFold failed — falling back to simple splits')
        # Create simple contiguous fold indices
        n = len(train_masks)
        indices = np.arange(n)
        folds = []
        sizes = np.full(K_FOLDS, n // K_FOLDS)
        sizes[:(n % K_FOLDS)] += 1
        start = 0
        for sz in sizes:
            val_idx = indices[start:start+sz]
            train_idx = np.setdiff1d(indices, val_idx)
            folds.append((train_idx, val_idx))
            start += sz

# Transforms
train_transform = get_train_transform(IMAGE_SIZE, USE_AUGMENTATION)
val_transform = get_val_transform(IMAGE_SIZE)

# Datasets and loaders (only create the one-shot split when not using k-fold)
if K_FOLDS == 1:
    val_img_path = VAL_IMG_PATH if PRE_SPLIT_DATASET else TRAIN_IMG_PATH
    val_mask_path = VAL_MASK_PATH if PRE_SPLIT_DATASET else TRAIN_MASK_PATH
    train_dataset = SegmentationDataset(TRAIN_IMG_PATH, TRAIN_MASK_PATH, train_images_split, train_masks_split, train_transform, label_mapping)
    val_dataset = SegmentationDataset(val_img_path, val_mask_path, val_images, val_masks, val_transform, label_mapping)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

BACKBONE = config.get('BACKBONE', 'resnet101')

# Determine model set from config: `MODEL_SET` takes precedence. Choices: 'standard', 'originals', 'all'.
model_set = config.get('MODEL_SET', 'standard')
ORIGINAL_MODELS = get_active_original_models(config)

# If running originals or all, enable original registrations in model_zoo via env vars.
# Do this temporarily while training to avoid polluting test environments when this
# module is imported by unit tests.
old_env = {k: os.environ.get(k) for k in ('USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL', 'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
os.environ['USE_UNET_ORIGINAL'] = str(config.get('USE_UNET_ORIGINAL', False)).lower()
os.environ['USE_DEEPLABV1_ORIGINAL'] = str(config.get('USE_DEEPLABV1_ORIGINAL', False)).lower()
os.environ['USE_DEEPLABV2_ORIGINAL'] = str(config.get('USE_DEEPLABV2_ORIGINAL', False)).lower()
os.environ['USE_DEEPLABV3_ORIGINAL'] = str(config.get('USE_DEEPLABV3_ORIGINAL', False)).lower()
os.environ['USE_MAXVIT_UNET'] = str(config.get('USE_MAXVIT_UNET', False)).lower()

# Select model names based on the requested set
MODEL_NAMES = get_selected_model_names(config)

# Helper to create dataloaders for a given split
def make_loaders(train_imgs, train_msks, val_imgs, val_msks):
    val_img_path = VAL_IMG_PATH if PRE_SPLIT_DATASET else TRAIN_IMG_PATH
    val_mask_path = VAL_MASK_PATH if PRE_SPLIT_DATASET else TRAIN_MASK_PATH
    train_dataset = SegmentationDataset(TRAIN_IMG_PATH, TRAIN_MASK_PATH, train_imgs, train_msks, train_transform, label_mapping)
    val_dataset = SegmentationDataset(val_img_path, val_mask_path, val_imgs, val_msks, val_transform, label_mapping)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader

# Training loop
training_history = {}
# Training per model (and per-fold when K_FOLDS>1)
training_history = {}
for model_name in MODEL_NAMES:
    metric_keys = [
        'train_loss', 'val_loss', 
        'train_f1_mean', 'train_precision_mean', 'train_recall_mean', 'train_iou_mean',
        'train_f1_weighted', 'train_precision_weighted', 'train_recall_weighted', 'train_iou_weighted',
        'train_f1_micro', 'train_precision_micro', 'train_recall_micro', 'train_iou_micro',
        'val_f1_mean', 'val_precision_mean', 'val_recall_mean', 'val_iou_mean',
        'val_f1_weighted', 'val_precision_weighted', 'val_recall_weighted', 'val_iou_weighted',
        'val_f1_micro', 'val_precision_micro', 'val_recall_micro', 'val_iou_micro'
    ]
    training_history[model_name] = {k: [] for k in metric_keys}
    logger.info("\n%s\nTraining %s\n%s", '='*70, model_name, '='*70)
    # K_FOLDS == 1: use the previously-created single split
    if K_FOLDS == 1:
        # compute class weights from the training split (use full train masks similar to original behavior)
        train_counts = count_pixels(TRAIN_MASK_PATH, train_masks, label_mapping)
        class_weights = 1.0 / np.maximum(train_counts.astype(np.float64), 1.0)
        class_weights = class_weights / class_weights.mean()
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # instantiate model
        model = get_models(NUM_CLASSES, backbone=BACKBONE, encoder_weights=ENCODER_WEIGHTS, specific_model=model_name)[model_name].to(device)
        
        if TRANSFER_LEARNING:
            pretrained_suffix = config.get('PRETRAINED_WEIGHT_SUFFIX', '_best.pth')
            ckpt_path = os.path.join(PRETRAINED_CHECKPOINT_DIR, f"{model_name}{pretrained_suffix}")
            apply_transfer_learning(model, ckpt_path, device)
            freeze_encoder_if_requested(model, FREEZE_ENCODER)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)
        
        loss_fn_name = config.get('LOSS_FUNCTION', 'CrossEntropy')
        if loss_fn_name == 'DiceBCE':
            # Use combined Dice + CrossEntropy (weighted)
            criterion = DiceBCELoss(weight=class_weights_tensor)
        elif loss_fn_name == 'DiceFocal':
            dice_w = float(config.get('DICE_WEIGHT', 1.0))
            focal_w = float(config.get('FOCAL_WEIGHT', 1.0))
            criterion = DiceFocalLoss(weight=class_weights_tensor, dice_weight=dice_w, focal_weight=focal_w)
        elif loss_fn_name == 'Dice':
             criterion = DiceLoss() 
        elif loss_fn_name == 'Focal':
             criterion = FocalLoss(weight=class_weights_tensor)
        else:
             criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
             
        train_metrics = SegmentationMetrics(NUM_CLASSES)
        val_metrics = SegmentationMetrics(NUM_CLASSES)
        best_val_iou = 0.0
        patience_counter = 0
        initial_lr = LEARNING_RATE
        train_loader, val_loader = make_loaders(train_images_split, train_masks_split, val_images, val_masks)
        for epoch in range(MAX_EPOCHS):
            current_lr = optimizer.param_groups[0]['lr']
            train_loss, train_metrics_dict = train_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics,
                epoch=epoch+1, max_epochs=MAX_EPOCHS, lr=current_lr, phase='Training')
            val_loss, val_metrics_dict = validate(
                model, val_loader, criterion, device, val_metrics,
                epoch=epoch+1, max_epochs=MAX_EPOCHS, lr=current_lr, phase='Validation')
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            effective_patience = int(PATIENCE * LR_PATIENCE_SCALE) if current_lr <= initial_lr * LR_PATIENCE_THRESHOLD else PATIENCE
            # Store metrics
            training_history[model_name]['train_loss'].append(train_loss)
            training_history[model_name]['val_loss'].append(val_loss)
            for m_type in ['mean', 'weighted', 'micro']:
                training_history[model_name][f'train_f1_{m_type}'].append(train_metrics_dict[f'f1_{m_type}'])
                training_history[model_name][f'train_precision_{m_type}'].append(train_metrics_dict[f'precision_{m_type}'])
                training_history[model_name][f'train_recall_{m_type}'].append(train_metrics_dict[f'recall_{m_type}'])
                training_history[model_name][f'train_iou_{m_type}'].append(train_metrics_dict[f'iou_{m_type}'])
                training_history[model_name][f'val_f1_{m_type}'].append(val_metrics_dict[f'f1_{m_type}'])
                training_history[model_name][f'val_precision_{m_type}'].append(val_metrics_dict[f'precision_{m_type}'])
                training_history[model_name][f'val_recall_{m_type}'].append(val_metrics_dict[f'recall_{m_type}'])
                training_history[model_name][f'val_iou_{m_type}'].append(val_metrics_dict[f'iou_{m_type}'])
            # Early stopping
            if val_metrics_dict['iou_mean'] > best_val_iou + MIN_DELTA:
                best_val_iou = val_metrics_dict['iou_mean']
                patience_counter = 0
                torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")
            else:
                patience_counter += 1
                if (epoch + 1) >= MIN_EPOCHS and patience_counter >= effective_patience:
                    logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch+1, effective_patience)
                    break
        logger.info("✓ %s training completed. Best mIoU: %.4f", model_name, best_val_iou)
        model.cpu()
        del model, optimizer, scheduler, criterion, train_metrics, val_metrics
        torch.cuda.empty_cache()
    else:
        # K-FOLD training
        logger.info("Pre-computing pixel counts for all masks to speed up K-Fold initialization...")
        pixel_counts_cache = get_pixel_counts_cache(TRAIN_MASK_PATH, train_masks, label_mapping)
        fold_idx = 0
        # keep per-fold best metrics so we can pick the best fold to initialize full-data retrain
        fold_best_iou = []
        for train_idx, val_idx in folds:
            fold_idx += 1
            train_imgs_fold = [train_images[i] for i in train_idx]
            train_msks_fold = [train_masks[i] for i in train_idx]
            val_imgs_fold = [train_images[i] for i in val_idx]
            val_msks_fold = [train_masks[i] for i in val_idx]
            logger.info('Fold %d: %d train, %d val', fold_idx, len(train_imgs_fold), len(val_imgs_fold))
            # per-fold class weights
            train_counts = np.sum([pixel_counts_cache[fname] for fname in train_msks_fold], axis=0)
            class_weights = 1.0 / np.maximum(train_counts.astype(np.float64), 1.0)
            class_weights = class_weights / class_weights.mean()
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

            # instantiate model per fold
            model = get_models(NUM_CLASSES, backbone=BACKBONE, encoder_weights=ENCODER_WEIGHTS, specific_model=model_name)[model_name].to(device)
            
            if TRANSFER_LEARNING:
                pretrained_suffix = config.get('PRETRAINED_WEIGHT_SUFFIX', '_best.pth')
                if pretrained_suffix == '_best.pth':
                    fold_ckpt_path = os.path.join(PRETRAINED_CHECKPOINT_DIR, f"{model_name}_fold{fold_idx}{pretrained_suffix}")
                    best_ckpt_path = os.path.join(PRETRAINED_CHECKPOINT_DIR, f"{model_name}{pretrained_suffix}")
                    if os.path.exists(fold_ckpt_path):
                        apply_transfer_learning(model, fold_ckpt_path, device)
                    else:
                        apply_transfer_learning(model, best_ckpt_path, device)
                else:
                    custom_ckpt_path = os.path.join(PRETRAINED_CHECKPOINT_DIR, f"{model_name}{pretrained_suffix}")
                    apply_transfer_learning(model, custom_ckpt_path, device)
                freeze_encoder_if_requested(model, FREEZE_ENCODER)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)
            
            loss_fn_name = config.get('LOSS_FUNCTION', 'CrossEntropy')
            if loss_fn_name == 'DiceBCE':
                criterion = DiceBCELoss(weight=class_weights_tensor)
            elif loss_fn_name == 'DiceFocal':
                dice_w = float(config.get('DICE_WEIGHT', 1.0))
                focal_w = float(config.get('FOCAL_WEIGHT', 1.0))
                criterion = DiceFocalLoss(weight=class_weights_tensor, dice_weight=dice_w, focal_weight=focal_w)
            elif loss_fn_name == 'Dice':
                criterion = DiceLoss()
            elif loss_fn_name == 'Focal':
                criterion = FocalLoss(weight=class_weights_tensor)
            else:
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
                
            train_metrics = SegmentationMetrics(NUM_CLASSES)
            val_metrics = SegmentationMetrics(NUM_CLASSES)
            best_val_iou = 0.0
            patience_counter = 0
            initial_lr = LEARNING_RATE
            train_loader, val_loader = make_loaders(train_imgs_fold, train_msks_fold, val_imgs_fold, val_msks_fold)
            for epoch in range(MAX_EPOCHS):
                current_lr = optimizer.param_groups[0]['lr']
                train_loss, train_metrics_dict = train_epoch(
                    model, train_loader, criterion, optimizer, device, train_metrics,
                    epoch=epoch+1, max_epochs=MAX_EPOCHS, lr=current_lr, phase='Training')
                val_loss, val_metrics_dict = validate(
                    model, val_loader, criterion, device, val_metrics,
                    epoch=epoch+1, max_epochs=MAX_EPOCHS, lr=current_lr, phase='Validation')
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                effective_patience = int(PATIENCE * LR_PATIENCE_SCALE) if current_lr <= initial_lr * LR_PATIENCE_THRESHOLD else PATIENCE
                # Store metrics (append per-fold epoch history)
                training_history[model_name]['train_loss'].append(train_loss)
                training_history[model_name]['val_loss'].append(val_loss)
                for m_type in ['mean', 'weighted', 'micro']:
                    training_history[model_name][f'train_f1_{m_type}'].append(train_metrics_dict[f'f1_{m_type}'])
                    training_history[model_name][f'train_precision_{m_type}'].append(train_metrics_dict[f'precision_{m_type}'])
                    training_history[model_name][f'train_recall_{m_type}'].append(train_metrics_dict[f'recall_{m_type}'])
                    training_history[model_name][f'train_iou_{m_type}'].append(train_metrics_dict[f'iou_{m_type}'])
                    training_history[model_name][f'val_f1_{m_type}'].append(val_metrics_dict[f'f1_{m_type}'])
                    training_history[model_name][f'val_precision_{m_type}'].append(val_metrics_dict[f'precision_{m_type}'])
                    training_history[model_name][f'val_recall_{m_type}'].append(val_metrics_dict[f'recall_{m_type}'])
                    training_history[model_name][f'val_iou_{m_type}'].append(val_metrics_dict[f'iou_{m_type}'])
                # Early stopping
                if val_metrics_dict['iou_mean'] > best_val_iou + MIN_DELTA:
                    best_val_iou = val_metrics_dict['iou_mean']
                    patience_counter = 0
                    torch.save(model.state_dict(), f"checkpoints/{model_name}_fold{fold_idx}_best.pth")
                else:
                    patience_counter += 1
                    if (epoch + 1) >= MIN_EPOCHS and patience_counter >= effective_patience:
                        logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch+1, effective_patience)
                        break
            logger.info("✓ %s fold %d training completed. Best mIoU: %.4f", model_name, fold_idx, best_val_iou)
            # record best iou for this fold
            fold_best_iou.append(best_val_iou)
            model.cpu()
            del model, optimizer, scheduler, criterion, train_metrics, val_metrics
            torch.cuda.empty_cache()

        # After all folds, optionally perform ensembling on test set
        if ENSEMBLE:
            logger.info('Performing ensemble evaluation for %s', model_name)
            # prepare test loader
            test_dataset = SegmentationDataset(TEST_IMG_PATH, TEST_MASK_PATH, test_images, test_masks, val_transform, label_mapping)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
            ensemble_models = []
            for fold_i in range(1, K_FOLDS+1):
                ckpt = f'checkpoints/{model_name}_fold{fold_i}_best.pth'
                if not os.path.exists(ckpt):
                    logger.warning('Missing checkpoint for fold %d: %s', fold_i, ckpt)
                    continue
                m = get_models(NUM_CLASSES, backbone=BACKBONE, encoder_weights=ENCODER_WEIGHTS, specific_model=model_name)[model_name].to(device)
                m.load_state_dict(torch.load(ckpt, map_location=device))
                m.eval()
                ensemble_models.append(m)
            if len(ensemble_models) == 0:
                logger.warning('No fold checkpoints found for ensemble for %s', model_name)
            else:
                metrics = SegmentationMetrics(NUM_CLASSES)
                metrics.reset()
                with torch.no_grad():
                    for images, masks in test_loader:
                        images = images.to(device)
                        masks = masks.to(device)
                        sum_logits = None
                        for m in ensemble_models:
                            out = m(images)
                            if sum_logits is None:
                                sum_logits = out
                            else:
                                sum_logits = sum_logits + out
                        avg_logits = sum_logits / float(len(ensemble_models))
                        metrics.update(avg_logits, masks)
                ensemble_metrics = metrics.compute_metrics()
                logger.info('Ensemble test metrics for %s: %s', model_name, str(ensemble_metrics))
                # cleanup ensemble models
                for m in ensemble_models:
                    m.cpu()
                    del m
                torch.cuda.empty_cache()

        # Retrain a single model on the full training set using the best-performing fold as initialization
        try:
            import numpy as _np
            if len(fold_best_iou) > 0:
                best_fold_idx = int(_np.argmax(_np.array(fold_best_iou))) + 1
            else:
                best_fold_idx = 1
        except Exception:
            best_fold_idx = 1

        logger.info('Retraining %s on full training data using fold %d as initialization', model_name, best_fold_idx)

        # compute class weights on full training masks
        # Reuse cache since train_masks covers all files
        train_counts_full = np.sum([pixel_counts_cache[fname] for fname in train_masks], axis=0)
        class_weights_full = 1.0 / np.maximum(train_counts_full.astype(np.float64), 1.0)
        class_weights_full = class_weights_full / class_weights_full.mean()
        class_weights_tensor_full = torch.tensor(class_weights_full, dtype=torch.float32, device=device)

        # instantiate and (optionally) load best-fold weights
        model = get_models(NUM_CLASSES, backbone=BACKBONE, encoder_weights=ENCODER_WEIGHTS, specific_model=model_name)[model_name].to(device)
        best_fold_ckpt = f'checkpoints/{model_name}_fold{best_fold_idx}_best.pth'
        if os.path.exists(best_fold_ckpt):
            try:
                state = torch.load(best_fold_ckpt, map_location=device)
                model.load_state_dict(state)
                logger.info('Loaded weights from %s to initialize full-data retrain', best_fold_ckpt)
            except Exception:
                logger.warning('Failed to load checkpoint %s — training from scratch', best_fold_ckpt)

        if TRANSFER_LEARNING:
            freeze_encoder_if_requested(model, FREEZE_ENCODER)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_DECAY_GAMMA)
        
        loss_fn_name = config.get('LOSS_FUNCTION', 'CrossEntropy')
        if loss_fn_name == 'DiceBCE':
            criterion = DiceBCELoss(weight=class_weights_tensor_full)
        elif loss_fn_name == 'DiceFocal':
            dice_w = float(config.get('DICE_WEIGHT', 1.0))
            focal_w = float(config.get('FOCAL_WEIGHT', 1.0))
            criterion = DiceFocalLoss(weight=class_weights_tensor_full, dice_weight=dice_w, focal_weight=focal_w)
        elif loss_fn_name == 'Dice':
             criterion = DiceLoss()
        elif loss_fn_name == 'Focal':
             criterion = FocalLoss(weight=class_weights_tensor_full)
        else:
             criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor_full)
             
        train_metrics = SegmentationMetrics(NUM_CLASSES)
        best_train_loss = float('inf')
        patience_counter = 0
        initial_lr = LEARNING_RATE

        # full training loader (no held-out validation)
        full_train_dataset = SegmentationDataset(TRAIN_IMG_PATH, TRAIN_MASK_PATH, train_images, train_masks, train_transform, label_mapping)
        full_train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

        for epoch in range(MAX_EPOCHS):
            current_lr = optimizer.param_groups[0]['lr']
            train_loss, train_metrics_dict = train_epoch(
                model, full_train_loader, criterion, optimizer, device, train_metrics,
                epoch=epoch+1, max_epochs=MAX_EPOCHS, lr=current_lr, phase='FullTrain')
            scheduler.step()
            
            # Store metrics for full retrain
            training_history[model_name]['train_loss'].append(train_loss)
            for m_type in ['mean', 'weighted', 'micro']:
                training_history[model_name][f'train_f1_{m_type}'].append(train_metrics_dict[f'f1_{m_type}'])
                training_history[model_name][f'train_precision_{m_type}'].append(train_metrics_dict[f'precision_{m_type}'])
                training_history[model_name][f'train_recall_{m_type}'].append(train_metrics_dict[f'recall_{m_type}'])
                training_history[model_name][f'train_iou_{m_type}'].append(train_metrics_dict[f'iou_{m_type}'])
            current_lr = optimizer.param_groups[0]['lr']
            # Save when training loss improves
            if train_loss < best_train_loss - MIN_DELTA:
                best_train_loss = train_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")
            else:
                patience_counter += 1
                effective_patience = int(PATIENCE * LR_PATIENCE_SCALE) if current_lr <= initial_lr * LR_PATIENCE_THRESHOLD else PATIENCE
                if (epoch + 1) >= MIN_EPOCHS and patience_counter >= effective_patience:
                    logger.info("Early stopping full-data retrain at epoch %d (no training-loss improvement for %d epochs)", epoch+1, effective_patience)
                    break

        # always save final model (overwrite) to ensure a final checkpoint exists
        torch.save(model.state_dict(), f"checkpoints/{model_name}_best.pth")
        logger.info('✓ %s full-data retrain complete. Checkpoint saved to %s', model_name, f'checkpoints/{model_name}_best.pth')
        model.cpu()
        del model, optimizer, scheduler, criterion, train_metrics
        torch.cuda.empty_cache()

# Save training history for evaluation
os.makedirs('outputs', exist_ok=True)
np.save('outputs/training_history.npy', training_history)
logger.info('✓ Training complete. History saved.')

# Restore original environment for any variables we changed above
for k, v in old_env.items():
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v
