"""
Pre-training script for self-predictive learning (MAE-style) on segmentation models.

Loads configuration from config/config.yaml, pre-trains the model to reconstruct
masked patches, and saves checkpoints to be used for downstream fine-tuning.

Usage:
    python pretrain.py --config config/config.yaml

Outputs:
    - Model checkpoints in checkpoints/{model_name}_pretrained.pth
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import torch
import torch.nn.functional as F
from pathlib import Path
from utils.config_loader import load_config
from utils.model_selection import get_selected_model_names
from utils.dataset import PretrainDataset
from utils.transforms import get_train_transform
from training.masking_utils import generate_random_mask, generate_object_centric_mask
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import random
import logging
import datetime
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML')
args = parser.parse_args()

# Load config
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
config = load_config(args.config)
DATASET_PATH = Path(config['DATASET_PATH'])
TRAIN_IMG_PATH = DATASET_PATH / 'train' / ('Image' if (DATASET_PATH / 'train' / 'Image').exists() else 'image')

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', config['BATCH_SIZE']))
PRETRAIN_EPOCHS = int(config.get('PRETRAIN_EPOCHS', 100))
LEARNING_RATE = config['LEARNING_RATE']
IMAGE_SIZE = config['IMAGE_SIZE']
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', config['NUM_WORKERS']))
USE_AUGMENTATION = config['USE_AUGMENTATION']
MASK_RATIO = float(config.get('MASK_RATIO', 0.75))
PATCH_SIZE = int(config.get('PATCH_SIZE', 16))
OBJECT_CENTRIC_EPOCH = int(config.get('OBJECT_CENTRIC_EPOCH', 50))
BACKBONE = config.get('BACKBONE', 'resnet101')
ENCODER_WEIGHTS = config.get('ENCODER_WEIGHTS', 'imagenet')

# Device and Logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("=" * 70)
logger.info("PRE-TRAINING CONFIGURATION")
logger.info("=" * 70)
logger.info("Config file          : %s", args.config)
logger.info("Device               : %s", device)
logger.info("Max Epochs           : %d", PRETRAIN_EPOCHS)
logger.info("Batch Size           : %d", BATCH_SIZE)
logger.info("Mask Ratio           : %.2f", MASK_RATIO)
logger.info("Patch Size           : %d", PATCH_SIZE)
logger.info("Object-Centric Epoch : %d", OBJECT_CENTRIC_EPOCH)
logger.info("=" * 70)

# Datasets
image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
train_images = sorted([f for f in os.listdir(TRAIN_IMG_PATH) if f.lower().endswith(image_extensions)])

# Because albumentations handles `image=image` just fine, we can reuse train_transform
train_transform = get_train_transform(IMAGE_SIZE, USE_AUGMENTATION)
pretrain_dataset = PretrainDataset(TRAIN_IMG_PATH, train_images, transform=train_transform)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

# If running originals or all, enable original registrations in model_zoo via env vars.
old_env = {k: os.environ.get(k) for k in ('USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL', 'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
os.environ['USE_UNET_ORIGINAL'] = str(config.get('USE_UNET_ORIGINAL', False)).lower()
os.environ['USE_DEEPLABV1_ORIGINAL'] = str(config.get('USE_DEEPLABV1_ORIGINAL', False)).lower()
os.environ['USE_DEEPLABV2_ORIGINAL'] = str(config.get('USE_DEEPLABV2_ORIGINAL', False)).lower()
os.environ['USE_DEEPLABV3_ORIGINAL'] = str(config.get('USE_DEEPLABV3_ORIGINAL', False)).lower()
os.environ['USE_MAXVIT_UNET'] = str(config.get('USE_MAXVIT_UNET', False)).lower()

MODEL_NAMES = get_selected_model_names(config)

from models.model_zoo import get_models

os.makedirs('checkpoints', exist_ok=True)

for model_name in MODEL_NAMES:
    logger.info("\n%s\nPre-training %s\n%s", '='*70, model_name, '='*70)
    
    # We create the model with 3 output classes to reconstruct RGB images
    try:
        models_dict = get_models(3, backbone=BACKBONE, encoder_weights=ENCODER_WEIGHTS, specific_model=model_name)
        if model_name not in models_dict:
            logger.warning("Model %s not found in model_zoo. Skipping.", model_name)
            continue
        model = models_dict[model_name].to(device)
    except Exception as e:
        logger.error("Failed to instantiate model %s: %s. Skipping.", model_name, e)
        continue

    # Note: For normalization consistency, standard images in pipeline are 0-255 scaled or mean/std subtracted?
    # Actually, get_train_transform includes `ToTensorV2`, which normally divides by 255.
    
    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == 'cuda' else None
    
    best_loss = float('inf')
    
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        
        desc = f"Epoch {epoch}/{PRETRAIN_EPOCHS}"
        pbar = tqdm(pretrain_loader, desc=desc, dynamic_ncols=True)
        
        for images in pbar:
            images = images.to(device)
            # Normalization check: if max > 1, the transforms didn't normalize to 0-1.
            # Usually ToTensorV2 converts to 0-1, but sometimes it doesn't without Normalize.
            # Pretraining is easier to interpret on 0-1 scale.
            if images.max() > 2.0:
                images = images / 255.0
            
            # Generate mask
            if epoch >= OBJECT_CENTRIC_EPOCH:
                mask = generate_object_centric_mask(images, MASK_RATIO, PATCH_SIZE)
            else:
                mask = generate_random_mask(images, MASK_RATIO, PATCH_SIZE)
            
            mask_float = mask.float() # 1 means masked (hidden), 0 means visible
            
            # Apply mask to images: replace masked areas with 0 (or a mean token).
            # masked_images will have 0 where mask_float is 1.
            masked_images = images * (1.0 - mask_float)
            
            optimizer.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(masked_images)
                    
                    # Compute MSE loss only on the masked patches
                    # loss tensor shape: (B, 3, H, W)
                    raw_loss = criterion(outputs, images)
                    
                    # Apply mask (mask_float has shape (B, 1, H, W))
                    # Average over the number of masked pixels to get scalar loss
                    mask_expanded = mask_float.expand_as(raw_loss)
                    masked_loss = (raw_loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
                    
                scaler.scale(masked_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(masked_images)
                raw_loss = criterion(outputs, images)
                mask_expanded = mask_float.expand_as(raw_loss)
                masked_loss = (raw_loss * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
                
                masked_loss.backward()
                optimizer.step()
                
            total_loss += masked_loss.item()
            pbar.set_postfix({'loss': f'{masked_loss.item():.4f}'})
            
        avg_loss = total_loss / len(pretrain_loader)
        logger.info("Epoch %d/%d - Avg Masked MSE Loss: %.4f", epoch, PRETRAIN_EPOCHS, avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = f"checkpoints/{model_name}_pretrained.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved new best pre-trained model to %s", ckpt_path)

    logger.info("✓ %s pre-training completed. Best Masked MSE: %.4f", model_name, best_loss)
    model.cpu()
    del model, optimizer, criterion
    torch.cuda.empty_cache()

logger.info("✓ All pre-training processes finished.")

# Restore original environment for any variables we changed above
for k, v in old_env.items():
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v
