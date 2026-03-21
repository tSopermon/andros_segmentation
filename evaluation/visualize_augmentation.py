import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.transforms import get_train_transform
from utils.config_loader import load_config

def visualize_augmentations(image_path, mask_path, config_path='config/config.yaml', num_samples=5):
    config = load_config(config_path)
    image_size = config.get('IMAGE_SIZE', 512)
    
    # Load original image and mask
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Get transform
    transform = get_train_transform(image_size, use_augmentation=True)
    
    # Create plot
    fig, axes = plt.subplots(num_samples + 1, 2, figsize=(10, 3 * (num_samples + 1)))
    
    # Plot original
    # Resize original for comparison
    resized_image = cv2.resize(image, (image_size, image_size))
    resized_mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    
    axes[0, 0].imshow(resized_image)
    axes[0, 0].set_title("Original Image (Resized)")
    axes[0, 1].imshow(resized_mask, cmap='jet')
    axes[0, 1].set_title("Original Mask (Resized)")
    
    # Plot augmented samples
    for i in range(num_samples):
        augmented = transform(image=image, mask=mask)
        aug_image = augmented['image'].permute(1, 2, 0).numpy()
        
        # Denormalize image for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        aug_image = (aug_image * std + mean).clip(0, 1)
        
        aug_mask = augmented['mask'].numpy()
        
        axes[i+1, 0].imshow(aug_image)
        axes[i+1, 0].set_title(f"Augmented Image {i+1}")
        axes[i+1, 1].imshow(aug_mask, cmap='jet')
        axes[i+1, 1].set_title(f"Augmented Mask {i+1}")
        
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    os.makedirs('outputs/debug', exist_ok=True)
    save_path = 'outputs/debug/augmentation_validation.png'
    plt.savefig(save_path)
    print(f"Augmentation visualization saved to {save_path}")

if __name__ == "__main__":
    # Load config to get paths
    config = load_config('config/config.yaml')
    dataset_path = Path(config['DATASET_PATH'])
    train_img_dir = dataset_path / 'train' / ('Image' if (dataset_path / 'train' / 'Image').exists() else 'image')
    train_mask_dir = dataset_path / 'train' / ('Mask' if (dataset_path / 'train' / 'Mask').exists() else 'mask')
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    images = sorted([f for f in os.listdir(train_img_dir) if f.lower().endswith(image_extensions)])
    masks = sorted([f for f in os.listdir(train_mask_dir) if f.lower().endswith(image_extensions)])
    
    if images and masks:
        image_path = train_img_dir / images[0]
        mask_path = train_mask_dir / masks[0]
        print(f"Using sample: Image={image_path.name}, Mask={mask_path.name}")
        visualize_augmentations(image_path, mask_path)
    else:
        print("No samples found in dataset.")
