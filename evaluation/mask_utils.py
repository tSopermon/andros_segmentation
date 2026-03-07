"""
Mask utilities for segmentation evaluation.

This module provides functions for mask encoding, decoding, and visualization, including helpers for test dataset creation and mask saving.
"""
import os
import numpy as np
from pathlib import Path
from utils.dataset import SegmentationDataset
from utils.transforms import get_val_transform
from PIL import Image

def get_test_dataset(config):
    """
    Create the test dataset using the same transform as evaluation.

    Args:
        config (dict): Configuration dictionary loaded from YAML.

    Returns:
        SegmentationDataset: Dataset for test images and masks.
    """
    dataset_path = config['DATASET_PATH']
    test_img_dir = Path(dataset_path) / 'test' / ('Image' if (Path(dataset_path) / 'test' / 'Image').exists() else 'image')
    test_mask_dir = Path(dataset_path) / 'test' / ('Mask' if (Path(dataset_path) / 'test' / 'Mask').exists() else 'mask')
    image_files = sorted(os.listdir(test_img_dir))
    mask_files = sorted(os.listdir(test_mask_dir))
    image_size = config['IMAGE_SIZE']
    test_transform = get_val_transform(image_size)
    return SegmentationDataset(test_img_dir, test_mask_dir, image_files, mask_files, transform=test_transform)

def save_mask(mask, save_path, num_classes):
    """
    Save a predicted mask as a PNG image with a color palette for visualization.

    Args:
        mask (np.ndarray): 2D array of class indices with shape (H, W).
        save_path (str): Path to save the PNG mask.
        num_classes (int): Number of classes for palette generation.

    Returns:
        None
    """
    mask_img = Image.fromarray(mask.astype(np.uint8), mode='P')
    # Use specific, human-chosen colors for the classes (index order must match class_names)
    # Order: 0: Water, 1: Woodland, 2: Arable land, 3: Frygana,
    # 4: Other, 5: Artificial land, 6: Perm. Cult, 7: Bareland
    custom_colors = [
        (0, 0, 255),      # Water
        (60, 16, 152),    # Woodland
        (132, 41, 246),   # Arable land
        (0, 255, 0),      # Frygana
        (155, 155, 155),  # Other
        (226, 169, 41),   # Artificial land
        (255, 255, 0),    # Perm. Cult
        (255, 255, 255),  # Bareland
    ]
    # Build full 256-color palette (768 values) and inject custom colors for the first N classes
    full_palette = [0] * (256 * 3)
    for i, (r, g, b) in enumerate(custom_colors[:num_classes]):
        full_palette[3 * i: 3 * i + 3] = [r, g, b]
    mask_img.putpalette(full_palette)
    mask_img.save(save_path)
