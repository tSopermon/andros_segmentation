import numpy as np
import torch
from utils.dataset import SegmentationDataset
import tempfile
import os
import cv2
from pathlib import Path
import pytest

def create_dummy_image(path, shape=(8, 8, 3)):
    img = np.random.randint(0, 255, shape, dtype=np.uint8)
    cv2.imwrite(str(path), img)

def create_dummy_mask(path, shape=(8, 8), num_classes=2):
    mask = np.random.randint(0, num_classes, shape, dtype=np.uint8)
    cv2.imwrite(str(path), mask)

def test_segmentation_dataset(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    image_files = []
    mask_files = []
    for i in range(3):
        img_path = img_dir / f"img_{i}.png"
        mask_path = mask_dir / f"mask_{i}.png"
        create_dummy_image(img_path)
        create_dummy_mask(mask_path, num_classes=2)
        image_files.append(f"img_{i}.png")
        mask_files.append(f"mask_{i}.png")
    label_mapping = {0: 0, 1: 1}
    dataset = SegmentationDataset(img_dir, mask_dir, image_files, mask_files, transform=None, label_mapping=label_mapping)
    assert len(dataset) == 3
    img, mask = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape[0] == 3
    assert mask.ndim == 2

def test_dataset_missing_image(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    # Only create mask, no image
    mask_path = mask_dir / "sample_mask.png"
    create_dummy_mask(mask_path)
    image_files = []
    mask_files = ["sample_mask.png"]
    label_mapping = {0: 0, 1: 1}
    # Should raise error due to missing image
    with pytest.raises((FileNotFoundError, IndexError, ValueError)):
        SegmentationDataset(img_dir, mask_dir, image_files, mask_files, label_mapping=label_mapping)

def test_dataset_inconsistent_mask_shape(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    img_path = img_dir / "sample_img.png"
    mask_path = mask_dir / "sample_mask.png"
    create_dummy_image(img_path, shape=(8, 8, 3))
    create_dummy_mask(mask_path, shape=(16, 16))  # Different shape
    image_files = ["sample_img.png"]
    mask_files = ["sample_mask.png"]
    label_mapping = {0: 0, 1: 1}
    dataset = SegmentationDataset(img_dir, mask_dir, image_files, mask_files, label_mapping=label_mapping)
    img, mask = dataset[0]
    assert img.shape[-2:] != mask.shape[-2:], "Image and mask shapes should differ"
