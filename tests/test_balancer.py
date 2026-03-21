import os
import yaml
import numpy as np
import cv2
import pytest
from pathlib import Path

from balancer.balance_dataset import (
    load_config,
    get_unique_classes,
    get_pixel_counts,
    make_dirs,
    balance_split
)

@pytest.fixture
def mock_dataset(tmp_path):
    """Fixture to create a mock dataset with a few dummy masks."""
    mask_dir = tmp_path / "Mask"
    mask_dir.mkdir()
    
    # Mask 1: Classes 0 and 1. 10x10 image = 100 pixels.
    # 25 pixels of class 1, 75 pixels of class 0
    mask1 = np.zeros((10, 10), dtype=np.uint8)
    mask1[0:5, 0:5] = 1
    cv2.imwrite(str(mask_dir / "mask1.png"), mask1)
    
    # Mask 2: Classes 0 and 2. 
    # 25 pixels of class 2, 75 pixels of class 0
    mask2 = np.zeros((10, 10), dtype=np.uint8)
    mask2[2:7, 2:7] = 2
    cv2.imwrite(str(mask_dir / "mask2.png"), mask2)
    
    # Mask 3: Classes 0, 1, and 2.
    mask3 = np.zeros((10, 10), dtype=np.uint8)
    mask3[0:2, 0:2] = 1
    mask3[8:10, 8:10] = 2
    cv2.imwrite(str(mask_dir / "mask3.png"), mask3)

    mask_files = ["mask1.png", "mask2.png", "mask3.png"]
    return mask_dir, mask_files


def test_load_config(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    config_data = {
        "SOURCE_PATH": "/mock/source",
        "OUTPUT_PATH": "/mock/output",
        "SPLIT_RATIOS": {"train": 0.7, "val": 0.15, "test": 0.15}
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    loaded = load_config(config_path)
    assert loaded["SOURCE_PATH"] == "/mock/source"
    assert loaded["SPLIT_RATIOS"]["val"] == 0.15


def test_make_dirs(tmp_path):
    base_path = tmp_path / "dataset_out"
    make_dirs(base_path, ["Image", "Mask"])
    
    for split in ["train", "val", "test"]:
        assert (base_path / split / "Image").is_dir()
        assert (base_path / split / "Mask").is_dir()


def test_get_unique_classes(mock_dataset):
    mask_dir, mask_files = mock_dataset
    
    files_map = {}
    for mf in mask_files:
        files_map[mf] = {'mask': str(mask_dir / mf), 'image': 'dummy.jpg'}
        
    classes = get_unique_classes(files_map)
    assert classes == [0, 1, 2]


def test_get_pixel_counts(mock_dataset):
    mask_dir, mask_files = mock_dataset
    label_mapping = {0: 0, 1: 1, 2: 2}
    
    files_map = {}
    for mf in mask_files:
        files_map[mf] = {'mask': str(mask_dir / mf), 'image': 'dummy.jpg'}
        
    cache, total_counts = get_pixel_counts(files_map, label_mapping)
    
    assert len(cache) == 3
    assert "mask1.png" in cache
    assert "mask2.png" in cache
    assert "mask3.png" in cache
    
    # Mask 1: 75 of 0, 25 of 1, 0 of 2
    assert np.array_equal(cache["mask1.png"], np.array([75, 25, 0]))
    
    # Mask 2: 75 of 0, 0 of 1, 25 of 2
    assert np.array_equal(cache["mask2.png"], np.array([75, 0, 25]))
    
    # Mask 3: 92 of 0, 4 of 1, 4 of 2
    assert np.array_equal(cache["mask3.png"], np.array([92, 4, 4]))
    
    # Total counts
    assert np.array_equal(total_counts, np.array([242, 29, 29]))


def test_balance_split():
    # Construct a synthetic cache for 10 identical files to test the split ratio logic easily
    cache = {}
    total_counts = np.zeros(2, dtype=np.int64)
    num_files = 10
    
    for i in range(num_files):
        filename = f"mask_{i}.png"
        counts = np.array([90, 10], dtype=np.int64)
        cache[filename] = counts
        total_counts += counts
        
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    split_files, split_counts = balance_split(cache, total_counts, split_ratios)
    
    # Check if the length of splits broadly aligns with the requested ratios
    assert len(split_files["train"]) == 8
    assert len(split_files["val"]) == 1
    assert len(split_files["test"]) == 1
    
    # Check if it balanced the counts correctly
    assert np.array_equal(split_counts["train"], np.array([720, 80]))
    assert np.array_equal(split_counts["val"], np.array([90, 10]))
    assert np.array_equal(split_counts["test"], np.array([90, 10]))
