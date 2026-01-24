import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, mask_files, transform=None, label_mapping=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files
        self.mask_files = mask_files
        if len(self.image_files) != len(self.mask_files):
            raise ValueError(f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) do not match.")
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        image = cv2.imread(str(self.image_dir / img_name))
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_dir / img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / mask_name), cv2.IMREAD_GRAYSCALE)
        if self.label_mapping is not None:
            mask_mapped = np.zeros_like(mask)
            for original_label, new_label in self.label_mapping.items():
                mask_mapped[mask == original_label] = new_label
            mask = mask_mapped
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        return image, mask

def count_pixels(mask_dir, mask_list, mapping):
    counts = np.zeros(len(mapping), dtype=np.int64)
    for fname in mask_list:
        m = cv2.imread(str(mask_dir / fname), cv2.IMREAD_GRAYSCALE)
        mapped = np.zeros_like(m)
        for orig, idx in mapping.items():
            mapped[m == orig] = idx
        vals, freqs = np.unique(mapped, return_counts=True)
        counts[vals] += freqs
    return counts

def get_pixel_counts_cache(mask_dir, mask_list, mapping):
    cache = {}
    for fname in mask_list:
        m = cv2.imread(str(mask_dir / fname), cv2.IMREAD_GRAYSCALE)
        mapped = np.zeros_like(m)
        for orig, idx in mapping.items():
            mapped[m == orig] = idx
        vals, freqs = np.unique(mapped, return_counts=True)
        counts = np.zeros(len(mapping), dtype=np.int64)
        counts[vals] = freqs
        cache[fname] = counts
    return cache