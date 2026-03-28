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

class DualStreamDataset(Dataset):
    def __init__(self, 
                 labeled_image_dir, labeled_mask_dir, labeled_image_files, labeled_mask_files,
                 unlabeled_image_dir, unlabeled_image_files,
                 transform=None, unl_transform_weak=None, unl_transform_strong=None, label_mapping=None):
        self.labeled_image_dir = labeled_image_dir
        self.labeled_mask_dir = labeled_mask_dir
        self.labeled_image_files = labeled_image_files
        self.labeled_mask_files = labeled_mask_files
        
        self.unlabeled_image_dir = unlabeled_image_dir
        self.unlabeled_image_files = unlabeled_image_files
        
        self.transform = transform
        self.unl_transform_weak = unl_transform_weak
        self.unl_transform_strong = unl_transform_strong
        self.label_mapping = label_mapping
        
        self.labeled_len = len(self.labeled_image_files)
        # If no unlabeled files provided (e.g. evaluating), we set it to 1 to avoid /0 errors, 
        # but it shouldn't be called without unlabeled data anway.
        self.unlabeled_len = max(1, len(self.unlabeled_image_files))

    def __len__(self):
        return self.labeled_len

    def __getitem__(self, idx):
        # --- Labeled Stream ---
        l_img_name = self.labeled_image_files[idx]
        l_mask_name = self.labeled_mask_files[idx]
        l_image = cv2.imread(str(self.labeled_image_dir / l_img_name))
        l_image = cv2.cvtColor(l_image, cv2.COLOR_BGR2RGB)
        l_mask = cv2.imread(str(self.labeled_mask_dir / l_mask_name), cv2.IMREAD_GRAYSCALE)
        
        if self.label_mapping is not None:
            mask_mapped = np.zeros_like(l_mask)
            for orig, new_idx in self.label_mapping.items():
                mask_mapped[l_mask == orig] = new_idx
            l_mask = mask_mapped
            
        if self.transform:
            augmented = self.transform(image=l_image, mask=l_mask)
            l_image = augmented['image']
            l_mask = augmented['mask']
        else:
            l_image = torch.from_numpy(l_image).permute(2, 0, 1).float()
            l_mask = torch.from_numpy(l_mask).long()

        if not isinstance(l_mask, torch.Tensor):
            l_mask = torch.from_numpy(l_mask).long()
        else:
            l_mask = l_mask.long()
            
        # --- Unlabeled Stream ---
        # Randomly sample an unlabeled image
        if len(self.unlabeled_image_files) == 0:
            # Fallback for unexpected empty unlabeled array: return the labeled image as dummy
            u_image_clean = l_image.clone() if isinstance(l_image, torch.Tensor) else l_image.copy()
            u_image_aug = l_image.clone() if isinstance(l_image, torch.Tensor) else l_image.copy()
        else:
            u_idx = np.random.randint(0, self.unlabeled_len)
            u_img_name = self.unlabeled_image_files[u_idx]
            u_image = cv2.imread(str(self.unlabeled_image_dir / u_img_name))
            u_image = cv2.cvtColor(u_image, cv2.COLOR_BGR2RGB)
            
            u_image_clean = u_image.copy()
            u_image_aug = u_image.copy()
            
            if self.unl_transform_weak:
                augmented_weak = self.unl_transform_weak(image=u_image_clean)
                u_image_clean = augmented_weak['image']
            else:
                u_image_clean = torch.from_numpy(u_image_clean).permute(2, 0, 1).float()
                
            if self.unl_transform_strong:
                augmented_strong = self.unl_transform_strong(image=u_image_aug)
                u_image_aug = augmented_strong['image']
            else:
                u_image_aug = torch.from_numpy(u_image_aug).permute(2, 0, 1).float()

        return l_image, l_mask, u_image_clean, u_image_aug

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

class PretrainDataset(Dataset):
    def __init__(self, image_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = cv2.imread(str(self.image_dir / img_name))
        if image is None:
            raise ValueError(f"Failed to load image: {self.image_dir / img_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            
        return image