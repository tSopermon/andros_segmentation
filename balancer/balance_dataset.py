import os
import shutil
import yaml
import glob
import numpy as np
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import datetime

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def collect_files(src_dir, img_subdir, mask_subdir):
    """
    Search for dataset files. Handles cases where files are:
    1. Flat in src_dir / img_subdir
    2. Split in src_dir / 'train' (or 'val', 'test') / img_subdir
    Returns a dictionary mapping base_filename -> {'image': img_path, 'mask': mask_path}
    """
    files_map = {}
    
    # Define possible root directories to search in
    search_dirs = [src_dir]
    for split_dir in ['train', 'val', 'test']:
        if (src_dir / split_dir).exists():
            search_dirs.append(src_dir / split_dir)
            
    for d in search_dirs:
        m_dir = d / mask_subdir
        i_dir = d / img_subdir
        
        if not m_dir.exists() or not i_dir.exists():
            continue
            
        m_files = [f for f in os.listdir(m_dir) if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.jpeg'))]
        for m_file in m_files:
            base_name = os.path.splitext(m_file)[0]
            # Try to find matching image
            matches = glob.glob(os.path.join(i_dir, base_name + '.*'))
            if matches:
                # Store full paths
                files_map[m_file] = {
                    'mask': str(m_dir / m_file),
                    'image': matches[0]
                }
                
    return files_map

def get_unique_classes(files_map):
    """Scan masks to find all unique pixel values representing classes."""
    logger.info("Scanning dataset for unique classes...")
    unique_classes = set()
    for m_file in tqdm(files_map.keys(), desc="Scanning unique classes"):
        mask = cv2.imread(files_map[m_file]['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            unique_classes.update(np.unique(mask))
    logger.info(f"Discovered classes: {sorted(list(unique_classes))}")
    return sorted(list(unique_classes))

def get_pixel_counts(files_map, label_mapping):
    """Compute per-image class counts using the provided label mapping."""
    cache = {}
    total_counts = np.zeros(len(label_mapping), dtype=np.int64)
    logger.info("Computing pixel counts for all masks...")
    for m_file in tqdm(files_map.keys(), desc="Computing pixel counts"):
        mask = cv2.imread(files_map[m_file]['mask'], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.warning(f"Could not read mask: {files_map[m_file]['mask']}")
            continue
            
        # Map labels
        mapped = np.zeros_like(mask)
        for orig, idx in label_mapping.items():
            mapped[mask == orig] = idx
            
        vals, freqs = np.unique(mapped, return_counts=True)
        counts = np.zeros(len(label_mapping), dtype=np.int64)
        counts[vals] = freqs
        
        cache[m_file] = counts
        total_counts += counts
        
    return cache, total_counts

def make_dirs(base_path, subdirs):
    for split in ['train', 'val', 'test']:
        for subdir in subdirs:
            os.makedirs(os.path.join(base_path, split, subdir), exist_ok=True)

def balance_split(cache, total_counts, split_ratios):
    """
    Greedy algorithm to distribute images across splits keeping class distributions similar.
    We iterate over images and place each one in the split where its inclusion 
    minimizes the divergence from the target split ratios and class distribution.
    """
    splits = ['train', 'val', 'test']
    ratios = [split_ratios['train'], split_ratios['val'], split_ratios['test']]
    
    # Initialize trackers
    split_counts = {s: np.zeros_like(total_counts) for s in splits}
    target_probs = total_counts / total_counts.sum()
    
    # Target capacity (in terms of total dataset files)
    num_files = len(cache)
    split_files = {s: [] for s in splits}
    target_files = {s: int(num_files * r) for s, r in zip(splits, ratios)}
    
    # Sort files by rarity of classes or just total 'foreground' pixels to place large ones first
    # We will score by frequency of rarest class present.
    # Inverse of global distribution:
    class_weights = 1.0 / (target_probs + 1e-6)
    
    def score_image(m_file):
        cnts = cache[m_file]
        return np.sum(cnts * class_weights)
        
    sorted_files = sorted(cache.keys(), key=score_image, reverse=True)
    
    for m_file in sorted_files:
        counts = cache[m_file]
        best_split = None
        best_score = float('inf')
        
        for s in splits:
            # If a split already has more files than its exact share, heavily penalize 
            # to prevent it from absorbing all smaller images in corner cases
            if len(split_files[s]) >= target_files[s] + max(1, num_files * 0.05):
                continue
                
            simulated_counts = split_counts[s] + counts
            simulated_total = np.sum(simulated_counts)
            
            if simulated_total == 0:
                score = 0
            else:
                simulated_probs = simulated_counts / simulated_total
                # We want the simulated probability to match target probabilities
                # Calculate MSE
                mse = np.mean((simulated_probs - target_probs) ** 2)
                
                # Heavy penalty if we exceed the target number of files for a split
                file_deficit = target_files[s] - len(split_files[s])
                
                if file_deficit <= 0:
                    size_penalty = 1000.0  # Big penalty for exceeding target sizes
                else:
                    current_ratio = (len(split_files[s]) + 1) / num_files
                    target_ratio = target_files[s] / num_files
                    size_penalty = abs(current_ratio - target_ratio) * 10
                
                score = mse + size_penalty
                
            if score < best_score:
                best_score = score
                best_split = s
                
        # If tolerance blocked all, fallback to the one with least files relative to target
        if best_split is None:
            best_split = min(splits, key=lambda x: len(split_files[x]) / (target_files[x] + 1))
            
        split_counts[best_split] += counts
        split_files[best_split].append(m_file)
        
    return split_files, split_counts

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'balancer_config.yaml')
    config = load_config(config_path)
    
    src_dir = Path(config['SOURCE_PATH'])
    out_dir = Path(config['OUTPUT_PATH'])
    img_subdir = config.get('IMAGE_SUBDIR', 'Image')
    mask_subdir = config.get('MASK_SUBDIR', 'Mask')
    
    files_map = collect_files(src_dir, img_subdir, mask_subdir)
    
    if not files_map:
        logger.error(f"No mask/image pairs found in {src_dir} (searching flat and train/val/test splits)")
        return

    logger.info(f"Found {len(files_map)} image-mask pairs.")
    
    # 1. Discover classes
    classes = get_unique_classes(files_map)
    label_mapping = {val: i for i, val in enumerate(classes)}
    logger.info(f"Label Mapping: {label_mapping}")
    
    # 2. Get counts
    cache, total_counts = get_pixel_counts(files_map, label_mapping)
    total_pixels = np.sum(total_counts)
    
    logger.info("Global Class Distribution:")
    for cls_val, cls_idx in label_mapping.items():
        pct = (total_counts[cls_idx] / total_pixels) * 100
        logger.info(f"  Class {cls_val} (idx {cls_idx}): {pct:.2f}%")
        
    # 3. Balance splits
    split_files, split_counts = balance_split(cache, total_counts, config['SPLIT_RATIOS'])
    
    for s in ['train', 'val', 'test']:
        s_total = np.sum(split_counts[s])
        logger.info(f"\n{s.upper()} Split: {len(split_files[s])} files")
        for cls_val, cls_idx in label_mapping.items():
            pct = (split_counts[s][cls_idx] / s_total) * 100 if s_total > 0 else 0
            logger.info(f"  Class {cls_val}: {pct:.2f}%")
            
    # 4. Copy files
    logger.info(f"\nCopying files to {out_dir}...")
    make_dirs(out_dir, [img_subdir, mask_subdir])
    
    for s in ['train', 'val', 'test']:
        logger.info(f"Copying {s} split...")
        s_img_dir = out_dir / s / img_subdir
        s_mask_dir = out_dir / s / mask_subdir
        for m_file in tqdm(split_files[s], desc=f"Copying {s}"):
            src_mask_file = files_map[m_file]['mask']
            src_image_file = files_map[m_file]['image']
            
            img_file_name = os.path.basename(src_image_file)
            
            shutil.copy2(src_image_file, s_img_dir / img_file_name)
            shutil.copy2(src_mask_file, s_mask_dir / m_file)
                
    logger.info("Dataset completely balanced and split!")

if __name__ == '__main__':
    main()
