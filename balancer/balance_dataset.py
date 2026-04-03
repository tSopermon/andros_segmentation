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

def collect_files(src_dir, img_subdir, mask_subdir, img_suffix="", mask_suffix=""):
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
            m_base = os.path.splitext(m_file)[0]
            
            # Strip mask_suffix if present
            if mask_suffix and m_base.endswith(mask_suffix):
                base_name = m_base[:-len(mask_suffix)]
            else:
                base_name = m_base
                
            # Form image search pattern with potential suffix
            img_search_base = base_name + img_suffix
            
            # Try to find matching image
            matches = glob.glob(os.path.join(i_dir, img_search_base + '.*'))
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
    Simulated Annealing/Swap-based algorithm to distribute images across splits.
    This guarantees exact file counts while minimizing the distribution error.
    """
    splits = ['train', 'val', 'test']
    ratios = [split_ratios['train'], split_ratios['val'], split_ratios['test']]
    target_probs = total_counts / total_counts.sum()
    
    num_files = len(cache)
    all_files = list(cache.keys())
    
    # Calculate exact integer target sizes for each split
    target_files = {s: int(num_files * r) for s, r in zip(splits, ratios)}
    
    # Distribute the remainder (due to int rounding) to the largest target
    remainder = num_files - sum(target_files.values())
    if remainder > 0:
        largest_split = max(target_files, key=target_files.get)
        target_files[largest_split] += remainder
        
    # Initial random assignment to meet exact sizes
    np.random.seed(42)  # For reproducibility, or remove for true random
    np.random.shuffle(all_files)
    
    split_files = {s: [] for s in splits}
    start = 0
    for s in splits:
        end = start + target_files[s]
        split_files[s] = all_files[start:end]
        start = end
        
    def get_split_counts(s_files):
        counts = np.zeros_like(total_counts)
        for f in s_files:
            counts += cache[f]
        return counts
        
    def evaluate_error(s_files):
        counts = get_split_counts(s_files)
        total = np.sum(counts)
        if total == 0:
            return float('inf')
        probs = counts / total
        # Weight differences inversely to target_probs to prioritize rare classes
        diffs = np.abs(probs - target_probs)
        weighted_diffs = diffs / (target_probs + 1e-4) 
        return np.mean(weighted_diffs)
        
    def evaluate_total_error(sf_dict):
        return sum(evaluate_error(sf_dict[s]) for s in splits)

    # Hill-climbing / swap optimization
    current_error = evaluate_total_error(split_files)
    logger.info(f"Initial random split error: {current_error:.4f}")
    
    # Try 10,000 random swaps to see if they improve the distribution
    patience = 0
    for iteration in range(10000):
        # Pick two different splits at random
        s1, s2 = np.random.choice(splits, 2, replace=False)
        
        # If one is empty, skip
        if not split_files[s1] or not split_files[s2]:
            continue
            
        # Pick one file from each to swap
        idx1 = np.random.randint(len(split_files[s1]))
        idx2 = np.random.randint(len(split_files[s2]))
        
        f1 = split_files[s1][idx1]
        f2 = split_files[s2][idx2]
        
        # Simulate swap
        split_files[s1][idx1] = f2
        split_files[s2][idx2] = f1
        
        new_error = evaluate_total_error(split_files)
        
        if new_error < current_error:
            # Keep swap
            current_error = new_error
            patience = 0
        else:
            # Revert swap
            split_files[s1][idx1] = f1
            split_files[s2][idx2] = f2
            patience += 1
            
        if patience > 2000:  # If we haven't found an improvement in 2000 tries, we've likely converged
            break
            
    logger.info(f"Final optimized split error: {current_error:.4f}")
    
    # Final counts for reporting
    split_counts = {s: get_split_counts(split_files[s]) for s in splits}
        
    return split_files, split_counts

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'balancer_config.yaml')
    config = load_config(config_path)
    
    src_dir = Path(config['SOURCE_PATH'])
    out_dir = Path(config['OUTPUT_PATH'])
    img_subdir = config.get('IMAGE_SUBDIR', 'Image')
    mask_subdir = config.get('MASK_SUBDIR', 'Mask')
    
    img_suffix = config.get('IMAGE_SUFFIX', "")
    mask_suffix = config.get('MASK_SUFFIX', "")
    
    files_map = collect_files(src_dir, img_subdir, mask_subdir, img_suffix, mask_suffix)
    
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
