# Balancer

This folder contains a small utility for splitting a segmentation dataset into `train`, `val`, and `test` subsets while trying to keep class-pixel distributions balanced across splits.

## Files

- `balance_dataset.py` - main script that scans masks, computes per-class pixel counts, chooses a balanced split, and copies files into a new dataset layout.
- `balancer_config.yaml` - configuration file for source path, output path, and split ratios.

## Expected Input

The source dataset should contain matching image and mask files, typically in folders like:

- `Image/`
- `Mask/` (Optional)

The script reads masks, builds a class-pixel profile for each image, and then assigns each sample to one of the three splits. If the dataset does not have masks (i.e., `MASK_SUBDIR` is empty or null in the config), the script will randomly split the images according to the given ratios.

## Output Layout

The generated dataset is written as:

- `train/Image/`
- `train/Mask/` (If masks were provided)
- `val/Image/`
- `val/Mask/` (If masks were provided)
- `test/Image/`
- `test/Mask/` (If masks were provided)

## Configuration

Edit `balancer_config.yaml` before running:

- `SOURCE_PATH` - path to the raw dataset.
- `OUTPUT_PATH` - path where the balanced split will be copied.
- `IMAGE_SUBDIR` - image folder name inside `SOURCE_PATH`.
- `MASK_SUBDIR` - mask folder name inside `SOURCE_PATH`. Set this to empty or `null` if the dataset only contains images.
- `SPLIT_RATIOS` - target ratios for `train`, `val`, and `test`.
- **Filename Matching Options (Optional)**:
  - `IMAGE_SUFFIX` / `MASK_SUFFIX` - Suffixes to ignore/match if files differ only at the very end of the base name (e.g., `_image`, `_mask`).
  - `REPLACE_MASK_STR` / `WITH_IMAGE_STR` - Use these if the image and mask files differ by a substring located in the *middle* of the filename (e.g. `..._Mask_001.tif` vs `..._Image_001.tif`).
## Run

From the project root:

```bash
python balancer/balance_dataset.py
```

## Notes

- Files are copied, not moved.
- The balancing strategy is greedy and aims to reduce class imbalance between splits. If no masks are provided, balancing is bypassed and a standard random split is performed.
- The resulting dataset can be used with the main training pipeline by pointing `DATASET_PATH` to `OUTPUT_PATH` and setting `PRE_SPLIT_DATASET: true`.
