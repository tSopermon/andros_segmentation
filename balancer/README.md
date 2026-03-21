# Balancer

This folder contains a small utility for splitting a segmentation dataset into `train`, `val`, and `test` subsets while trying to keep class-pixel distributions balanced across splits.

## Files

- `balance_dataset.py` - main script that scans masks, computes per-class pixel counts, chooses a balanced split, and copies files into a new dataset layout.
- `balancer_config.yaml` - configuration file for source path, output path, and split ratios.

## Expected Input

The source dataset should contain matching image and mask files, typically in folders like:

- `Image/`
- `Mask/`

The script reads masks, builds a class-pixel profile for each image, and then assigns each sample to one of the three splits.

## Output Layout

The generated dataset is written as:

- `train/Image/`
- `train/Mask/`
- `val/Image/`
- `val/Mask/`
- `test/Image/`
- `test/Mask/`

## Configuration

Edit `balancer_config.yaml` before running:

- `SOURCE_PATH` - path to the raw dataset.
- `OUTPUT_PATH` - path where the balanced split will be copied.
- `IMAGE_SUBDIR` - image folder name inside `SOURCE_PATH`.
- `MASK_SUBDIR` - mask folder name inside `SOURCE_PATH`.
- `SPLIT_RATIOS` - target ratios for `train`, `val`, and `test`.

## Run

From the project root:

```bash
python balancer/balance_dataset.py
```

## Notes

- Files are copied, not moved.
- The balancing strategy is greedy and aims to reduce class imbalance between splits.
- The resulting dataset can be used with the main training pipeline by pointing `DATASET_PATH` to `OUTPUT_PATH` and setting `PRE_SPLIT_DATASET: true`.
