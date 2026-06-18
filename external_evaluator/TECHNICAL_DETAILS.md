# Technical Details: Mask Evaluation Script

This document provides a comprehensive technical overview of how the `evaluate_masks.py` script operates, detailing its masking logic, RGB decoding, and the mathematical formulations used to compute segmentation metrics.

## 1. Input Processing & Decoding

### 1.1 Mask Matching
The script iterates through both the Ground Truth (`gt_dir`) and Predictions (`pred_dir`) directories. It performs an exact filename string match to ensure that `image_001.png` in the GT folder is evaluated strictly against `image_001.png` in the predictions folder. Shape mismatches between paired images are gracefully skipped with a warning.

### 1.2 RGB Visualization Decoding
Segmentation masks are often saved as "paletted" or RGB visualization images (e.g., shape `[H, W, 3]`), meaning a pixel value isn't a simple integer (like `1` or `2`) but an RGB tuple (like `(60, 16, 152)` for Woodland).

If the script detects a 3-channel image, it utilizes the `decode_rgb_mask()` function:
1. **Palette Loading:** Loads the project's official 8+ class palette mapping.
2. **Boolean Matching:** For each color in the palette, it finds all pixels matching that exact RGB tuple.
3. **Index Assignment:** Reassigns those pixels to their corresponding 1D class index (`0` through `num_classes - 1`).

## 2. Metric Accumulation (Per-Image)

Metrics are accumulated continuously across all images in the dataset to calculate true global dataset metrics, rather than calculating per-image metrics and averaging them. 

For each image pair and each class $c$:
- A boolean tensor is created for the Ground Truth where class == $c$ (`target_c`).
- A boolean tensor is created for the Prediction where class == $c$ (`pred_c`).

### 2.1 Invalid Pixel Masking (Ignore Index)
A `valid_mask` is generated to ignore unlabeled pixels or boundaries (often stored as `255` or `-1`). 
Any pixel falling outside the range `[0, num_classes - 1]` is mathematically zeroes out in both `pred_c` and `target_c` before accumulation.

### 2.2 Confusion Matrix Core Components
For each class $c$, we sum the pixels across the image:
- **True Positives (TP):** `sum(pred_c * target_c)`
- **False Positives (FP):** `sum(pred_c * (1 - target_c))`
- **False Negatives (FN):** `sum((1 - pred_c) * target_c)`
- **Intersection:** `sum(pred_c * target_c)` (Identical to TP)
- **Union:** `sum(pred_c) + sum(target_c) - Intersection`

These raw counts are accumulated globally across the entire test set.

## 3. Evaluation Metrics (Per-Class)

Once all images are processed, the script computes the standard segmentation metrics using the globally accumulated TP, FP, FN, and Union.

1. **Precision:** $\frac{TP}{TP + FP}$
   *(How many of the predicted pixels for class $c$ were actually correct?)*
2. **Recall:** $\frac{TP}{TP + FN}$
   *(How many of the true pixels for class $c$ did the model successfully find?)*
3. **F1-Score (Dice):** $\frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$
   *(Harmonic mean of precision and recall).*
4. **IoU (Jaccard Index):** $\frac{Intersection}{Union}$
   *(Spatial overlap between prediction and ground truth).*

*Note: Division by zero (e.g., if a class never appears in the dataset or predictions) safely returns `NaN`.*

## 4. Aggregation Strategies

To provide a holistic view of the model's performance, the per-class metrics are averaged using three distinct strategies.

### 4.1 Macro (Mean) Average
Calculates the unweighted mean of the metric across all valid classes. 
- **Formula:** $\frac{1}{C} \sum_{c=1}^{C} Metric_c$
- **Use Case:** Treats all classes equally. Highly sensitive to extreme misclassifications in rare/minority classes.

### 4.2 Weighted Average
Averages the metrics, but weights each class's contribution by its true frequency (number of pixels) in the ground truth dataset.
- **Weight ($W_c$):** $TP_c + FN_c$ (Total true pixels for class $c$)
- **Formula:** $\sum_{c=1}^{C} (Metric_c \cdot \frac{W_c}{\sum W})$
- **Use Case:** Reflects overall visual performance. Massive background classes will dominate this metric, making it less sensitive to minority class failures.

### 4.3 Micro Average
Calculates the metric globally by aggregating the raw TP, FP, and FN counts across all classes *before* computing the ratio.
- **Micro Precision:** $\frac{\sum TP}{\sum TP + \sum FP}$
- **Use Case:** In multi-class segmentation where every pixel is assigned exactly one class, Micro Precision, Micro Recall, and Micro F1 will mathematically equal the overall **Pixel Accuracy**.
