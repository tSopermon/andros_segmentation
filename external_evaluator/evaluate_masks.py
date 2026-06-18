import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MaskEvaluator:
    """
    Evaluator to compute Precision, Recall, F1, and IoU from predicted and ground truth masks.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, pred, target):
        """
        Update metric accumulators with new masks.
        
        Args:
            pred (np.ndarray): Predicted mask (H, W).
            target (np.ndarray): Ground truth mask (H, W).
        """
        # Valid mask: ignore pixels where target is an IGNORE_INDEX (e.g. -1 or 255)
        # Assuming typical valid classes 0 to num_classes-1
        valid_mask = ((target >= 0) & (target < self.num_classes)).astype(np.float32)

        for c in range(self.num_classes):
            pred_c = (pred == c).astype(np.float32)
            target_c = (target == c).astype(np.float32)
            
            # Mask out invalid pixels before accumulation
            pred_c = pred_c * valid_mask
            target_c = target_c * valid_mask
            
            self.tp[c] += (pred_c * target_c).sum()
            self.fp[c] += (pred_c * (1 - target_c)).sum()
            self.fn[c] += ((1 - pred_c) * target_c).sum()
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - intersection
            self.intersection[c] += intersection
            self.union[c] += union

    def compute_metrics(self):
        """
        Compute and return all metrics as a dictionary.
        """
        metrics = {}
        # Per-class metrics
        precision = np.divide(self.tp, self.tp + self.fp, out=np.full_like(self.tp, np.nan), where=(self.tp + self.fp) != 0)
        recall = np.divide(self.tp, self.tp + self.fn, out=np.full_like(self.tp, np.nan), where=(self.tp + self.fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.full_like(precision, np.nan), where=(precision + recall) != 0)
        iou = np.divide(self.intersection, self.union, out=np.full_like(self.intersection, np.nan), where=self.union != 0)
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['iou'] = iou

        # 1. Macro (Mean) Averages
        metrics['precision_mean'] = np.nanmean(precision)
        metrics['recall_mean'] = np.nanmean(recall)
        metrics['f1_mean'] = np.nanmean(f1)
        metrics['iou_mean'] = np.nanmean(iou)

        # 2. Weighted Averages
        weights = self.tp + self.fn
        total_pixels = weights.sum()
        if total_pixels > 0:
            w = weights / total_pixels
            metrics['precision_weighted'] = np.nansum(precision * w)
            metrics['recall_weighted'] = np.nansum(recall * w)
            metrics['f1_weighted'] = np.nansum(f1 * w)
            metrics['iou_weighted'] = np.nansum(iou * w)
        else:
            metrics['precision_weighted'] = metrics['precision_mean']
            metrics['recall_weighted'] = metrics['recall_mean']
            metrics['f1_weighted'] = metrics['f1_mean']
            metrics['iou_weighted'] = metrics['iou_mean']

        # 3. Micro Averages
        tp_sum = self.tp.sum()
        fp_sum = self.fp.sum()
        fn_sum = self.fn.sum()
        intersection_sum = self.intersection.sum()
        union_sum = self.union.sum()

        p_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 1.0
        r_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 1.0
        metrics['precision_micro'] = p_micro
        metrics['recall_micro'] = r_micro
        metrics['f1_micro'] = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0
        metrics['iou_micro'] = intersection_sum / union_sum if union_sum > 0 else 1.0
        
        return metrics

def decode_rgb_mask(rgb_img, num_classes):
    """
    Decodes an RGB visualization mask back to a 2D array of class indices 
    using the project's standard color palette.
    """
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

    if num_classes > len(custom_colors):
        rng = np.random.RandomState(42)
        for _ in range(num_classes - len(custom_colors)):
            custom_colors.append(tuple(rng.randint(0, 256, size=3)))

    # Start with an array of -1 (IGNORE_INDEX)
    indices = np.full(rgb_img.shape[:2], -1, dtype=np.int32)
    
    # Check if there's an alpha channel, drop it if so
    if rgb_img.shape[-1] == 4:
        rgb_img = rgb_img[:, :, :3]
        
    for class_idx, color in enumerate(custom_colors[:num_classes]):
        # Find pixels matching this color
        mask = np.all(rgb_img == color, axis=-1)
        indices[mask] = class_idx
        
    return indices

def main():
    parser = argparse.ArgumentParser(description="Evaluate predicted masks against ground truth masks.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing Ground Truth masks")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing Predicted masks")
    parser.add_argument("--num_classes", type=int, required=True, help="Number of segmentation classes")
    args = parser.parse_args()

    gt_dir = args.gt_dir
    pred_dir = args.pred_dir

    if not os.path.exists(gt_dir):
        logger.error(f"Ground truth directory not found: {gt_dir}")
        return
    if not os.path.exists(pred_dir):
        logger.error(f"Predictions directory not found: {pred_dir}")
        return

    # Gather matching files
    valid_exts = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(valid_exts)])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(valid_exts)])

    gt_set = set(gt_files)
    pred_set = set(pred_files)
    common_files = sorted(list(gt_set.intersection(pred_set)))

    if len(common_files) == 0:
        logger.error("No matching filenames found between Ground Truth and Predictions directories.")
        logger.info(f"Sample GT files: {gt_files[:3]}")
        logger.info(f"Sample Pred files: {pred_files[:3]}")
        return

    logger.info(f"Found {len(common_files)} matching mask pairs. Starting evaluation...")

    evaluator = MaskEvaluator(num_classes=args.num_classes)

    for filename in tqdm(common_files, desc="Evaluating Masks"):
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        try:
            gt_img = np.array(Image.open(gt_path))
            pred_img = np.array(Image.open(pred_path))
            # If masks are loaded as RGB, decode them back to class indices
            if gt_img.ndim > 2 and gt_img.shape[-1] >= 3:
                gt_img = decode_rgb_mask(gt_img, args.num_classes)
            if pred_img.ndim > 2 and pred_img.shape[-1] >= 3:
                pred_img = decode_rgb_mask(pred_img, args.num_classes)

            if gt_img.shape != pred_img.shape:
                logger.warning(f"Shape mismatch for {filename}: GT {gt_img.shape} != Pred {pred_img.shape}. Skipping.")
                continue

            evaluator.update(pred_img, gt_img)
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")

    metrics = evaluator.compute_metrics()

    print("\n" + "="*85)
    print(f"{'Class':<20} {'Precision':<15} {'Recall':<15} {'F1':<15} {'IoU':<15}")
    print("-" * 85)
    for c in range(args.num_classes):
        print(f"Class_{c:<14} {metrics['precision'][c]:<15.4f} {metrics['recall'][c]:<15.4f} "
              f"{metrics['f1'][c]:<15.4f} {metrics['iou'][c]:<15.4f}")
    
    print("-" * 85)
    print(f"{'MACRO (Mean)':<20} {metrics['precision_mean']:<15.4f} {metrics['recall_mean']:<15.4f} "
          f"{metrics['f1_mean']:<15.4f} {metrics['iou_mean']:<15.4f}")
    print(f"{'WEIGHTED':<20} {metrics['precision_weighted']:<15.4f} {metrics['recall_weighted']:<15.4f} "
          f"{metrics['f1_weighted']:<15.4f} {metrics['iou_weighted']:<15.4f}")
    print(f"{'MICRO':<20} {metrics['precision_micro']:<15.4f} {metrics['recall_micro']:<15.4f} "
          f"{metrics['f1_micro']:<15.4f} {metrics['iou_micro']:<15.4f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
