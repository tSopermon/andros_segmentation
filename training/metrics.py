import numpy as np
import torch

class SegmentationMetrics:
    """
    Compute per-class metrics for segmentation: Precision, Recall, F1, IoU, Dice.
    """
    def __init__(self, num_classes, num_batches=0):
        """
        Initialize the metrics object.

        Args:
            num_classes (int): Number of segmentation classes.
            num_batches (int): Number of batches (optional, for tracking).
        """
        self.num_classes = num_classes
        self.reset()
        self.num_batches = num_batches

    def reset(self):
        """
        Reset all metric accumulators to zero.
        """
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

    def update(self, pred, target):
        """
        Update metric accumulators with a new batch of predictions and targets.

        Args:
            pred (torch.Tensor): Model output logits (N, C, H, W).
            target (torch.Tensor): Ground truth mask (N, H, W).
        """
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        target = target.cpu().numpy()
        
        # Valid mask: ignore pixels where target is an IGNORE_INDEX (e.g. -1 or 255)
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

        Returns:
            dict: Dictionary with per-class, mean (macro), weighted, and micro metrics.
        """
        metrics = {}
        # Per-class metrics
        precision = np.divide(self.tp, self.tp + self.fp, where=(self.tp + self.fp) != 0, out=np.ones_like(self.tp))
        recall = np.divide(self.tp, self.tp + self.fn, where=(self.tp + self.fn) != 0, out=np.ones_like(self.tp))
        f1 = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0, out=np.zeros_like(precision))
        iou = np.divide(self.intersection, self.union, where=self.union != 0, out=np.ones_like(self.intersection))
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['iou'] = iou

        # 1. Macro (Mean) Averages - already in system
        metrics['precision_mean'] = np.nanmean(precision)
        metrics['recall_mean'] = np.nanmean(recall)
        metrics['f1_mean'] = np.nanmean(f1)
        metrics['iou_mean'] = np.nanmean(iou)

        # 2. Weighted Averages (Weighted by number of pixels in each class)
        weights = self.tp + self.fn
        total_pixels = weights.sum()
        if total_pixels > 0:
            w = weights / total_pixels
            metrics['precision_weighted'] = np.sum(precision * w)
            metrics['recall_weighted'] = np.sum(recall * w)
            metrics['f1_weighted'] = np.sum(f1 * w)
            metrics['iou_weighted'] = np.sum(iou * w)
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