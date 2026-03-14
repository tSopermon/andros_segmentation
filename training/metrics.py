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
        for c in range(self.num_classes):
            pred_c = (pred == c).astype(np.float32)
            target_c = (target == c).astype(np.float32)
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
            dict: Dictionary with per-class and mean metrics (precision, recall, f1, iou).
        """
        metrics = {}
        precision = np.divide(self.tp, self.tp + self.fp, where=(self.tp + self.fp) != 0, out=np.ones_like(self.tp))
        recall = np.divide(self.tp, self.tp + self.fn, where=(self.tp + self.fn) != 0, out=np.ones_like(self.tp))
        f1 = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0, out=np.zeros_like(precision))
        iou = np.divide(self.intersection, self.union, where=self.union != 0, out=np.ones_like(self.intersection))
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        metrics['iou'] = iou
        metrics['precision_mean'] = np.nanmean(precision)
        metrics['recall_mean'] = np.nanmean(recall)
        metrics['f1_mean'] = np.nanmean(f1)
        metrics['iou_mean'] = np.nanmean(iou)
        return metrics