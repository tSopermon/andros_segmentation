"""
Evaluation utilities for segmentation models.

This module provides helper functions for evaluating segmentation models, aggregating metrics, and formatting results for reporting.
"""
import torch
from training.metrics import SegmentationMetrics
from training.train_utils import evaluate_model
import logging

logger = logging.getLogger(__name__)

def evaluate_all_models(models_dict, test_loader, device, num_classes, label_mapping, class_names=None):
    """
    Evaluate all models on the test set and print per-class and mean metrics.

    Args:
        models_dict (dict): Dictionary mapping model names to model instances.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        device (torch.device): Device to run evaluation on (e.g., 'cuda' or 'cpu').
        num_classes (int): Number of segmentation classes.
        label_mapping (dict): Mapping from original to new class labels.

    Returns:
        dict: Dictionary mapping model names to their metrics dictionary.
    """
    all_test_results = {}
    for model_name, model in models_dict.items():
        model.load_state_dict(torch.load(f"checkpoints/{model_name}_best.pth"))
        model = model.to(device)
        test_metrics = SegmentationMetrics(num_classes)
        test_metrics_dict = evaluate_model(model, test_loader, device, test_metrics)
        all_test_results[model_name] = test_metrics_dict
        logger.info("%s", "\n" + model_name.upper())
        logger.info('%s', '-' * 80)
        logger.info("%s", f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'IoU':<12}")
        logger.info('%s', '-' * 80)
        for c in range(num_classes):
            cname = class_names[c] if (class_names is not None and c < len(class_names)) else f'Class_{c}'
            logger.info(f"{cname:<20} {test_metrics_dict['precision'][c]:<12.4f} {test_metrics_dict['recall'][c]:<12.4f} "
                f"{test_metrics_dict['f1'][c]:<12.4f} {test_metrics_dict['iou'][c]:<12.4f}")
        logger.info('%s', '-' * 80)
        logger.info(f"{'MEAN':<10} {test_metrics_dict['precision_mean']:<12.4f} {test_metrics_dict['recall_mean']:<12.4f} "
                  f"{test_metrics_dict['f1_mean']:<12.4f} {test_metrics_dict['iou_mean']:<12.4f}")
        logger.info('%s', '=' * 80)
    return all_test_results
