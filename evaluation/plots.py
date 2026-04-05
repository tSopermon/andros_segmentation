
"""
Plotting utilities for segmentation model evaluation.

This module provides functions to visualize confusion matrices, metric distributions, correlations, and per-class/model performance for segmentation models.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# --- Plotting Functions for Model Evaluation ---

def plot_confusion_matrices(all_preds_dict, all_targets, label_mapping, class_names=None, output_dir="outputs"): 
    """
    Plot confusion matrix heatmaps for each model using cached predictions.

    Args:
        all_preds_dict (dict): Dictionary mapping model names to predicted masks (np.ndarray of shape [N, H, W]).
        all_targets (np.ndarray): Target masks (np.ndarray of shape [N, H, W]).
        label_mapping (dict): Mapping from original to new class labels.
        output_dir (str): Directory to save confusion matrix images. Defaults to "outputs".

    Returns:
        None
    """
    class_names_list = class_names if class_names is not None else [str(l) for l in label_mapping.keys()]
    targets_flat = all_targets.reshape(-1)
    
    for model_name, preds in all_preds_dict.items():
        preds_flat = preds.reshape(-1)
        cm = confusion_matrix(targets_flat, preds_flat, labels=list(label_mapping.values()))
        
        # Normalize the confusion matrix by row (True labels)
        # Using np.errstate to handle division by zero for classes not present in the test set
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Normalized Confusion Matrix - {model_name}')
        plt.tight_layout()
        save_path = f"{output_dir}/confusion_matrix_{model_name}.png"
        plt.savefig(save_path)
        plt.close()

def plot_metric_vs_class_frequency(all_test_results, all_targets, label_mapping=None, class_names=None, output_dir="outputs"):
    """
    Plot metric (IoU, F1) vs. class frequency for each model.

    Args:
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        all_targets (np.ndarray): Ground truth target masks (shape N, H, W).
        label_mapping (dict): Mapping from original to new class labels.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    # Compute class frequencies directly from pre-cached targets
    num_classes = len(class_names) if class_names is not None else (len(label_mapping) if label_mapping is not None else 0)
    class_counts = np.zeros(num_classes)
    vals, counts = np.unique(all_targets, return_counts=True)
    for v, c in zip(vals, counts):
        if 0 <= v < len(class_counts):
            class_counts[int(v)] += c
            
    class_freq = class_counts / class_counts.sum() if class_counts.sum() > 0 else np.zeros_like(class_counts)
    metrics = ['iou', 'f1']
    class_indices = np.arange(1, len(class_counts) + 1)  # 1,2,...,n
    class_names_list = class_names if class_names is not None else [str(i) for i in class_indices]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plotted_any = False
        # build matrix of metric values: shape (num_models, num_classes)
        all_model_values = []
        model_names = []
        for model_name, results in all_test_results.items():
            y = np.array(results[metric])
            all_model_values.append(y)
            model_names.append(model_name)
        if len(all_model_values) == 0:
            continue
        all_model_values = np.stack(all_model_values, axis=0)  # (num_models, num_classes)
        for model_idx in range(all_model_values.shape[0]):
            y = all_model_values[model_idx]
            valid = (class_freq > 0) & np.isfinite(y)
            if np.any(valid):
                plt.scatter(class_freq[valid], y[valid], label=model_names[model_idx], s=60, alpha=0.8)
                plotted_any = True
        # Annotate points with class names using mean across models for vertical placement
        mean_y = np.nanmean(all_model_values, axis=0)
        for i in range(len(class_indices)):
            if class_freq[i] > 0 and np.isfinite(mean_y[i]):
                plt.annotate(str(class_names_list[i]), (class_freq[i], mean_y[i]), fontsize=9,
                             xytext=(6, 3), textcoords='offset points')
        plt.xlabel('Class Frequency')
        plt.ylabel(metric.upper())
        plt.title(f'{metric.upper()} vs. Class Frequency')
        if plotted_any:
            plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        save_path = f"{output_dir}/{metric}_vs_class_frequency.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_per_image_metric_distribution(all_preds_dict, all_targets, class_names=None, output_dir="outputs"):
    """
    Plot boxplots of per-image IoU and F1 for each model using cached predictions.

    Args:
        all_preds_dict (dict): Dictionary mapping model names to predicted masks (np.ndarray of shape [N, H, W]).
        all_targets (np.ndarray): Target masks (np.ndarray of shape [N, H, W]).
        class_names (list): List of class names.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    metric_names = ['iou', 'f1']
    per_image_metrics = {model_name: {m: [] for m in metric_names} for model_name in all_preds_dict}
    
    num_classes = len(class_names) if class_names is not None else int(np.max(all_targets)) + 1
    
    for model_name, preds in all_preds_dict.items():
        for i in range(preds.shape[0]):
            pred_i = preds[i]
            target_i = all_targets[i]
            valid_mask = ((target_i >= 0) & (target_i < num_classes)).astype(np.float32)

            image_iou = []
            image_f1 = []
            for c in range(num_classes):
                pred_c = (pred_i == c).astype(np.float32) * valid_mask
                target_c = (target_i == c).astype(np.float32) * valid_mask
                
                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum() - intersection
                
                tp = intersection
                fp = pred_c.sum() - tp
                fn = target_c.sum() - tp
                
                if union > 0:
                    image_iou.append(intersection / union)
                
                if (tp + fp + fn) > 0:
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    image_f1.append(f1)
                    
            if len(image_iou) > 0:
                per_image_metrics[model_name]['iou'].append(np.mean(image_iou))
            if len(image_f1) > 0:
                per_image_metrics[model_name]['f1'].append(np.mean(image_f1))
                
    for metric in metric_names:
        plt.figure(figsize=(10, 6))
        data = [per_image_metrics[m][metric] for m in all_preds_dict if len(per_image_metrics[m][metric]) > 0]
        valid_models = [m for m in all_preds_dict if len(per_image_metrics[m][metric]) > 0]
        
        if not data:
            continue
            
        # Show means and medians
        box = plt.boxplot(data, labels=valid_models, patch_artist=True, showmeans=True,
                          meanprops={"marker":"D", "markeredgecolor":"black", "markerfacecolor":"gold"})
        plt.ylabel(metric.upper())
        plt.title(f'Per-Image {metric.upper()} Distribution')
        # annotate mean values
        for i, vals in enumerate(data):
            if len(vals) > 0:
                mean_val = np.nanmean(vals)
                plt.text(i+1, mean_val, f"{mean_val:.3f}", horizontalalignment='center', verticalalignment='bottom')
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        save_path = f"{output_dir}/per_image_{metric}_boxplot.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_metric_correlation_matrix(all_test_results, output_dir="outputs"):
    """
    Plot correlation matrix of metrics across all models and classes.

    Args:
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    metrics = ['precision', 'recall', 'f1', 'iou']
    for model_name, results in all_test_results.items():
        data = np.stack([results[m] for m in metrics], axis=1)  # shape: (num_classes, num_metrics)
        corr = np.corrcoef(data, rowvar=False)
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, xticklabels=metrics, yticklabels=metrics, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Metric Correlation Matrix - {model_name}')
        plt.tight_layout()
        save_path = f"{output_dir}/metric_correlation_{model_name}.png"
        plt.savefig(save_path)
        plt.close()

# Also move the previously defined metric plots for completeness

def plot_metric_per_class(all_test_results, metric_name, class_labels, output_dir="outputs"):
    """
    Plot a given metric per class for each model.

    Args:
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        metric_name (str): Name of the metric to plot (e.g., 'iou', 'f1').
        class_labels (list): List of class labels.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    num_classes = len(class_labels)
    class_indices = np.arange(1, num_classes + 1)
    for model_name, metrics in all_test_results.items():
        plt.plot(class_indices, metrics[metric_name], marker='o', label=model_name)
    plt.xlabel('Class')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} per Class for Each Model')
    plt.xticks(class_indices, class_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = f"{output_dir}/{metric_name}_per_class.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_mean_metrics(all_test_results, output_dir="outputs"):
    """
    Plot mean metrics (precision, recall, F1, IoU) for each model.

    Args:
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    metrics = ['precision_mean', 'recall_mean', 'f1_mean', 'iou_mean']
    model_names = list(all_test_results.keys())
    values = [[all_test_results[m][metric] for metric in metrics] for m in model_names]
    values = np.array(values)
    x = np.arange(len(metrics))
    width = 0.18
    plt.figure(figsize=(10, 6))
    for i, (model_name, vals) in enumerate(zip(model_names, values)):
        plt.bar(x + i*width, vals, width, label=model_name)
    plt.xticks(x + width*(len(model_names)-1)/2, [m.replace('_mean','').capitalize() for m in metrics])
    plt.ylabel('Score')
    plt.title('Mean Metrics for Each Model')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    save_path = f"{output_dir}/mean_metrics.png"
    plt.savefig(save_path)
    plt.close()

def plot_metric_per_model_per_class(all_test_results, class_labels, output_dir="outputs"):
    """
    Plot metrics for each class across all models.

    Args:
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        class_labels (list): List of class labels.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    metrics = ['precision', 'recall', 'f1', 'iou']
    model_names = list(all_test_results.keys())
    for idx in range(len(class_labels)):
        class_num = idx + 1
        class_name = class_labels[idx]
        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            vals = [all_test_results[m][metric][idx] for m in model_names]
            plt.bar(np.arange(len(model_names)) + i*0.2, vals, width=0.18, label=metric.capitalize() if i==0 else "")
        plt.xticks(np.arange(len(model_names)) + 0.3, model_names)
        plt.ylim(0, 1.05)
        plt.title(f'Metrics for "{class_name}" class')
        plt.ylabel('Score')
        plt.legend(metrics)
        plt.tight_layout()
        # Use class name in filename (sanitise spaces)
        safe_name = class_name.replace(' ', '_').replace('/', '_')
        save_path = f"{output_dir}/{safe_name}_metrics.png"
        plt.savefig(save_path)
        plt.close()

def plot_all_averages(all_test_results, output_dir="outputs"):
    """
    Plot comparison of Macro, Weighted, and Micro IoU for each model.

    Args:
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        output_dir (str): Directory to save plots. Defaults to "outputs".

    Returns:
        None
    """
    model_names = list(all_test_results.keys())
    if not model_names:
        return
    
    # We focus on IoU as the representative metric
    types = ['mean', 'weighted', 'micro']
    labels = ['Macro (Mean)', 'Weighted', 'Micro']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, t in enumerate(types):
        vals = [all_test_results[m].get(f'iou_{t}', 0) for m in model_names]
        plt.bar(x + i*width, vals, width, label=labels[i])
        
    plt.xlabel('Model')
    plt.ylabel('IoU Score')
    plt.title('IoU Comparison: Macro vs Weighted vs Micro')
    plt.xticks(x + width, model_names, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    save_path = f"{output_dir}/iou_averages_comparison.png"
    plt.savefig(save_path)
    plt.close()
