
"""
Export utilities for segmentation model metrics.

This module provides functions to export per-class and summary metrics for all models to CSV files for further analysis and reporting.
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def export_metrics(models_dict, all_test_results, training_history, num_classes, class_names=None, output_dir="outputs"):
    """
    Export per-class and summary metrics for all models to CSV files.

    Args:
        models_dict (dict): Dictionary mapping model names to model instances.
        all_test_results (dict): Dictionary mapping model names to test metrics dictionaries.
        training_history (dict): Dictionary mapping model names to training/validation history.
        num_classes (int): Number of segmentation classes.
        output_dir (str): Directory to save CSV files. Defaults to "outputs".

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    for model_name in models_dict.keys():
        test_result = all_test_results[model_name]
        train_history = training_history[model_name]
        final_train_f1 = train_history['train_f1_mean'][-1]
        final_train_prec = train_history['train_precision_mean'][-1]
        final_train_rec = train_history['train_recall_mean'][-1]
        final_train_iou = train_history['train_iou_mean'][-1]
        final_val_f1 = train_history['val_f1_mean'][-1]
        final_val_prec = train_history['val_precision_mean'][-1]
        final_val_rec = train_history['val_recall_mean'][-1]
        final_val_iou = train_history['val_iou_mean'][-1]
        # Determine class labels for CSV export
        if class_names is not None and len(class_names) >= num_classes:
            csv_class_labels = class_names[:num_classes]
        else:
            csv_class_labels = [f'Class {i}' for i in range(num_classes)]

        df = pd.DataFrame({
            'Class': csv_class_labels,
            'Test_Precision': test_result['precision'],
            'Test_Recall': test_result['recall'],
            'Test_F1': test_result['f1'],
            'Test_IoU': test_result['iou']
        })
        mean_row = pd.DataFrame({
            'Class': ['TEST_MEAN'],
            'Test_Precision': [test_result['precision_mean']],
            'Test_Recall': [test_result['recall_mean']],
            'Test_F1': [test_result['f1_mean']],
            'Test_IoU': [test_result['iou_mean']]
        })
        train_row = pd.DataFrame({
            'Class': ['TRAIN_MEAN'],
            'Test_Precision': [final_train_prec],
            'Test_Recall': [final_train_rec],
            'Test_F1': [final_train_f1],
            'Test_IoU': [final_train_iou]
        })
        val_row = pd.DataFrame({
            'Class': ['VAL_MEAN'],
            'Test_Precision': [final_val_prec],
            'Test_Recall': [final_val_rec],
            'Test_F1': [final_val_f1],
            'Test_IoU': [final_val_iou]
        })
        df = pd.concat([train_row, val_row, pd.DataFrame({'Class': ['']}), df, mean_row], ignore_index=True)
        csv_path = os.path.join(output_dir, f"{model_name}_test_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ {model_name} metrics saved to {csv_path}")

    summary_data = []
    for model_name in models_dict.keys():
        test_result = all_test_results[model_name]
        train_history = training_history[model_name]
        summary_data.append({
            'Model': model_name,
            'Train_Precision': train_history['train_precision_mean'][-1],
            'Train_Recall': train_history['train_recall_mean'][-1],
            'Train_F1': train_history['train_f1_mean'][-1],
            'Train_mIoU': train_history['train_iou_mean'][-1],
            'Val_Precision': train_history['val_precision_mean'][-1],
            'Val_Recall': train_history['val_recall_mean'][-1],
            'Val_F1': train_history['val_f1_mean'][-1],
            'Val_mIoU': train_history['val_iou_mean'][-1],
            'Test_Precision': test_result['precision_mean'],
            'Test_Recall': test_result['recall_mean'],
            'Test_F1': test_result['f1_mean'],
            'Test_mIoU': test_result['iou_mean']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, "models_summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\n✓ Summary metrics saved to '{summary_csv}'")
    logger.info("\nSummary Table:")
    logger.info(summary_df.to_string(index=False))