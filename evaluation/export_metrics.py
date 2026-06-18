
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
        import numpy as np
        
        val_ious = train_history.get('val_iou_mean', [])
        train_ious = train_history.get('train_iou_mean', [])
        
        if len(val_ious) > 0:
            best_idx = np.argmax(val_ious)
            final_val_f1 = train_history['val_f1_mean'][best_idx]
            final_val_prec = train_history['val_precision_mean'][best_idx]
            final_val_rec = train_history['val_recall_mean'][best_idx]
            final_val_iou = train_history['val_iou_mean'][best_idx]
            
            final_train_f1 = train_history['train_f1_mean'][best_idx]
            final_train_prec = train_history['train_precision_mean'][best_idx]
            final_train_rec = train_history['train_recall_mean'][best_idx]
            final_train_iou = train_history['train_iou_mean'][best_idx]
        elif len(train_ious) > 0:
            best_idx = np.argmax(train_ious)
            final_train_f1 = train_history['train_f1_mean'][best_idx]
            final_train_prec = train_history['train_precision_mean'][best_idx]
            final_train_rec = train_history['train_recall_mean'][best_idx]
            final_train_iou = train_history['train_iou_mean'][best_idx]
            final_val_f1 = final_val_prec = final_val_rec = final_val_iou = 0.0
        else:
            final_train_f1 = final_train_prec = final_train_rec = final_train_iou = 0.0
            final_val_f1 = final_val_prec = final_val_rec = final_val_iou = 0.0

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
        weighted_row = pd.DataFrame({
            'Class': ['TEST_WEIGHTED'],
            'Test_Precision': [test_result.get('precision_weighted', 0)],
            'Test_Recall': [test_result.get('recall_weighted', 0)],
            'Test_F1': [test_result.get('f1_weighted', 0)],
            'Test_IoU': [test_result.get('iou_weighted', 0)]
        })
        micro_row = pd.DataFrame({
            'Class': ['TEST_MICRO'],
            'Test_Precision': [test_result.get('precision_micro', 0)],
            'Test_Recall': [test_result.get('recall_micro', 0)],
            'Test_F1': [test_result.get('f1_micro', 0)],
            'Test_IoU': [test_result.get('iou_micro', 0)]
        })
        df = pd.concat([train_row, val_row, pd.DataFrame({'Class': ['']}), df, mean_row, weighted_row, micro_row], ignore_index=True)
        csv_path = os.path.join(output_dir, f"{model_name}_test_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ {model_name} metrics saved to {csv_path}")

    summary_data = []
    for model_name in models_dict.keys():
        test_result = all_test_results[model_name]
        train_history = training_history[model_name]
        import numpy as np
        val_ious = train_history.get('val_iou_mean', [])
        train_ious = train_history.get('train_iou_mean', [])
        
        if len(val_ious) > 0:
            best_idx = np.argmax(val_ious)
            t_prec = train_history['train_precision_mean'][best_idx]
            t_rec = train_history['train_recall_mean'][best_idx]
            t_f1 = train_history['train_f1_mean'][best_idx]
            t_iou = train_history['train_iou_mean'][best_idx]
            v_prec = train_history['val_precision_mean'][best_idx]
            v_rec = train_history['val_recall_mean'][best_idx]
            v_f1 = train_history['val_f1_mean'][best_idx]
            v_iou = train_history['val_iou_mean'][best_idx]
        elif len(train_ious) > 0:
            best_idx = np.argmax(train_ious)
            t_prec = train_history['train_precision_mean'][best_idx]
            t_rec = train_history['train_recall_mean'][best_idx]
            t_f1 = train_history['train_f1_mean'][best_idx]
            t_iou = train_history['train_iou_mean'][best_idx]
            v_prec = v_rec = v_f1 = v_iou = 0.0
        else:
            t_prec = t_rec = t_f1 = t_iou = 0.0
            v_prec = v_rec = v_f1 = v_iou = 0.0

        summary_data.append({
            'Model': model_name,
            'Train_Precision': t_prec,
            'Train_Recall': t_rec,
            'Train_F1': t_f1,
            'Train_mIoU': t_iou,
            'Val_Precision': v_prec,
            'Val_Recall': v_rec,
            'Val_F1': v_f1,
            'Val_mIoU': v_iou,
            'Test_Precision': test_result['precision_mean'],
            'Test_Recall': test_result['recall_mean'],
            'Test_F1': test_result['f1_mean'],
            'Test_mIoU': test_result['iou_mean'],
            'Test_Weighted_IoU': test_result.get('iou_weighted', 0),
            'Test_Micro_IoU': test_result.get('iou_micro', 0)
        })
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, "models_summary_metrics.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"\n✓ Summary metrics saved to '{summary_csv}'")
    logger.info("\nSummary Table:")
    logger.info(summary_df.to_string(index=False))