
"""
Visualization utilities for segmentation model predictions.

This module provides functions to overlay color masks on images and visualize predictions from multiple models for qualitative comparison.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

logger = logging.getLogger(__name__)

def apply_color_mask(image, mask, colors, alpha=0.4):
    """
    Apply a color mask overlay to an image for each class in the mask.

    Args:
        image (np.ndarray): Input RGB image of shape (H, W, 3).
        mask (np.ndarray): Segmentation mask of shape (H, W), with integer class labels.
        colors (np.ndarray): Array of RGB colors for each class (num_classes, 3).
        alpha (float): Opacity of the overlay. Defaults to 0.4.

    Returns:
        np.ndarray: Image with color mask overlay (uint8, shape (H, W, 3)).
    """
    overlay = image.copy()
    for class_id in range(len(colors)):
        mask_class = (mask == class_id)
        overlay[mask_class] = overlay[mask_class] * (1 - alpha) + colors[class_id] * alpha
    return overlay.astype(np.uint8)

def visualize_predictions(models_dict, test_dataset, test_images, test_masks, TEST_IMG_PATH, TEST_MASK_PATH, label_mapping, device, class_names=None):
    """
    Visualize and compare predictions from multiple models on random test images.

    Args:
        models_dict (dict): Dictionary mapping model names to model instances.
        test_dataset (torch.utils.data.Dataset): Test dataset object.
        test_images (list): List of test image filenames.
        test_masks (list): List of test mask filenames.
        logger.info("✓ Prediction overlays saved to 'outputs/prediction_overlays.png'")
        label_mapping (dict): Mapping from original to new class labels.
        device (torch.device): Device to run inference on.

    Saves:
        'outputs/prediction_overlays.png' with visualized predictions.
    """
    # Use explicit class colors provided by user (order must match class_names passed in)
    num_classes = len(class_names) if class_names is not None else len(label_mapping)
    explicit_colors = [
        (0, 0, 255),      # Water
        (60, 16, 152),    # Woodland
        (132, 41, 246),   # Arable land
        (0, 255, 0),      # Frygana
        (155, 155, 155),  # Other
        (226, 169, 41),   # Artificial land
        (255, 255, 0),    # Perm. Cult
        (255, 255, 255),  # Bareland
    ]
    colors = np.array(explicit_colors[:num_classes], dtype=np.uint8)
    viz_indices = np.random.choice(len(test_images), size=3, replace=False)
    num_models = len(models_dict)
    fig, axes = plt.subplots(3, num_models + 2, figsize=(4 * (num_models + 2), 12))
    for row_idx, img_idx in enumerate(viz_indices):
        img_path = TEST_IMG_PATH / test_images[img_idx]
        mask_path = TEST_MASK_PATH / test_masks[img_idx]
        original_img = cv2.imread(str(img_path))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_mask_mapped = np.zeros_like(gt_mask)
        for orig, idx in label_mapping.items():
            gt_mask_mapped[gt_mask == orig] = idx
        axes[row_idx, 0].imshow(original_img)
        axes[row_idx, 0].set_title(f'Original Image ({test_images[img_idx]})')
        axes[row_idx, 0].axis('off')
        gt_overlay = apply_color_mask(original_img, gt_mask_mapped, colors)
        axes[row_idx, 1].imshow(gt_overlay)
        axes[row_idx, 1].set_title(f'Ground Truth ({test_masks[img_idx]})')
        axes[row_idx, 1].axis('off')
        for model_idx, (model_name, model) in enumerate(models_dict.items()):
            model.load_state_dict(torch.load(f"checkpoints/{model_name}_best.pth"))
            model.eval()
            input_tensor, _ = test_dataset[img_idx]
            input_batch = input_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_batch)
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            pred_overlay = apply_color_mask(original_img, pred_mask, colors)
            axes[row_idx, model_idx + 2].imshow(pred_overlay)
            axes[row_idx, model_idx + 2].set_title(f'{model_name} ({test_masks[img_idx]})')
            axes[row_idx, model_idx + 2].axis('off')
    plt.suptitle('Model Predictions Comparison on Test Images', fontsize=16, y=0.995)
    # Add a legend mapping colors to class names if available
    if class_names is not None:
        import matplotlib.patches as mpatches
        legend_handles = []
        for i, cname in enumerate(class_names[:num_classes]):
            color = tuple(colors[i] / 255.0)
            legend_handles.append(mpatches.Patch(color=color, label=cname))
        plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('outputs/prediction_overlays.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("✓ Prediction overlays saved to 'outputs/prediction_overlays.png'")