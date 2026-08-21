"""Generate Class Activation Maps (CAMs) for segmentation models on the test set.

For every image in the test subset, this script produces Grad-CAM-style
explanations for each evaluated model using the ``pytorch-grad-cam`` library,
and stores raw grayscale heatmaps plus JET overlays under ``outputs/gradcam/``.

Usage:
    python gradcam.py --config config/config.yaml
    python gradcam.py --config config/config.yaml --models UNetPlusPlus --limit 2
    python gradcam.py --config config/config.yaml --print-target-layers
"""
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from utils.config_loader import load_config
from utils.model_selection import get_selected_model_names
from models.model_zoo import get_models
from utils.dataset import SegmentationDataset
from utils.transforms import get_val_transform
from evaluation.gradcam_utils import (
    DEFAULT_METHODS,
    METHODS,
    build_config_summary,
    classes_present_in_mask,
    get_reshape_transform,
    module_paths,
    resolve_target_layers,
    sanitize,
    unnormalize_image,
)
from utils.logging_config import configure_logging

try:
    import pytorch_grad_cam
    from pytorch_grad_cam import EigenCAM, SegEigenCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError as exc:  # pragma: no cover - depends on environment
    raise SystemExit(
        "The 'grad-cam' package is required. Install it with: pip install grad-cam"
    ) from exc


class _Float32EigenCAM(EigenCAM):
    """EigenCAM variant that upcasts activations so numpy SVD avoids float16."""

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        if isinstance(activations, np.ndarray):
            activations = activations.astype(np.float32)
        return super().get_cam_image(input_tensor, target_layer, target_category,
                                     activations, grads, eigen_smooth)


class _Float32SegEigenCAM(SegEigenCAM):
    """SegEigenCAM variant that upcasts activations/grads for numpy SVD."""

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        if isinstance(activations, np.ndarray):
            activations = activations.astype(np.float32)
        if isinstance(grads, np.ndarray):
            grads = grads.astype(np.float32)
        return super().get_cam_image(input_tensor, target_layer, target_category,
                                     activations, grads, eigen_smooth)


def _get_cam_class(method: str):
    if method == 'EigenCAM':
        return _Float32EigenCAM
    if method == 'SegEigenCAM':
        return _Float32SegEigenCAM
    return getattr(pytorch_grad_cam, METHODS[method]['import_name'])


def _release_cam_graph(cam) -> None:
    """Drop references to a CAM's retained autograd graph to free GPU memory."""
    aag = getattr(cam, 'activations_and_grads', None)
    if aag is not None:
        aag.activations = []
        aag.gradients = []
    if hasattr(cam, 'outputs'):
        cam.outputs = None

CHECKPOINTS_DIR = 'checkpoints'
OUTPUTS_DIR = 'outputs'
GRADCAM_ROOT = os.path.join(OUTPUTS_DIR, 'gradcam')

MODEL_CHECKPOINTS = {
    'DeepLabV3': 'DeepLabV3_best.pth',
    'DeepLabV3Plus': 'DeepLabV3Plus_best.pth',
    'UNet': 'UNet_best.pth',
    'UNetPlusPlus': 'UNetPlusPlus_best.pth',
    'UNet_original': 'UNet_original_best.pth',
    'Segformer': 'Segformer_best.pth',
}


def _build_checkpoint_map() -> dict:
    mapping = dict(MODEL_CHECKPOINTS)
    mapping['DeepLabV1_original'] = 'DeepLabV1_original_best.pth'
    mapping['DeepLabV2_original'] = 'DeepLabV2_original_best.pth'
    mapping['DeepLabV3_original'] = 'DeepLabV3_original_best.pth'
    mapping['MaxViTSmallUNet'] = 'MaxViTSmallUNet_best.pth'
    return mapping


def _detect_num_classes(checkpoint: dict, fallback: int) -> int:
    for key in checkpoint.keys():
        if any(k in key for k in ['segmentation_head.0.weight', 'final_conv.weight',
                                  'head.weight', 'classifier.4.weight']):
            return int(checkpoint[key].shape[0])
    for key, val in checkpoint.items():
        if ('head' in key or 'final' in key) and val.ndim == 4:
            return int(val.shape[0])
    return fallback


def _resolve_class_labels(config, dataset_path: Path):
    """Replicate evaluate.py's grayscale -> index label mapping and class names."""
    all_classes = set()
    for subset in ['train', 'val', 'test', 'lowres']:
        mask_dir = dataset_path / subset / ('Mask' if (dataset_path / subset / 'Mask').exists() else 'mask')
        if not mask_dir.exists():
            continue
        for mask_file in os.listdir(mask_dir):
            if mask_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                mask = cv2.imread(str(mask_dir / mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    all_classes.update(np.unique(mask).tolist())

    class_labels = sorted(all_classes)
    label_mapping = {original: idx for idx, original in enumerate(class_labels)}

    class_names_config = config.get('CLASS_NAMES', None)
    class_names = []
    if class_names_config is not None:
        gray_to_name = {}
        for rgb_str, name in class_names_config.items():
            rgb_str = str(rgb_str).strip()
            rgb_parts = [p.strip() for p in rgb_str.replace('[', '').replace(']', '').split(',')]
            if len(rgb_parts) == 3:
                r, g, b = int(rgb_parts[0]), int(rgb_parts[1]), int(rgb_parts[2])
                bgr_pixel = np.array([[[b, g, r]]], dtype=np.uint8)
                gray_val = int(cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2GRAY)[0, 0])
                gray_to_name[gray_val] = name
            elif len(rgb_parts) == 1 and rgb_parts[0].isdigit():
                gray_to_name[int(rgb_parts[0])] = name

        for orig_val in class_labels:
            if gray_to_name:
                nearest_gray = min(gray_to_name.keys(), key=lambda k: abs(k - orig_val))
                if abs(nearest_gray - orig_val) <= 2:
                    class_names.append(gray_to_name[nearest_gray])
                    continue
            class_names.append(f'Class_{orig_val}')

    elif config.get('NUM_CLASSES', len(class_labels)) == 8:
        class_names = [
            'Water', 'Woodland', 'Arable land', 'Frygana',
            'Other', 'Artificial land', 'Perm. Cult', 'Bareland',
        ]
    else:
        class_names = [f'Class_{i}' for i in range(config.get('NUM_CLASSES', len(class_labels)))]

    if len(class_names) != len(class_labels):
        if len(class_names) > len(class_labels):
            class_names = class_names[:len(class_labels)]
        else:
            class_names = class_names + [f'Class_{i}' for i in range(len(class_names), len(class_labels))]

    return class_labels, label_mapping, class_names


def _run_cam(cam, input_tensor: torch.Tensor, targets, use_amp: bool) -> np.ndarray:
    """Run a CAM object, optionally under CUDA autocast to halve activation memory."""
    if use_amp:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return cam(input_tensor=input_tensor, targets=targets)
    return cam(input_tensor=input_tensor, targets=targets)


def _save_cam(cam_map: np.ndarray, rgb_image: np.ndarray, raw_path: str, overlay_path: str) -> None:
    """Save a normalized grayscale heatmap and a JET overlay."""
    raw = (np.clip(cam_map, 0.0, 1.0) * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    cv2.imwrite(raw_path, raw)

    overlay_rgb = show_cam_on_image(rgb_image, cam_map, use_rgb=True)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)


def main():
    parser = argparse.ArgumentParser(description='Generate Grad-CAM visualizations.')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML file')
    parser.add_argument('--methods', default=None,
                        help='Comma-separated CAM methods (default: config GRADCAM_METHODS or all)')
    parser.add_argument('--models', default=None,
                        help='Comma-separated model names to restrict to (default: config selection)')
    parser.add_argument('--limit', type=int, default=None, help='Only process the first N test images')
    parser.add_argument('--device', default=None, help="Device override (default: cuda if available)")
    parser.add_argument('--print-target-layers', action='store_true',
                        help='Resolve and print target layers per model, then exit')
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))
    logger = logging.getLogger(__name__)

    dataset_path = Path(config['DATASET_PATH'])
    image_size = config['IMAGE_SIZE']
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ---- CAM methods ----
    if args.methods:
        methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    else:
        methods = list(config.get('GRADCAM_METHODS', None) or DEFAULT_METHODS)
    invalid_methods = [m for m in methods if m not in METHODS]
    if invalid_methods:
        raise SystemExit(f"Unknown CAM method(s): {invalid_methods}. Valid: {list(METHODS.keys())}")

    # ---- Model selection ----
    selected_models = ([m.strip() for m in args.models.split(',') if m.strip()]
                       if args.models else get_selected_model_names(config))
    checkpoint_map = _build_checkpoint_map()
    local_checkpoints = {k: checkpoint_map[k] for k in selected_models if k in checkpoint_map}

    # ---- Test dataset paths ----
    test_img_path = dataset_path / 'test' / ('Image' if (dataset_path / 'test' / 'Image').exists() else 'image')
    test_mask_path = dataset_path / 'test' / ('Mask' if (dataset_path / 'test' / 'Mask').exists() else 'mask')
    test_images = sorted([f for f in os.listdir(test_img_path)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    test_masks = sorted([f for f in os.listdir(test_mask_path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])

    class_labels, label_mapping, class_names = _resolve_class_labels(config, dataset_path)
    num_label_classes = len(class_labels)

    val_transform = get_val_transform(image_size)
    test_dataset = SegmentationDataset(test_img_path, test_mask_path, test_images, test_masks,
                                       val_transform, label_mapping)

    limit = args.limit if args.limit is not None else len(test_dataset)
    limit = max(0, min(limit, len(test_dataset)))

    # ---- Checkpoints ----
    missing = []
    for model_name, ckpt_file in local_checkpoints.items():
        path = os.path.join(CHECKPOINTS_DIR, ckpt_file)
        if not os.path.exists(path):
            logger.error('No such file: %s', path)
            missing.append(path)
    if missing:
        raise SystemExit(2)

    model_set = config.get('MODEL_SET', 'standard')
    backbone = config.get('BACKBONE', 'resnet101')
    encoder_weights = config.get('ENCODER_WEIGHTS', 'imagenet')
    num_classes_fallback = config.get('NUM_CLASSES', num_label_classes or 8)

    target_layers_meta = {}
    gradcam_methods_meta = list(methods)

    # ---- Per-model CAM generation ----
    for model_name, ckpt_file in local_checkpoints.items():
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, ckpt_file)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        num_classes = _detect_num_classes(checkpoint, num_classes_fallback)
        logger.info('Detected %d classes from checkpoint for %s', num_classes, model_name)

        # Mirror generate_masks.py env-var registration for the original models.
        old_env = {k: os.environ.get(k) for k in (
            'USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL',
            'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
        try:
            if model_set in ('originals', 'all'):
                os.environ['USE_UNET_ORIGINAL'] = 'true'
                os.environ['USE_DEEPLABV1_ORIGINAL'] = 'true'
                os.environ['USE_DEEPLABV2_ORIGINAL'] = 'true'
                os.environ['USE_DEEPLABV3_ORIGINAL'] = 'true'
                os.environ['USE_MAXVIT_UNET'] = 'true'
            else:
                os.environ['USE_DEEPLABV1_ORIGINAL'] = str(config.get('USE_DEEPLABV1_ORIGINAL', False)).lower()
                os.environ['USE_DEEPLABV2_ORIGINAL'] = str(config.get('USE_DEEPLABV2_ORIGINAL', False)).lower()
                os.environ['USE_DEEPLABV3_ORIGINAL'] = str(config.get('USE_DEEPLABV3_ORIGINAL', False)).lower()
                os.environ['USE_MAXVIT_UNET'] = str(config.get('USE_MAXVIT_UNET', False)).lower()

            model = get_models(num_classes, backbone=backbone, encoder_weights=encoder_weights,
                               specific_model=model_name)[model_name]
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        target_layers = resolve_target_layers(model, model_name, config=config)
        reshape_transform = get_reshape_transform(model_name, backbone)
        target_layers_meta[model_name] = module_paths(model, target_layers)

        if args.print_target_layers:
            for layer in target_layers:
                print(f"{model_name}: {type(layer).__name__} at {module_paths(model, [layer])}")
            continue

        model_dir = os.path.join(GRADCAM_ROOT, f"{model_name}__{sanitize(backbone)}")

        # Instantiate each CAM object once per model.
        cams = {}
        for method in methods:
            cam_class = _get_cam_class(method)
            cams[method] = cam_class(model=model, target_layers=target_layers,
                                     reshape_transform=reshape_transform)

        logger.info('Generating CAMs for %s (%d images)', model_name, limit)
        use_amp = device.type == 'cuda'
        for idx in tqdm(range(limit), desc=f"GradCAM {model_name}"):
            input_tensor, mask_tensor = test_dataset[idx]
            input_tensor = input_tensor.unsqueeze(0).to(device)
            gt_mapped = mask_tensor.cpu().numpy()
            rgb_image = unnormalize_image(input_tensor[0])

            base_name = os.path.splitext(test_images[idx])[0]

            for method in methods:
                method_dir = os.path.join(model_dir, method, 'test')
                needs_target = bool(METHODS[method]['needs_target'])

                if needs_target:
                    present = [c for c in classes_present_in_mask(gt_mapped) if c < num_classes]
                    for c in present:
                        class_name = sanitize(class_names[c] if c < len(class_names) else f'Class_{c}')
                        targets = [SemanticSegmentationTarget(
                            category=c, mask=(gt_mapped == c).astype(np.uint8))]
                        cam_map = _run_cam(cams[method], input_tensor, targets, use_amp)[0, :]
                        _save_cam(
                            cam_map, rgb_image,
                            os.path.join(method_dir, f"{base_name}__{class_name}__raw.png"),
                            os.path.join(method_dir, f"{base_name}__{class_name}__overlay.png"),
                        )
                else:
                    cam_map = _run_cam(cams[method], input_tensor, None, use_amp)[0, :]
                    _save_cam(
                        cam_map, rgb_image,
                        os.path.join(method_dir, f"{base_name}__raw.png"),
                        os.path.join(method_dir, f"{base_name}__overlay.png"),
                    )

                _release_cam_graph(cams[method])
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    if args.print_target_layers:
        return

    # ---- Config summary (shared across models) ----
    summary = build_config_summary(config, {
        'selected_models': selected_models,
        'num_classes': config.get('NUM_CLASSES') or num_label_classes,
        'class_names': class_names,
        'gradcam_methods': gradcam_methods_meta,
        'target_layers': target_layers_meta,
    })
    os.makedirs(GRADCAM_ROOT, exist_ok=True)
    with open(os.path.join(GRADCAM_ROOT, 'config_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('Wrote config summary to %s', os.path.join(GRADCAM_ROOT, 'config_summary.json'))


if __name__ == '__main__':
    main()
