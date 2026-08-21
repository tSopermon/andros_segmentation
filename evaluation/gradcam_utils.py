"""Utilities for Grad-CAM / XAI visualization of segmentation models.

This module holds the CAM method registry, target-layer resolution, filename
sanitization, config-metadata building, and class-presence helpers. It is kept
free of a module-level ``pytorch_grad_cam`` import so that the pure helpers
remain importable and unit-testable in environments where the optional
``grad-cam`` dependency is not installed; the CAM classes are resolved lazily
by ``gradcam.py`` via :data:`METHODS`.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CAM method registry
# ---------------------------------------------------------------------------
# ``import_name`` is the attribute name on the ``pytorch_grad_cam`` package.
# The class object is imported lazily in ``gradcam.py``.
METHODS: Dict[str, Dict[str, object]] = {
    'GradCAM': {'import_name': 'GradCAM', 'needs_target': True},
    'SegEigenCAM': {'import_name': 'SegEigenCAM', 'needs_target': True},
    'LayerCAM': {'import_name': 'LayerCAM', 'needs_target': True},
    'EigenCAM': {'import_name': 'EigenCAM', 'needs_target': False},
    'HiResCAM': {'import_name': 'HiResCAM', 'needs_target': True},
}

DEFAULT_METHODS: List[str] = list(METHODS.keys())

SMP_MODEL_NAMES = {'DeepLabV3', 'DeepLabV3Plus', 'UNet', 'UNetPlusPlus', 'Segformer'}

# Candidate dotted paths tried in order for standard SMP encoders. The first
# path that resolves against the concrete backbone wins.
SMP_TARGET_LAYER_CANDIDATES: List[str] = [
    'encoder.layer4[-1]',                # ResNet-family SMP encoders
    'encoder.model.stages_3.blocks[-1]',  # timm 'tu-' encoders (MaxViT/EfficientNet/...)
    'encoder.model.blocks[-1]',          # timm encoders exposing blocks directly
    'encoder.blocks[-1]',                # generic last block
    'encoder.layer4',                    # last stage without a block index
]

# Per-model default candidate paths for the original / custom implementations.
DEFAULT_TARGET_LAYERS: Dict[str, List[str]] = {
    'UNet_original': ['middle_conv.second', 'up_conv[-1].second'],
    'DeepLabV1_original': ['conv7', 'conv6', 'block_5'],
    'DeepLabV2_original': ['layer4[-1]', 'layer3[-1]'],
    'DeepLabV3_original': ['base.classifier[-1]', 'base.backbone.layer4[-1]'],
    'MaxViTSmallUNet': ['up1.conv2', 'up1.conv1'],
}

_SEGMENT_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)(?:\[(-?\d+)\])?$')
_FEATURE_CLASS_RE = re.compile(r'(Conv|Block|Stage)')

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def available_methods() -> List[str]:
    """Return the ordered list of supported CAM method names."""
    return list(METHODS.keys())


def needs_target(method: str) -> bool:
    """Return True if ``method`` is class-discriminative and requires a target."""
    if method not in METHODS:
        raise KeyError(f"Unknown CAM method '{method}'. Valid: {list(METHODS.keys())}")
    return bool(METHODS[method]['needs_target'])


def _resolve_path(root: nn.Module, path: str) -> Optional[nn.Module]:
    """Walk a dotted attribute path (with optional ``[index]`` segments) from ``root``.

    Returns the resolved ``nn.Module`` or ``None`` if any segment fails.
    """
    obj = root
    for segment in str(path).split('.'):
        segment = segment.strip()
        match = _SEGMENT_RE.match(segment)
        if not match:
            return None
        name, index = match.group(1), match.group(2)
        try:
            obj = getattr(obj, name)
        except AttributeError:
            return None
        if index is not None:
            try:
                obj = obj[int(index)]
            except (IndexError, KeyError, TypeError):
                return None
    return obj if isinstance(obj, nn.Module) else None


def _deepest_matching_module(root: nn.Module, pattern) -> Optional[nn.Module]:
    """Return the deepest module under ``root`` whose class name matches ``pattern``."""
    best: Optional[nn.Module] = None
    best_depth = -1

    def walk(module: nn.Module, depth: int) -> None:
        nonlocal best, best_depth
        for child in module.children():
            child_depth = depth + 1
            if pattern.search(type(child).__name__) and child_depth > best_depth:
                best_depth = child_depth
                best = child
            walk(child, child_depth)

    walk(root, 0)
    return best


def resolve_target_layers(
    model: nn.Module,
    model_name: str,
    config: Optional[Dict] = None,
) -> List[nn.Module]:
    """Resolve the CAM target layer(s) for ``model``.

    Resolution order:
      1. ``config['GRADCAM_TARGET_LAYERS'][model_name]`` override (str or list of str).
      2. :data:`DEFAULT_TARGET_LAYERS` for original implementations.
      3. :data:`SMP_TARGET_LAYER_CANDIDATES` for SMP models.
      4. Fallback: deepest module under ``model.encoder`` (or ``model``) whose
         class name contains ``Conv``/``Block``/``Stage``.
    """
    candidates: List[str] = []

    override = None
    if config:
        override_map = config.get('GRADCAM_TARGET_LAYERS', None)
        if isinstance(override_map, dict) and model_name in override_map:
            override = override_map[model_name]

    if override is not None:
        if isinstance(override, (list, tuple)):
            candidates = [str(p) for p in override]
        else:
            candidates = [str(override)]
    elif model_name in DEFAULT_TARGET_LAYERS:
        candidates = list(DEFAULT_TARGET_LAYERS[model_name])
    else:
        candidates = list(SMP_TARGET_LAYER_CANDIDATES)

    for path in candidates:
        module = _resolve_path(model, path)
        if module is not None:
            logger.info(
                "Resolved target layer for %s via path '%s' -> %s",
                model_name, path, type(module).__name__,
            )
            return [module]

    root = model.encoder if hasattr(model, 'encoder') else model
    fallback = _deepest_matching_module(root, _FEATURE_CLASS_RE)
    if fallback is not None:
        logger.warning(
            "No explicit target layer resolved for %s; using deepest feature "
            "module %s (class '%s')",
            model_name, type(fallback).__name__, type(fallback).__name__,
        )
        return [fallback]

    raise RuntimeError(f"Could not resolve a target layer for model '{model_name}'.")


def get_reshape_transform(model_name: str, backbone: Optional[str]):
    """Return a reshape transform for token-sequence encoders, else ``None``.

    SMP encoders (both ResNet-family and ``tu-`` timm encoders) emit 2D feature
    maps at every stage, so their conv/stage target layers need no token ->
    spatial reshape. Only a bare ViT/Swin-style token encoder would need
    ``vit_reshape_transform`` / ``swinT_reshape_transform``.
    """
    if model_name in SMP_MODEL_NAMES:
        return None

    backbone_l = (backbone or '').lower()
    if 'swin' in backbone_l:
        try:
            from pytorch_grad_cam.utils.reshape_transforms import swinT_reshape_transform
            return swinT_reshape_transform
        except Exception:
            return None
    if any(k in backbone_l for k in ('vit', 'maxvit', 'mit', 'segformer', 'beit', 'deit')):
        try:
            from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
            return vit_reshape_transform
        except Exception:
            return None
    return None


def sanitize(name: str) -> str:
    """Normalize a name for use in filesystem paths and filenames.

    Lowercases, strips the ``tu-`` timm prefix, and collapses ``-``, ``.``,
    ``/``, ``\\`` and whitespace runs into single underscores.
    """
    s = str(name).strip().lower()
    if s.startswith('tu-'):
        s = s[3:]
    s = re.sub(r'[\s\-\./\\]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s


def classes_present_in_mask(gt_mapped) -> List[int]:
    """Return the sorted class indices present in a (mapped) ground-truth mask."""
    arr = np.asarray(gt_mapped)
    return sorted(int(v) for v in np.unique(arr).tolist())


def module_paths(model: nn.Module, modules: Sequence[nn.Module]) -> List[str]:
    """Return the dotted ``named_modules`` paths for the given modules."""
    path_by_id = {id(m): name for name, m in model.named_modules()}
    paths = []
    for module in modules:
        path = path_by_id.get(id(module))
        if path is not None:
            paths.append(path)
    return paths


def build_config_summary(config: Dict, resolved: Dict) -> Dict:
    """Build the ``config_summary.json`` payload from the config and resolved info.

    ``resolved`` is expected to carry: ``selected_models``, ``class_names``,
    ``gradcam_methods``, and ``target_layers`` (dict model_name -> list[str]).
    """
    return {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'dataset_path': config.get('DATASET_PATH'),
        'model_set': config.get('MODEL_SET'),
        'selected_models': resolved.get('selected_models'),
        'backbone': config.get('BACKBONE'),
        'encoder_weights': config.get('ENCODER_WEIGHTS'),
        'num_classes': resolved.get('num_classes', config.get('NUM_CLASSES')),
        'class_names': resolved.get('class_names'),
        'image_size': config.get('IMAGE_SIZE'),
        'batch_size': config.get('BATCH_SIZE'),
        'optimizer': config.get('OPTIMIZER'),
        'learning_rate': config.get('LEARNING_RATE'),
        'lr_decay_gamma': config.get('LR_DECAY_GAMMA'),
        'max_epochs': config.get('MAX_EPOCHS'),
        'min_epochs': config.get('MIN_EPOCHS'),
        'loss_function': config.get('LOSS_FUNCTION'),
        'dice_weight': config.get('DICE_WEIGHT'),
        'focal_weight': config.get('FOCAL_WEIGHT'),
        'use_augmentation': config.get('USE_AUGMENTATION'),
        'transfer_learning': config.get('TRANSFER_LEARNING'),
        'freeze_encoder': config.get('FREEZE_ENCODER'),
        'self_training': config.get('SELF_TRAINING'),
        'pseudo_label_threshold': config.get('PSEUDO_LABEL_THRESHOLD'),
        'k_folds': config.get('K_FOLDS'),
        'ensemble': config.get('ENSEMBLE'),
        'gradcam_methods': resolved.get('gradcam_methods'),
        'target_layers': resolved.get('target_layers'),
    }


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization on a ``(3, H, W)`` tensor.

    Returns an RGB ``float32`` array in ``[0, 1]`` with shape ``(H, W, 3)``,
    suitable for ``show_cam_on_image``.
    """
    t = tensor.detach().cpu()
    rgb = t.permute(1, 2, 0).numpy().astype(np.float32)
    rgb = rgb * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(rgb, 0.0, 1.0)
