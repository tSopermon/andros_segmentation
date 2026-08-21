import numpy as np
import pytest
import torch
from torch import nn

from evaluation.gradcam_utils import (
    DEFAULT_METHODS,
    METHODS,
    build_config_summary,
    classes_present_in_mask,
    needs_target,
    resolve_target_layers,
    sanitize,
)


def test_sanitize_lowercases_and_collapses_separators():
    assert sanitize('tu-maxvit_large_tf_512') == 'maxvit_large_tf_512'
    assert sanitize('UNetPlusPlus') == 'unetplusplus'
    assert sanitize('Arable Land') == 'arable_land'
    assert sanitize('DeepLab/V3.Plus-extra') == 'deeplab_v3_plus_extra'


def test_sanitize_handles_empty_and_non_string():
    assert sanitize('') == ''
    assert sanitize('  tu-  ') == ''


def test_classes_present_in_mask_sorted_unique():
    mask = np.array([[0, 2], [2, 0]], dtype=np.uint8)
    assert classes_present_in_mask(mask) == [0, 2]
    mask2 = np.array([5])
    assert classes_present_in_mask(mask2) == [5]


def test_methods_registry_defaults():
    assert set(DEFAULT_METHODS) == {'GradCAM', 'SegEigenCAM', 'LayerCAM', 'EigenCAM', 'HiResCAM'}
    assert needs_target('EigenCAM') is False
    assert needs_target('GradCAM') is True
    assert needs_target('SegEigenCAM') is True
    assert needs_target('LayerCAM') is True
    assert needs_target('HiResCAM') is True
    with pytest.raises(KeyError):
        needs_target('NotAMethod')


def test_build_config_summary_fields():
    config = {
        'DATASET_PATH': '/data', 'MODEL_SET': 'standard', 'BACKBONE': 'resnet101',
        'ENCODER_WEIGHTS': 'imagenet', 'NUM_CLASSES': 8, 'IMAGE_SIZE': 512,
        'BATCH_SIZE': 2, 'OPTIMIZER': 'AdamW', 'LEARNING_RATE': 0.0001,
        'LR_DECAY_GAMMA': 0.95, 'MAX_EPOCHS': 200, 'MIN_EPOCHS': 20,
        'LOSS_FUNCTION': 'DiceFocal', 'DICE_WEIGHT': 1.0, 'FOCAL_WEIGHT': 1.0,
        'USE_AUGMENTATION': False, 'TRANSFER_LEARNING': False, 'FREEZE_ENCODER': False,
        'SELF_TRAINING': False, 'PSEUDO_LABEL_THRESHOLD': 0.85,
        'K_FOLDS': 1, 'ENSEMBLE': False,
    }
    resolved = {
        'selected_models': ['UNetPlusPlus'],
        'class_names': ['Water', 'Frygana'],
        'gradcam_methods': ['EigenCAM'],
        'target_layers': {'UNetPlusPlus': ['encoder.layer4[-1]']},
    }
    summary = build_config_summary(config, resolved)

    assert summary['dataset_path'] == '/data'
    assert summary['backbone'] == 'resnet101'
    assert summary['selected_models'] == ['UNetPlusPlus']
    assert summary['class_names'] == ['Water', 'Frygana']
    assert summary['gradcam_methods'] == ['EigenCAM']
    assert summary['target_layers'] == {'UNetPlusPlus': ['encoder.layer4[-1]']}
    assert 'timestamp' in summary

    expected_keys = {
        'timestamp', 'dataset_path', 'model_set', 'selected_models', 'backbone',
        'encoder_weights', 'num_classes', 'class_names', 'image_size', 'batch_size',
        'optimizer', 'learning_rate', 'lr_decay_gamma', 'max_epochs', 'min_epochs',
        'loss_function', 'dice_weight', 'focal_weight', 'use_augmentation',
        'transfer_learning', 'freeze_encoder', 'self_training',
        'pseudo_label_threshold', 'k_folds', 'ensemble', 'gradcam_methods', 'target_layers',
    }
    assert expected_keys <= set(summary.keys())


class _DummyWithSecond(nn.Module):
    def __init__(self):
        super().__init__()
        self.second = nn.Conv2d(4, 4, 1)


class _UNetOriginalLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.middle_conv = _DummyWithSecond()


class _EncoderLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 1))


class _SMPLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _EncoderLike()


class _NoDefaultPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))


def test_resolve_target_layers_override_via_dotted_path():
    model = _SMPLike()
    config = {'GRADCAM_TARGET_LAYERS': {'MyModel': 'encoder.layer4[1]'}}
    layers = resolve_target_layers(model, 'MyModel', config=config)
    assert len(layers) == 1
    assert isinstance(layers[0], nn.Conv2d)


def test_resolve_target_layers_default_table_original():
    model = _UNetOriginalLike()
    layers = resolve_target_layers(model, 'UNet_original')
    assert len(layers) == 1
    assert isinstance(layers[0], nn.Conv2d)


def test_resolve_target_layers_smp_candidate():
    model = _SMPLike()
    layers = resolve_target_layers(model, 'UNetPlusPlus')
    assert len(layers) == 1
    assert isinstance(layers[0], nn.Conv2d)


def test_resolve_target_layers_fallback_deepest_feature_module():
    model = _NoDefaultPath()
    layers = resolve_target_layers(model, 'SomeUnknownModel')
    assert len(layers) == 1
    assert isinstance(layers[0], nn.Conv2d)


def test_resolve_target_layers_raises_when_no_module():
    empty = nn.Module()
    with pytest.raises(RuntimeError, match='Could not resolve a target layer'):
        resolve_target_layers(empty, 'NoEncoderModel')
