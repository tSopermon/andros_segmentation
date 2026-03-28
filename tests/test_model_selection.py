import pytest

from utils.model_selection import (
    STANDARD_MODELS,
    get_active_original_models,
    get_selected_model_names,
    get_selected_standard_models,
)


def test_get_selected_standard_models_default_all_when_missing():
    config = {'MODEL_SET': 'standard'}
    assert get_selected_standard_models(config) == STANDARD_MODELS


def test_get_selected_standard_models_default_all_when_empty():
    config = {'MODEL_SET': 'standard', 'STANDARD_MODELS': []}
    assert get_selected_standard_models(config) == STANDARD_MODELS


def test_get_selected_standard_models_subset_with_dedup_preserves_order():
    config = {'STANDARD_MODELS': ['UNet', 'DeepLabV3', 'UNet']}
    assert get_selected_standard_models(config) == ['UNet', 'DeepLabV3']


def test_get_selected_standard_models_invalid_type_raises():
    config = {'STANDARD_MODELS': 'UNet'}
    with pytest.raises(RuntimeError, match='STANDARD_MODELS must be a YAML list'):
        get_selected_standard_models(config)


def test_get_selected_standard_models_invalid_name_raises():
    config = {'STANDARD_MODELS': ['UNet', 'NotAModel']}
    with pytest.raises(RuntimeError, match='Invalid STANDARD_MODELS entries'):
        get_selected_standard_models(config)


def test_get_active_original_models_from_flags():
    config = {
        'USE_UNET_ORIGINAL': True,
        'USE_DEEPLABV2_ORIGINAL': True,
        'USE_MAXVIT_UNET': True,
    }
    assert get_active_original_models(config) == [
        'UNet_original',
        'DeepLabV2_original',
        'MaxViTSmallUNet',
    ]


def test_get_selected_model_names_standard_uses_subset():
    config = {
        'MODEL_SET': 'standard',
        'STANDARD_MODELS': ['DeepLabV3Plus', 'UNetPlusPlus'],
        'USE_UNET_ORIGINAL': True,
    }
    assert get_selected_model_names(config) == ['DeepLabV3Plus', 'UNetPlusPlus']


def test_get_selected_model_names_all_combines_subset_and_originals():
    config = {
        'MODEL_SET': 'all',
        'STANDARD_MODELS': ['DeepLabV3', 'UNet'],
        'USE_UNET_ORIGINAL': True,
        'USE_DEEPLABV1_ORIGINAL': True,
    }
    assert get_selected_model_names(config) == [
        'DeepLabV3',
        'UNet',
        'UNet_original',
        'DeepLabV1_original',
    ]


def test_get_selected_model_names_invalid_model_set_raises():
    config = {'MODEL_SET': 'unknown'}
    with pytest.raises(RuntimeError, match='Unknown MODEL_SET'):
        get_selected_model_names(config)
