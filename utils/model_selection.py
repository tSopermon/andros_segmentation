from typing import Dict, List


STANDARD_MODELS = ['DeepLabV3', 'DeepLabV3Plus', 'UNet', 'UNetPlusPlus']


def get_selected_standard_models(config: Dict) -> List[str]:
    configured_models = config.get('STANDARD_MODELS', None)

    if configured_models is None or configured_models == []:
        return list(STANDARD_MODELS)

    if not isinstance(configured_models, list):
        raise RuntimeError('STANDARD_MODELS must be a YAML list of model names.')

    if any(not isinstance(model_name, str) for model_name in configured_models):
        raise RuntimeError('STANDARD_MODELS must contain only string model names.')

    selected_models = []
    for model_name in configured_models:
        if model_name not in selected_models:
            selected_models.append(model_name)

    invalid_models = [model_name for model_name in selected_models if model_name not in STANDARD_MODELS]
    if invalid_models:
        raise RuntimeError(
            f"Invalid STANDARD_MODELS entries: {invalid_models}. "
            f"Valid options are: {STANDARD_MODELS}."
        )

    return selected_models


def get_active_original_models(config: Dict) -> List[str]:
    active_originals = []
    if config.get('USE_UNET_ORIGINAL', False):
        active_originals.append('UNet_original')
    if config.get('USE_DEEPLABV1_ORIGINAL', False):
        active_originals.append('DeepLabV1_original')
    if config.get('USE_DEEPLABV2_ORIGINAL', False):
        active_originals.append('DeepLabV2_original')
    if config.get('USE_DEEPLABV3_ORIGINAL', False):
        active_originals.append('DeepLabV3_original')
    if config.get('USE_MAXVIT_UNET', False):
        active_originals.append('MaxViTSmallUNet')
    return active_originals


def get_selected_model_names(config: Dict) -> List[str]:
    model_set = config.get('MODEL_SET', 'standard')
    standard_models = get_selected_standard_models(config)
    original_models = get_active_original_models(config)

    if model_set == 'standard':
        return standard_models
    if model_set == 'originals':
        return original_models
    if model_set == 'all':
        return standard_models + original_models

    raise RuntimeError(
        f"Unknown MODEL_SET '{model_set}' in config; expected 'standard', 'originals', or 'all'."
    )
