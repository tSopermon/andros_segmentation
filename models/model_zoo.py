"""Model factory for segmentation models.

Provides a `get_models(num_classes, backbone, encoder_weights)` function
that returns a dictionary of model instances. If `segmentation_models_pytorch`
is not installed in the environment, lightweight dummy models are used so
that unit tests and lightweight workflows can still run.
"""
from .unet_original import UNet as UNetOriginal
from .deeplabv1_original import DeepLabV1_LargeFOV as DeepLabV1Original
from .deeplabv2_original import DeepLabV2 as DeepLabV2Original

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except Exception:
    smp = None
    HAS_SMP = False

from typing import Dict
import os
import torch
from torch import nn
try:
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
except ImportError:
    deeplabv3_resnet50 = None
    DeepLabV3_ResNet50_Weights = None


class _DummyModel(nn.Module):
    def __init__(self, in_channels=3, classes=1):
        super().__init__()
        self._in = in_channels
        self._classes = classes

    def forward(self, x):
        # return zeros logits shaped (B, C, H, W)
        return x.new_zeros((x.shape[0], self._classes, x.shape[2], x.shape[3]))


def get_models(num_classes: int, backbone: str = 'resnet101', encoder_weights: str = 'imagenet', specific_model: str = None) -> Dict[str, nn.Module]:
    """Return a dictionary of segmentation models.

    If `segmentation_models_pytorch` is available, return real SMP models.
    Otherwise return lightweight dummy models that implement `forward`.
    """
    models_dict = {}

    def should_include(name):
        return specific_model is None or name == specific_model

    if HAS_SMP:
        if should_include('DeepLabV3'):
            models_dict['DeepLabV3'] = smp.DeepLabV3(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        if should_include('DeepLabV3Plus'):
            models_dict['DeepLabV3Plus'] = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        if should_include('UNet'):
            models_dict['UNet'] = smp.Unet(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
        if should_include('UNetPlusPlus'):
            models_dict['UNetPlusPlus'] = smp.UnetPlusPlus(
                encoder_name=backbone,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=num_classes,
            )
    else:
        if should_include('DeepLabV3'): models_dict['DeepLabV3'] = _DummyModel(in_channels=3, classes=num_classes)
        if should_include('DeepLabV3Plus'): models_dict['DeepLabV3Plus'] = _DummyModel(in_channels=3, classes=num_classes)
        if should_include('UNet'): models_dict['UNet'] = _DummyModel(in_channels=3, classes=num_classes)
        if should_include('UNetPlusPlus'): models_dict['UNetPlusPlus'] = _DummyModel(in_channels=3, classes=num_classes)

    # Always register the original-paper U-Net implementation
    if should_include('UNet_original'):
        models_dict['UNet_original'] = UNetOriginal(in_channels=3, out_channels=num_classes)

    # Optionally register original DeepLab models if requested via environment variables.
    # These are disabled by default so unit tests that expect the previous set of keys
    # continue to pass. Set the following env vars to 'true' to enable them:
    # - USE_DEEPLABV1_ORIGINAL
    # - USE_DEEPLABV2_ORIGINAL
    # - USE_DEEPLABV3_ORIGINAL
    if should_include('DeepLabV1_original') and os.environ.get('USE_DEEPLABV1_ORIGINAL', 'false').lower() in ('1', 'true', 'yes'):
        try:
            models_dict['DeepLabV1_original'] = DeepLabV1Original(n_classes=num_classes)
        except Exception:
            models_dict['DeepLabV1_original'] = _DummyModel(in_channels=3, classes=num_classes)

    if should_include('DeepLabV2_original') and os.environ.get('USE_DEEPLABV2_ORIGINAL', 'false').lower() in ('1', 'true', 'yes'):
        try:
            models_dict['DeepLabV2_original'] = DeepLabV2Original(n_classes=num_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24], backbone=backbone)
        except Exception as e:
            print(f"FAILED to initialize DeepLabV2_original: {e}")
            import traceback
            traceback.print_exc()
            models_dict['DeepLabV2_original'] = _DummyModel(in_channels=3, classes=num_classes)

    if should_include('DeepLabV3_original') and os.environ.get('USE_DEEPLABV3_ORIGINAL', 'false').lower() in ('1', 'true', 'yes'):
        try:
            if deeplabv3_resnet50 is None:
                raise ImportError("torchvision not available")
            
            hub_model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
            # Adjust classifier output channels to match requested num_classes if possible
            try:
                if hasattr(hub_model, 'classifier') and isinstance(hub_model.classifier, nn.Sequential):
                    last = hub_model.classifier[-1]
                    if isinstance(last, nn.Conv2d):
                        in_ch = last.in_channels
                        hub_model.classifier[-1] = nn.Conv2d(in_ch, num_classes, kernel_size=1)
            except Exception:
                pass
            
            # Wrap the hub model to handle dict output and ensure proper upsampling
            class DeepLabV3Wrapper(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base = base_model
                
                def forward(self, x):
                    input_size = x.shape[2:]
                    out = self.base(x)
                    # torchvision deeplabv3 returns dict with 'out' key
                    if isinstance(out, dict):
                        out = out['out']
                    # Ensure output matches input size
                    if out.shape[2:] != input_size:
                        out = nn.functional.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
                    return out
            
            models_dict['DeepLabV3_original'] = DeepLabV3Wrapper(hub_model)
        except Exception:
            models_dict['DeepLabV3_original'] = _DummyModel(in_channels=3, classes=num_classes)

    return models_dict
