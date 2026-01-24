# Model Zoo Documentation

This document describes the segmentation models available in this repository, their implementation sources, and configuration details.

The system supports two categories of models: **Standard (SMP-backed)** and **Original Implementations**.

## Configuration

Model selection is controlled via the `MODEL_SET` option in `config/config.yaml`:
- `standard`: SMP-backed models.
- `originals`: Native PyTorch implementations of original papers.
- `all`: Both sets.

Common parameters:
- `BACKBONE`: Encoder backbone (e.g., `resnet101`, `resnet50`).
- `ENCODER_WEIGHTS`: Pretrained weights (e.g., `imagenet`).

---

## Standard Models
These models are provided by the [segmentation-models-pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch) library. They are highly optimized and support various backbones.

### Available Architectures
| Model | Description |
| :--- | :--- |
| **DeepLabV3** | DeepLabv3 with Atrous Spatial Pyramid Pooling (ASPP). |
| **DeepLabV3Plus** | DeepLabv3+ extending DeepLabv3 with a decoder module. |
| **UNet** | Classic U-Net architecture with encoder-decoder structure. |
| **UNetPlusPlus** | Nested U-Net architecture for improved feature aggregation. |

### Configuration
- **Backbone:** Configurable via `BACKBONE` config key. Use lighter backbones (e.g., `resnet18`) for speed or heavier ones (e.g., `resnet101`) for accuracy.
- **Weights:** Uses `ENCODER_WEIGHTS` (default: `imagenet`).

---

## Original Implementations
These are native PyTorch implementations or wrappers around official Torchvision models, focused on reproducing specific paper architectures.

### 1. UNet_original
- **Source:** Local implementation (`models/unet_original.py`) adapted from [nn.labml.ai](https://nn.labml.ai/unet/index.html).
- **Description:** A faithful implementation of the original U-Net paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".
- **Config:** Fixed architecture; does not use the configurable backbone settings.

### 2. DeepLabV1_original
- **Source:** [rulixiang/deeplab-pytorch](https://github.com/rulixiang/deeplab-pytorch/blob/master/models/DeepLabV1_LargeFOV.py)
- **Description:** DeepLabV1 with Large Field-of-View (LargeFOV).
- **Activation:** Requires environment variable `USE_DEEPLABV1_ORIGINAL=true` (handled automatically by `MODEL_SET: originals`).

### 3. DeepLabV2_original
- **Source:** Adapted from [kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/models/deeplabv2.py)
- **Description:** DeepLabV2 using Dilated ResNet and ASPP.
- **Config:**
    - Uses ResNet backbone (defaulting to `resnet50` if `resnet18` is requested, for compatibility).
    - Hardcoded `n_blocks=[3, 4, 23, 3]` and `atrous_rates=[6, 12, 18, 24]`.
- **Activation:** Requires environment variable `USE_DEEPLABV2_ORIGINAL=true`.

### 4. DeepLabV3_original
- **Source:** Wrapper around `torchvision.models.segmentation.deeplabv3_resnet50`.
- **Description:** Official PyTorch Hub DeepLabV3 with a ResNet50 backbone.
- **Weights:** Automatically downloads `DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1`.
- **Wrappers:**
    - Modified classifier layer to match the dataset's `NUM_CLASSES`.
    - Custom wrapper ensures output is a tensor (handles Torchvision's dictionary output) and enforces bilinear upsampling to input size.
- **Activation:** Requires environment variable `USE_DEEPLABV3_ORIGINAL=true`.
