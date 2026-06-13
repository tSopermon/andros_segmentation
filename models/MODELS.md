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

### Backbone and Decoder Compatibility (Timm Encoders)

When using Timm encoders (prefix `tu-`), specific rules apply depending on your chosen Decoder and Input Image Resolution.

#### 1. U-Net Family (UNet, UNetPlusPlus) - The Feature Scale Rule
- **Rule:** You **MUST** use encoders from the [Traditional-Style list](https://smp.readthedocs.io/en/latest/encoders_timm.html#traditional-style).
- **Why:** U-Net variants mathematically require a dense feature pyramid at every scale (1/2, 1/4, 1/8, 1/16, 1/32). "Transformer-Style" encoders (like Swin, SAM2, ConvNeXt) typically skip the 1/2 scale (stride 2), outputting `0` channels at that stage. This missing scale completely breaks U-Net skip connections, causing a `RuntimeError` during convolution initialization.
- **Compatible:** `ResNet`, `EfficientNet`, `MobileNet`, `MaxVit` (MaxVit is Traditional-Style because it has a full CNN stem).
- **Incompatible:** `Swin`, `SAM2`, `ConvNeXt`, `DaViT`.

#### 2. DeepLab Family (DeepLabV3, DeepLabV3+) - The Dilation Rule
- **Rule:** You **MUST** use encoders that support dilation (marked with a `✅` in the SMP docs).
- **Why:** DeepLab models rely heavily on Atrous Spatial Pyramid Pooling (ASPP). To avoid shrinking the image to an unusable size, DeepLab forcibly overrides the encoder's `output_stride` (to 8 or 16). If the backbone lacks structural support for this override (like many fixed-stride Vision Transformers), it will crash with a `TypeError: unexpected keyword argument 'output_stride'`.
- **Compatible:** `ResNet`, `EfficientNet`, `MobileNet`, `ConvNeXt` (ConvNeXt is a rare transformer that supports dilation).
- **Incompatible:** `MaxVit`, `Swin`, `SAM2`.

#### 3. Image Resolution Constraints (256px vs 512px)
- **Fixed-Window Models:** Certain modern models (like `MaxVit` or `Swin`) use fixed-size attention windows. Their required resolution is usually baked directly into their name (e.g., `tu-maxvit_large_tf_512` or `tu-swinv2_base_window16_256`). If you pass 512px images to a `_256` model, it crashes instantly with an `AssertionError`.
- **Fully Convolutional Models:** Standard CNNs (e.g., `tu-resnet101`, `tu-efficientnet_b0`) dynamically adapt to the spatial dimensions of your input and can be seamlessly swapped between 256px and 512px setups.

#### Compatibility Summary Matrix
| Backbone Type | Works with UNet / UNet++ ? | Works with DeepLabV3 / V3+ ? | Resolution Handling |
| :--- | :---: | :---: | :--- |
| **ResNet / EfficientNet** | ✅ Yes | ✅ Yes | Flexible (Accepts 256px or 512px) |
| **MaxVit** (e.g. `_512`) | ✅ Yes | ❌ No | Strict (Must match name suffix) |
| **Swin / SAM2** | ❌ No | ❌ No | Strict (Must match name suffix) |
| **ConvNeXt** | ❌ No | ✅ Yes | Flexible |

---

## Original Implementations
These are native PyTorch implementations or wrappers around official Torchvision models, focused on reproducing specific paper architectures.

### 1. UNet_original
- **Source:** Local implementation (`models/unet_original.py`) adapted from [nn.labml.ai](https://nn.labml.ai/unet/index.html).
- **Description:** A faithful implementation of the original U-Net paper "U-Net: Convolutional Networks for Biomedical Image Segmentation".
  - *Note:* `BatchNorm2d` has been injected between `Conv2d` and `ReLU` layers to stabilize and accelerate modern training, preventing vanishing gradients that plagued the original architecture without pretraining.
- **Config:** Fixed architecture; does not use the configurable backbone settings.

### 2. DeepLabV1_original
- **Source:** [rulixiang/deeplab-pytorch](https://github.com/rulixiang/deeplab-pytorch/blob/master/models/DeepLabV1_LargeFOV.py)
- **Description:** DeepLabV1 with Large Field-of-View (LargeFOV).
  - *Note:* Similar to UNet, `BatchNorm2d` layers have been added after convolutional layers to ensure stable training from random initialization.
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
