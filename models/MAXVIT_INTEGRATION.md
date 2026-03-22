# MaxViT-S U-Net Integration Guide

This document outlines the implementation details and system compatibility adjustments made to integrate the custom `MaxViTSmallUNet` architecture into the `andros_segmentation` pipeline.

## 1. Architecture Implementation

The `MaxViTSmallUNet` model utilizes a custom `timm` (PyTorch Image Models) MaxViT-S encoder combined with a standard CNN-based U-Net decoder.

* **File Location**: The core architecture and upsampling blocks were extracted into `models/maxvit_unet.py`.
* **Initialization**: The `__init__` signature was standardized to natively match the system's factory patterns: `__init__(self, in_channels: int = 3, out_channels: int = 7, pretrained: bool = True)`. Using the `in_channels` and `out_channels` standard ensures it initializes perfectly via config files.
* **Dependencies**: Added `timm` to `requirements.txt` to support the backbone retrieval. The model fails safely during tests if `timm` is missing.

## 2. Dynamic Batch Padding (Crucial Patch)

**The Problem:**
MaxViT uses windowed attention blocks with a $7\times7$ window size layered across a network downsampling stride of $32$. Therefore, it rigorously requires incoming input grid combinations to be cleanly divisible by $32 \times 7 = 224$ (e.g., $224$, $448$, etc.). Passing `IMAGE_SIZE: 256` or `512` physically breaks the matrix tensor partitions, raising `AssertionError: height must be divisible by window`.

**The Solution:**
Instead of restricting config values artificially to `224`, a silent wrapper handles mapping dimensions intelligently inside the `forward` pass:
1. Calculates spatial difference up to the nearest multiple of `224`.
2. Applies symmetric `reflect` padding to push the input batch size safely to valid ViT windows.
3. Passes features safely through both the encoder and decoder. 
4. Shrinks/Interpolates the network logits seamlessly back to the intended shape natively requested (`256x256`) before the loss function computes.

## 3. System Registration (`model_zoo`)

To make `MaxViTSmallUNet` natively usable:
1. Lookups were added to `get_models()` in `models/model_zoo.py`.
2. It's instantiated dynamically observing the `encoder_weights` flags (to optionally bypass network fetching).

## 4. Execution Scripts & Config

**YAML Control:**
The model defaults to `"originals"` inside the system. You manage standard toggle behavior explicitly inside `config/config.yaml`:
```yaml
USE_MAXVIT_UNET: true
MODEL_SET: originals # Allows MaxViT to trigger via the dedicated flag while leaving other originals False
```

**Training, Evaluation, Masking:**
`train.py`. `evaluate.py`, and `generate_masks.py` evaluate the config dictionary and append explicitly flagged models into the `ORIGINAL_MODELS` lists logic dynamically, preventing unwanted memory spikes while testing overlapping architectures. 

**Checkpoints**: The checkpoint generation dynamically maps the string layout to construct files labeled `MaxViTSmallUNet_best.pth`.