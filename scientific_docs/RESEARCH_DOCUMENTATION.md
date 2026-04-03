# Andros Segmentation: Research Documentation

## 1. Executive Summary
The Andros Segmentation system is a highly advanced, reproducible deep learning pipeline designed to perform multi-class semantic segmentation. It is architected for seamless transitions across various learning paradigms—from traditional supervised learning to state-of-the-art Self-Supervised Learning (SSL) and Semi-Supervised Self-Training. The system mitigates severe overfitting on small datasets via a robust, multi-stage methodology and highly configurable loss objectives, using PyTorch and `segmentation_models_pytorch` as the core framework.

This documentation serves as a comprehensive overview of the theoretical and architectural design decisions implemented in the repository, explicitly mapping the scientific foundations of the models and methodologies to their physical code representations within the system.

---

## 2. Model Architectures & Encoders

The system employs a factory pattern (`models/model_zoo.py`) orchestrating two main groups of models, driven by the `MODEL_SET` config parameter: **Standard** (SMP-backed) and **Originals** (Native implementations).

### 2.1 Standard Architectures (SMP-backed)
These dynamic models utilize the `segmentation-models-pytorch` (SMP) library, allowing the arbitrary interchange of encoder backbones.

*   **UNet & UNetPlusPlus**: 
    *   `UNet`: The standard U-shaped encoder-decoder network featuring skip connections to preserve spatial details across convolutions.
    *   `UNet++`: An advanced nested architecture featuring densely connected skip pathways, improving the fusion of semantic and spatial features.
*   **DeepLabV3 & DeepLabV3Plus**:
    *   `DeepLabV3`: Leverages Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context by applying dilated convolutions at multiple rates.
    *   `DeepLabV3+`: Extends DeepLabV3 with a powerful decoder module to refine object boundaries, typically yielding superior results on distinct boundary segmentation.

### 2.2 Original Implementations: Scientific Foundations
Faithful native-PyTorch reconstructions or explicitly wrapped Torchvision instances for reproducing specific foundational research papers:

*   **`UNet_original`**:
    *   *Citation*: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015).
    *   *Implementation* (`models/unet_original.py`): Explicitly mimics the unpadded, fully convolutional network described in the paper. It symmetrically defines the contracting path using `DoubleConvolution` loops mapping to `MaxPool2d` (`DownSample`). The expansive path correctly executes spatial doubling using `ConvTranspose2d` (`UpSample`) followed strictly by a `CropAndConcat` mechanism. This crops the deeply encoded tensor arrays to identically match the upsampled spatial tensor grids prior to dimension combination, natively aligning high-frequency features.
*   **`DeepLabV1_original`**:
    *   *Citation*: Chen et al., "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs" (ICLR 2015).
    *   *Implementation* (`models/deeplabv1_original.py`): This implements the Large Field-of-View (LargeFOV) adaptation inside `Block`. It intentionally modifies standard convolutional parameters by enforcing `dilation=12` on the terminal blocks, bypassing the standard spatial reduction poolings to explicitly preserve higher-resolution outputs against the cost of the structural receptive field.
*   **`DeepLabV2_original`**:
    *   *Citation*: Chen et al., "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs" (TPAMI 2017).
    *   *Implementation* (`models/deeplabv2_original.py`): Re-structures the ResNet topology via TorchVision weights. It overrides pooling mechanics by passing `replace_stride_with_dilation=[False, True, True]` through the core feature extractor, enforcing an `Output Stride` of 8. The terminal layers intersect with a heavily modified `_ASPP` (Atrous Spatial Pyramid Pooling) layer branching across custom sparse rates (6, 12, 18, 24) to holistically unify multiple scales.
*   **`MaxViTSmallUNet`**:
    *   *Implementation* (`models/maxvit_unet.py`): A hybrid fusion structure integrating cutting-edge Vision Transformers inside classical convolutions. It requests the internal architecture of MaxViT (`maxvit_small_tf_224`) through the `timm` library with `features_only=True`. A `ConvBNAct` decoding bridge manually intersects with hierarchical stage outputs, reconstructing spatial mappings via chained bilinear `UpBlock` interpolation paths.

### 2.3 Encoders & Backbones
The pipeline uses interchangeable encoders, decoupling the feature extractor from the decoder head.
*   **Configured Backbone**: The default is `se_resnext50_32x4d`. It introduces Squeeze-and-Excitation (SE) blocks on top of the ResNeXt topology, actively re-calibrating channel-wise feature responses and providing superior representational power with minimal parameter overhead.

---

## 3. Objective Functions (Loss Mechanisms)

The learning objective is locally stored in `training/losses.py`, engineered to tackle complex real-world data distributions such as extreme class imbalance.

*   **CrossEntropy Loss**: Calculates pixel-wise negative log-likelihood. Standard for multi-class classification but suffers significantly under heavy class imbalance (e.g., massive background areas overpowering distinct small objects).
*   **Dice Loss**: 
    *   *Citation*: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (3DV 2016).
    *   *Theory & Implementation*: Optimizes the overlap between network predictions and geometric ground truth. Rather than calculating per-pixel classification accuracy, Dice Loss flattens predictions structurally into a 1-dimensional array across the batch and calculates: `(2. * Intersection + smooth) / (Union + smooth)`. It completely insulates training gradients from class-ratio imbalances because the overlapping intersections of small objects are equally penalized proportional to their overall mass.
*   **Focal Loss**: 
    *   *Citation*: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).
    *   *Theory & Implementation*: A highly specialized iteration of standard CrossEntropy natively embedded inside standard PyTorch mechanics. The formula calculates the raw pure probability (`pt`) exponentially tracking standard negative losses (`torch.exp(-ce_loss_unweighted)`). The Focal equation acts mathematically: `((1 - pt) ** gamma) * ce_loss`. Easy-to-classify broad pixels correctly estimated near probability ~1 aggressively drop to 0 loss, compelling gradient momentum back onto incredibly narrow, rigid boundary contours.
*   **DiceBCE**: A hybrid combination summing Dice Loss and CrossEntropy, stabilizing the unstable converging gradients of pure Dice Loss with the smoothing convexity of CrossEntropy.
*   **DiceFocal (Default)**: Combines the overlap-optimization properties of Dice with the hard-example mining of Focal Loss via an elegant mathematical sum: `(dice_w * Dice) + (focal_w * Focal)`.

---

## 4. Methodologies & Training Workflows

A defining trait of this system is its robust guardrails against **catastrophic forgetting** and extreme overfitting when dealing with limited labeled patches. It does this via a structured **3-Phase + 1-Refinement** training methodology.

### 4.1 Phase 1: Pre-Training / Self-Predictive Learning (MAE)
Implemented in `pretrain.py`, this acts as a Self-Supervised Learning (SSL) phase structurally inspired by classic Masked Autoencoders (MAE).
*   **Objective**: Reconstruct raw RGB pixels over heavily obscured continuous dimensions.
*   **Theory & Implementation**: `training/masking_utils.py` contains the core methodology logic:
    *   *Random Mask Generation*: It initiates mathematically by segmenting the spatial dimension into rigid `patch_size` blocks. Gaussian noise is populated across an unrolled dimensional matrix, mapping purely via a `topk` mechanism retaining only `MASK_RATIO` (e.g. 75%) arrays flagged identically as True/False Boolean targets. 
    *   *Object-Centric Mask Evolution*: Activated by `OBJECT_CENTRIC_EPOCH`, the masking shifts theoretically into Self-Guided Heuristics. The tensors actively execute convolutions simulating 3x3 `Sobel_X` & `Sobel_Y` edge-magnitude filters. It internally applies max-pooling calculations averaging the resultant gradient edges directly over the patch block. This mathematically builds a high-tier structural density map that dynamically weights the original random noise matrix, aggressively prioritizing occlusion mapping over complex object geometry instead of empty skies or flat waters.

### 4.2 Phase 2: Head Swap & Transfer Learning (Frozen)
*   *Mechanics*: A purely structural logic pass that intersects incompatible topologies across continuous training. `TRANSFER_LEARNING: true` signals the framework to pull Stage 1 arrays (often 3-channel RGB regeneration outputs), intentionally unhooking and dropping incompatible final mapping layer classifiers natively. 
*   *Implementation*: A boolean `requires_grad=False` is injected into traversing layers protecting the robust, globally mapped Stage 1 representations. The 8-class prediction head generates chaotic initialization mappings but safely optimizes against an entirely frozen baseline encoder.

### 4.3 Phase 3: Full-Network Unfrozen Fine-Tuning
*   *Differential Optimizations*: A cascading adjustment of state. With `FREEZE_ENCODER: false` executing natively across the unhooked tensor mappings, gradients resume flowing smoothly across the encoder. However, mathematically, they execute identically at approximately 10x-100x slower updates (e.g., `1e-5` simulated parameters), ensuring they smoothly attach the updated head with negligible semantic variation, finalizing structural memory.

### 4.4 Phase 4: Semi-Supervised Learning (Self-Training / Pseudo-Labeling)
Engineered structurally into a dual-streaming logic paradigm, the pipeline acts autonomously to refine boundaries leveraging unlabeled datasets (`DualStreamDataset`). 
*   *Citations*: 
    *   Lee, "Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks" (ICML W, 2013).
    *   Xie et al., "Self-training with Noisy Student improves ImageNet classification" (CVPR 2020) regarding dual varying augmentations algorithms.
*   *Theoretical Concept & Execution*:
    *   **Teacher-Student Architecture**: The trained Phase 3 model assumes the identity of a Frozen "Teacher" while a parallel, active "Student" model attempts learning.
    *   **Confidence Filtering (`IGNORE_INDEX`)**: The Teacher propagates an initial classification guess over completely unlabeled pixels. To prevent catastrophic ingestion of faulty inferences, a rigorous mathematical clip passes the distribution through a `PSEUDO_LABEL_THRESHOLD` max classification. Crucially, any mapped pixel under confidence (e.g. `0.90`) mathematically drops its index signature cleanly to the specific `IGNORE_INDEX` value (`-1`), silently instructing the PyTorch optimization core backwards pass to drop any calculation penalty regarding that single pixel axis entirely.
    *   **Assymetric Augmentation**: Adhering to the "Noisy Student" framework, the `DualStreamDataset` explicitly diverges tensor arrays into weak augmentations providing rigid context mapping identically back to the Teacher, alongside a strongly-augmented path systematically forcing invariant structural bounds into the Student.

---

## 5. End-to-End System Components

*   **Mixed Precision**: Accelerates compute dynamically utilizing Automatic Mixed Precision (`torch.amp`), halving VRAM requirements.
*   **Stratified K-Fold Cross Validation**: Bypasses typical sub-optimal arbitrary Validation dataset splitting. Computes dataset-level pixel proportions, structurally enforcing an even class distribution across folds algorithmically.
*   **Metric Computations**: Reports Macro (Mean), Micro, and inherently Weighted averages spanning Precision, Recall, F1, and IoU to comprehensively display accurate model behaviors outside pure uniform distribution scenarios. 
*   **Deterministic Reproducibility**: Absolute global initialization of `random`, `numpy`, `torch` and `cuda` stochastic seeds, neutralizing non-deterministic run variations.
