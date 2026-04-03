# Technical and Scientific Models Documentation

This document serves as the master technical architecture reference detailing the inner mechanics and foundational origins of the eight distinct semantic segmentation algorithms instantiated across the Andros Segmentation system. 

It specifically breaks down the mathematical structures and engineering compromises bridging the original scientific publications with their physical instantiations in this repository.

---

## Part 1: Original Implemented Architectures (Native PyTorch)

These models strictly represent faithful PyTorch layer constructions intended to mirror foundational scientific papers.

### 1. UNet (Original)
*   **Scientific Origin**: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., MICCAI 2015).
*   **Implementation Source**: `models/unet_original.py`, adapted from [nn.labml.ai](https://nn.labml.ai/unet/index.html).
*   **Technical Mechanics**:
    *   **Architecture**: Symmetrical "U" shape mapping an expanding decoder perfectly inverse to a contracting encoder. 
    *   **Contracting Path (Encoder)**: Exclusively utilizes 3x3 `DoubleConvolution` sets followed by a non-linear ReLU mapping. Crucially, in this strict implementation, the convolutions are formulated explicitly **without** padding (`padding=1`), though our specific implementation sets `padding=1` on the `nn.Conv2d` to handle our 256x256 image arrays seamlessly. The tensor is downsampled sequentially using rigid 2x2 `MaxPool2D` operators (`DownSample`).
    *   **Expansive Path (Decoder)**: Feature map upsampling is driven exclusively via transposed convolutions (`nn.ConvTranspose2d`) executing a 2x2 kernel with a stride of 2 (`UpSample`), ensuring fully trainable interpolation rather than naive bilinear interpolation.
    *   **Skip Connections**: The critical UNet element—the skip pathways branching from encoder to decoder—are intercepted by the internal `CropAndConcat` class. Because our implementation handles standardized input geometries, `torchvision.transforms.functional.center_crop` programmatically ensures spatial bounding identical to the corresponding upscale tensor prior to channel-wise concatenation across $dim=1$.

### 2. DeepLabV1 (LargeFOV)
*   **Scientific Origin**: "Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs" (Chen et al., ICLR 2015).
*   **Implementation Source**: `models/deeplabv1_original.py`, explicitly mimicking [rulixiang/deeplab-pytorch](https://github.com/rulixiang/deeplab-pytorch/blob/master/models/DeepLabV1_LargeFOV.py).
*   **Technical Mechanics**:
    *   **Large Field-of-View**: This is specifically the "DeepLab-LargeFOV" V1 iteration, famously modifying standard VGG16 parameters to exponentially increase the spatial receptive field without parameter ballooning.
    *   **Atrous Kernels**: Up to `block_5`, the network implements conventional chained 3x3 Convolutions and ReLUs separated by MaxPool striding pools. However, upon reaching the terminal `conv6` layer, pooling mechanics are explicitly bypassed. Instead, the model initiates an open `dilation=12` kernel matrix combined synchronously with `padding=12`.
    *   **Spatial Reconstitution**: The raw probability matrices omit dense logic pathways, passing `conv6` and `conv7` through heavy `.5` dropouts before interpolating via bilinear alignments specifically restoring the final output matrix scale to the exact original $(H, W)$ input space.

### 3. DeepLabV2 (Dilated ResNet + ASPP)
*   **Scientific Origin**: "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs" (Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille, TPAMI 2017).
*   **Implementation Source**: `models/deeplabv2_original.py`, originating via [kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch).
*   **Technical Mechanics**:
    *   **Backbone Subversion**: Retires VGG topologies substituting modern, intensely deep `torchvision.models` instances (ResNet50 / ResNet101).
    *   **Output Stride Rigidification**: DeepLabs inherently suffer when extensive pooling decimates high-frequency localized features. Here, the implementation explicitly manipulates the PyTorch native weights API, forcing `replace_stride_with_dilation=[False, True, True]` directly into the ResNet logic gates `layer3` and `layer4`. This rigidly caps the "Output Stride" of the feature extractor globally at 8. 
    *   **ASPP Layer Integration**: The dense localized layer representations are routed symmetrically into the internal `_ASPP` (Atrous Spatial Pyramid Pooling) layer. ASPP runs four mathematically parallel `nn.Conv2d` blocks executing fixed atrous rates natively `[6, 12, 18, 24]`. Integrating these massive dilations captures varied holistic scales concurrently without adding pooling decimation, before summating their spatial results dynamically.

### 4. MaxViT-UNet
*   **Scientific Origin**: Hybrid architectural mapping combining hierarchical Vision Transformers with typical CNN spatial topologies (Originating PRLAB21 configs).
*   **Implementation Source**: `models/maxvit_unet.py`, integrated from the `PRLAB21` MMSegmentation derivations ([Encoder](https://github.com/PRLAB21/MaxViT-UNet/blob/c822cbd283e8af45276e4888b771591250836012/mmseg/models/backbones/maxvit_encoder.py) / [Decoder](https://github.com/PRLAB21/MaxViT-UNet/blob/c822cbd283e8af45276e4888b771591250836012/mmseg/models/decode_heads/maxvit_decoder.py)).
*   **Technical Mechanics**:
    *   **Hierarchical Transformer Extraction**: Utilizes the external dependency `timm` to initialize the `maxvit_small_tf_224` backbone. Crucially, the system isolates intermediate tensor streams natively employing `features_only=True` mapping `out_indices=(0,1,2,3)`. This generates hierarchical, CNN-equivalent spatial layers rather than traditional transformer single-token outputs.
    *   **Divisibility Padding Requirement**: MaxViT’s explicit local Window Attention heavily crashes on asymmetrical dimensions. Consequently, the encoder module mathematically pads arrays via reflections (`F.pad()`) ensuring continuous divisibility down to the strict `224` scale required by a terminal $7x7$ window across an absolute 32x downsampling network.
    *   **Decoder Bridge**: Reverses the scale mapping bridging via `ConvBNAct` blocks and `UpBlock` decoders, relying on simple classical $3x3$ interpolation bridging logic versus heavy ASPP parameter loops.

---

## Part 2: Standard Architectures (Segmentation Models PyTorch - SMP)

These architectures run efficiently out-of-the-box leveraging the high-level API wrapper [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch).

### 1. UNet (SMP)
*   **Scientific Overview**: [SMP UNet Documentation](https://smp.readthedocs.io/en/latest/models.html#unet)
*   **Technical Breakdown**: Bypassing the rigid parameters of `UNet_original`, SMP fundamentally re-engineers UNet connectivity. The classical layers pad systematically by default, structurally securing symmetric layer bridging across non-divisible integer planes. Any imported backbone (e.g. `se_resnext50_32x4d`) naturally unhooks average pooling structures inherently, feeding layer depths smoothly into an interchangeable generic decoder array.

### 2. UNet++ (SMP)
*   **Scientific Overview**: [SMP UNet++ Documentation](https://smp.readthedocs.io/en/latest/models.html#unetplusplus)
*   **Technical Breakdown**: Evolves the UNet into a nested network. Where a normal UNet bridge connects Encoder Layer `A` $\rightarrow$ Decoder Layer `Z` traversing a simple single structural link, `UNet++` routes mappings through an aggressively dense nested sub-network structure. It strings intermittent internal convolution nodes across the horizontal semantic gaps. The intent is forcing the network to structurally inter-pollinate high-level semantic data mathematically with hard high-frequency local textures sequentially, massively stabilizing extreme boundary predictions.

### 3. DeepLabV3 (SMP)
*   **Scientific Overview**: [SMP DeepLabV3 Documentation](https://smp.readthedocs.io/en/latest/models.html#deeplabv3)
*   **Technical Breakdown**: The immediate architectural evolution addressing the limitations of ASPP found globally in DeepLabV2. At exceptionally high Atrous Rates ($>24$), the 3x3 dilated kernel filter spreads its sparse grid so massively that its internal parameter points map almost permanently to padding grids (zeros) severely disrupting spatial calculations (the 3x3 filter degrades mechanically into a 1x1 filter effect). V3 theoretically rectifies this flaw structurally by integrating explicit Image-Level features—appending a direct 1x1 Global Average Pooling sub-layer strictly into the primary ASPP matrix.

### 4. DeepLabV3+ (SMP)
*   **Scientific Overview**: [SMP DeepLabV3+ Documentation](https://smp.readthedocs.io/en/latest/models.html#deeplabv3plus)
*   **Technical Breakdown**: Extends and finalizes the architecture mapping DeepLab methodologies against general encoder-decoder geometries. While V3 provides incredibly robust semantic contexts utilizing multi-scale ASPP interpolation, V3 theoretically loses strict boundary resolution because the terminal ASPP output relies on a naive 8X Bilinear upscale map over continuous structures. V3+ solves this geometrically by establishing a lightweight independent Decoder module. It actively routes the massively scaled ASPP structural mapping backwards, directly concatenating with low-level spatial features intrinsically extracted natively out of the fundamental backbone operations.
