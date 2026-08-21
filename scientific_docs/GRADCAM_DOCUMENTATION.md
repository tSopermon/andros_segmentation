# Scientific Documentation: Grad-CAM / XAI Visualization

This document provides the scientific and mathematical foundation for the Class Activation Map (CAM) methods integrated into the Andros Segmentation system (`gradcam.py` + `evaluation/gradcam_utils.py`), and explains how the generated explanations should be interpreted for tested models, training sessions, and evaluation results.

The implementation is built on the [`jacobgil/pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library and exposes five methods: **GradCAM**, **HiResCAM**, **LayerCAM**, **EigenCAM**, and **SegEigenCAM**.

---

## 1. Background: Post-Hoc Explainability for Segmentation

Deep segmentation networks are conventionally evaluated with aggregate, spatially-blind metrics (IoU, F1, Overall Accuracy). These summarize *how often* the model is right but reveal nothing about *where* or *why* it focuses its decision. Post-hoc Class Activation Mapping fills this gap by producing a saliency map $L^c \in \mathbb{R}^{H \times W}$ that highlights the input regions contributing most to the model's prediction for a class $c$, without modifying or retraining the network.

The CAM family originates from [Zhou et al. (CVPR 2016)](https://arxiv.org/abs/1512.04150), who observed that the global-average-pooled feature maps of a convolutional classifier already encode discriminative localization. Gradient-based generalizations removed the architectural constraint of a global-average-pooling layer, making saliency computable for arbitrary networks — including the encoder-decoder segmentation architectures used in this repository.

For segmentation, the standard classification target is replaced by a **segmentation-aware target**: the class score is aggregated only over the pixels of the ground-truth mask of class $c$, so the resulting map localizes the evidence the model uses for that specific class rather than for the image as a whole.

---

## 2. Grad-CAM (Gradient-weighted Class Activation Mapping)

- **Scientific Base**: Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (ICCV 2017), *"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization."*
- **Primary Source**: [arXiv:1610.02391](https://arxiv.org/abs/1610.02391), [pytorch-grad-cam `GradCAM`](https://github.com/jacobgil/pytorch-grad-cam).

**Theoretical Mechanics**:

Grad-CAM computes a scalar importance weight $\alpha_k^c$ for each activation channel $A^k$ of a target layer by global-average-pooling the gradient of the class score $y^c$ with respect to that channel:

$$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

The saliency map is then a ReLU-gated, channel-weighted linear combination of the activations:

$$L^c_{\text{Grad-CAM}} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right)$$

- **Why ReLU**: Gradients quantify how a *perturbation* of each feature map changes the class score. Only channels with a positive influence ($\partial y^c / \partial A^k > 0$) are retained; negative-influence regions (evidence *against* the class) are suppressed. The map is finally upsampled to the input resolution.
- **Strength**: Simple, architecture-agnostic, and the de-facto baseline for gradient-based localization.
- **Limitation**: Global average pooling of gradients discards spatial structure within each channel, which reduces spatial *faithfulness* — a limitation that HiResCAM and Layer-CAM were explicitly designed to address.

---

## 3. HiResCAM (High-Resolution CAM)

- **Scientific Base**: Draelos, R. L., & Carin, L. (2021), *"Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks."*
- **Primary Source**: [arXiv:2011.08891](https://arxiv.org/abs/2011.08891), [pytorch-grad-cam `HiResCAM`](https://github.com/jacobgil/pytorch-grad-cam).

**Theoretical Mechanics**:

HiResCAM removes the gradient global-average-pooling step entirely, retaining the full element-wise gradient-activation product:

$$L^c_{\text{HiResCAM}} = \text{ReLU}\!\left(\sum_k g_k^c \odot A^k\right), \qquad g_k^c = \frac{\partial y^c}{\partial A^k}$$

- **Faithfulness guarantee**: Draelos & Carin prove that HiResCAM satisfies the formal notion of faithfulness that Grad-CAM violates — namely, that the explanation reflects the actual gradient-activation interaction at every spatial location, rather than a pooled approximation. This makes HiResCAM the preferred choice when the goal is to report *faithful* attribution rather than a smoothed saliency.
- **Practical trade-off**: Because it preserves per-pixel gradient structure, HiResCAM maps are typically noisier than Grad-CAM maps.

---

## 4. Layer-CAM

- **Scientific Base**: Jiang, P.-T., Zhang, C.-B., Hou, Q., Cheng, M.-M., & Wei, Y. (IEEE TIP 2021), *"LayerCAM: Exploring Hierarchical Class Activation Maps for Localization."*
- **Primary Source**: [arXiv:2103.03362](https://arxiv.org/abs/2103.03362), [pytorch-grad-cam `LayerCAM`](https://github.com/jacobgil/pytorch-grad-cam).

**Theoretical Mechanics**:

Layer-CAM gates each activation channel with the *positive* part of its gradient, applied element-wise:

$$L^c_{\text{LayerCAM}} = \text{ReLU}\!\left(\sum_k \text{ReLU}\!\left(g_k^c\right) \odot A^k\right)$$

- **Positive-gradient gating**: Unlike Grad-CAM's scalar $\alpha_k^c$, Layer-CAM keeps the spatial map of positive gradients. This both preserves spatial resolution *and* suppresses channels whose gradient is negative for the target class, yielding fine-grained localization without relying on a pooling operation.
- **Hierarchical extension**: The original paper aggregates such maps across multiple layers to build a hierarchy of explanations from coarse to fine. In this repository a single target layer is used for comparability with the other methods, but the mechanism is identical.

---

## 5. Eigen-CAM

- **Scientific Base**: Muhammad, M. B., & Yeasin, M. (2020), *"Eigen-CAM: Class Activation Map using Principal Components."*
- **Primary Source**: [arXiv:2008.00299](https://arxiv.org/abs/2008.00299), [pytorch-grad-cam `EigenCAM`](https://github.com/jacobgil/pytorch-grad-cam).

**Theoretical Mechanics**:

Eigen-CAM is **class-agnostic** and **gradient-free**. It treats the spatially-flattened activation matrix of the target layer as a data matrix, centers it, and computes its Singular Value Decomposition (SVD):

$$A_{\text{centered}} = U \Sigma V^\top$$

The saliency map is the leading principal-component direction $V_1$ projected back onto the spatial grid:

$$L^{\text{EigenCAM}} = \text{ReLU}\!\left(A_{\text{centered}} \cdot V_1\right)$$

- **Interpretation**: $V_1$ captures the direction of maximum variance among the activation responses — i.e., the shared pattern that dominates the layer's representation, independent of any specific class. It highlights *what the model "sees" as salient structure overall*.
- **Use in this system**: Because no target class is required, Eigen-CAM produces exactly one map per image. It is valuable for detecting *model-level* biases (e.g., the network fixating on a particular texture or object regardless of the requested class) rather than per-class decision evidence.

---

## 6. Seg-Eigen-CAM

- **Scientific Base**: Chung, C.-T., & Ying, J.J.-C. (2025), *"Seg-Eigen-CAM: Eigen-Value-Based Visual Explanations for Semantic Segmentation Models."* Applied Sciences, 15(13), 7562.
- **Primary Source**: [DOI:10.3390/app15137562](https://doi.org/10.3390/app15137562), [pytorch-grad-cam `SegEigenCAM`](https://github.com/jacobgil/pytorch-grad-cam).

**Theoretical Mechanics**:

Seg-Eigen-CAM extends Eigen-CAM with two segmentation-specific contributions:

1. **Gradient weighting** — the activations are re-weighted by the absolute value of the class gradient before decomposition, injecting per-pixel, class-specific spatial information:
$$A' = |g^c| \odot A$$

2. **Sign correction** — SVD has an inherent sign ambiguity ($V_1$ and $-V_1$ are equivalent). Seg-Eigen-CAM resolves this by comparing the magnitude of the most positive and most negative projection values and flipping the sign so that salient regions are always positive, guaranteeing a consistent, interpretable polarity.

- **Result**: a class-discriminative, eigenvalue-based saliency map that combines the global variance structure of Eigen-CAM with pixel-wise gradient evidence — purpose-built for dense prediction tasks such as our land-cover segmentation.

---

## 7. Implementation in the Andros System

### 7.1 Method registry

`evaluation/gradcam_utils.py::METHODS` records, for each method, its library class and whether it requires a segmentation target:

| Method | Class-discriminative (needs target) | Gradient-based | Decomposition |
|---|---|---|---|
| GradCAM | Yes | Yes | weighted sum |
| HiResCAM | Yes | Yes | element-wise sum |
| LayerCAM | Yes | Yes | positive-gated sum |
| EigenCAM | No | No | SVD / PCA |
| SegEigenCAM | Yes | Yes | SVD + sign correction |

### 7.2 Class targeting with `SemanticSegmentationTarget`

For class-discriminative methods, the class score $y^c$ in the formulas above is replaced by the library's `SemanticSegmentationTarget(category=c, mask=(gt == c))`, which computes the sum of class-$c$ logits restricted to the ground-truth pixels of class $c$. The gradient $\partial y^c / \partial A^k$ therefore reflects only how those pixels influence the representation — the standard, principled adaptation of CAM to semantic segmentation.

### 7.3 Target-layer resolution

CAM is computed with respect to a single target layer whose activations serve as $A^k$. `resolve_target_layers` selects it via:

1. an optional `GRADCAM_TARGET_LAYERS` per-model override in `config/config.yaml`;
2. a built-in default table — `encoder.layer4[-1]` for ResNet-family SMP encoders, `encoder.model.stages_3.blocks[-1]` for `tu-` timm encoders, `middle_conv.second` for `UNet_original`, etc.;
3. a fallback that selects the deepest module under the encoder whose class name contains `Conv`/`Block`/`Stage`.

For the default configuration (`UNetPlusPlus` with `tu-maxvit_large_tf_512`), this resolves to the terminal `MaxxVitBlock` of the deepest encoder stage — the semantically richest, lowest-resolution feature block, which is the conventional choice for CAM.

### 7.4 Preprocessing alignment and output

- The input tensor is produced by the **same** `get_val_transform` (pad → center-crop → ImageNet normalize) used in `evaluate.py`, and the ground-truth mask uses the **same** grayscale `label_mapping`. Heatmaps are therefore spatially aligned and directly comparable to the reported metrics and confusion matrices.
- The RGB overlay is composited on the **de-normalized** post-transform image (`show_cam_on_image`, JET colormap), so the heatmap grid coincides exactly with the 512×512 CAM.
- Two artifacts are saved per (method, class): a raw grayscale heatmap (`*_raw.png`, 0–255) and the JET overlay (`*_overlay.png`). Raw heatmaps are kept because they are the input required by quantitative faithfulness metrics (see §8.4).

### 7.5 Numerical and memory handling

- The CAM forward/backward runs under `torch.autocast(dtype=float16)` to fit large backbones (e.g. `maxvit_large` at 512×512) in GPU memory.
- The SVD-based methods (`EigenCAM`, `SegEigenCAM`) upcast activations/gradients to float32 before decomposition, since NumPy's `linalg.svd` does not accept float16.
- After each method, the retained autograd graph is explicitly released and the CUDA cache emptied, preventing accumulation across the five methods.

---

## 8. Interpreting the Results

### 8.1 For tested models (per-image)

- **GradCAM / HiResCAM / LayerCAM** answer the question *"which pixels drive the prediction of class $c$?"*. A high-quality model concentrates activation on the true extent of $c$ (e.g., water bodies for `Water`) with sharp, well-localized boundaries and little activation in spatially distant or unrelated regions.
- **LayerCAM** provides the finest spatial detail and is useful for inspecting boundary fidelity; **HiResCAM** is the most faithful and should be preferred when reporting formal attribution; **GradCAM** offers a smoothed, stable overview.
- **EigenCAM** (class-agnostic) reveals the dominant structure the encoder represents *regardless of class*. Repeated, class-independent activation on a single texture (e.g., a building pattern or a field) can expose dataset- or backbone-level bias that per-class maps cannot.

### 8.2 For training sessions (longitudinal)

By generating CAMs for the same test image across checkpoints or training epochs, one can observe whether the model's attention:

- **contracts** onto the correct object as training converges (a healthy signal), or
- **diffuses / latches onto spurious cues** (texture shortcuts, image borders, or a single dominant class) — an early warning of overfitting or loss-function imbalance.

This is complementary to the loss curves: loss values decrease monotonically while attention can simultaneously drift to incorrect regions, a failure mode invisible in scalar metrics alone.

### 8.3 For evaluation results (cross-referencing metrics)

- **Confusion-matrix-guided inspection**: for every class with low IoU/F1, inspect its CAMs to distinguish the failure mode — *under-segmentation* (activation covers only part of the class), *over-segmentation* (activation bleeds into neighbors), or *mis-attribution* (activation fires on the wrong class entirely). Each points to a different remedy (loss weighting, more data, or architectural change).
- **Boundary vs. interior errors**: LayerCAM/HiResCAM maps localized on boundaries indicate the model is using edge cues; interior-only activation with metric errors suggests coarse, low-frequency reasoning.
- **Class-agnostic cross-check**: if Eigen-CAM ignores a low-IoU class entirely, the class is likely underrepresented in the encoder's feature space, implicating the data pipeline rather than the decoder.

### 8.4 Toward quantitative XAI (future work)

The persisted raw heatmaps enable computing objective faithfulness metrics without re-running inference, notably:

- **ROAD** (Remove And Debias) — Rong, Leemann et al. (ICML 2022), [arXiv:2202.00449](https://arxiv.org/abs/2202.00449): measures the drop in model confidence/accuracy as the most salient pixels are progressively removed and imputed, rewarding *accurate* saliency. The reference implementation ships with the library in `pytorch_grad_cam.metrics.road`.
- **Perturbation-confidence metrics** (`CamMultImageConfidenceChange` / `DropInConfidence` / `IncreaseInConfidence`) — available in `pytorch_grad_cam.metrics.cam_mult_image`, these multiply the input by the CAM and measure the induced change in the predicted class score, providing a fast, parameter-free sanity check on saliency quality.

These metrics, deferred out of scope for the current integration, would turn the qualitative maps described here into statistically comparable explanation-quality scores across models.

---

## 9. References

1. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391).
2. Draelos, R. L., & Carin, L. (2021). *Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks.* [arXiv:2011.08891](https://arxiv.org/abs/2011.08891).
3. Jiang, P.-T., Zhang, C.-B., Hou, Q., Cheng, M.-M., & Wei, Y. (2021). *LayerCAM: Exploring Hierarchical Class Activation Maps for Localization.* IEEE Transactions on Image Processing. [arXiv:2103.03362](https://arxiv.org/abs/2103.03362).
4. Muhammad, M. B., & Yeasin, M. (2020). *Eigen-CAM: Class Activation Map using Principal Components.* IJCNN 2020. [arXiv:2008.00299](https://arxiv.org/abs/2008.00299).
5. Chung, C.-T., & Ying, J.J.-C. (2025). *Seg-Eigen-CAM: Eigen-Value-Based Visual Explanations for Semantic Segmentation Models.* Applied Sciences, 15(13), 7562. [DOI:10.3390/app15137562](https://doi.org/10.3390/app15137562).
6. Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). *Learning Deep Features for Discriminative Localization.* CVPR 2016. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150).
7. Gildenblat, J., & contributors. *pytorch-grad-cam.* [github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
8. Rong, Y., Leemann, T., Borisov, V., Kasneci, G., & Kasneci, E. (2022). *A Consistent and Efficient Evaluation Strategy for Attribution Methods.* ICML 2022 (ROAD). [arXiv:2202.00449](https://arxiv.org/abs/2202.00449).
