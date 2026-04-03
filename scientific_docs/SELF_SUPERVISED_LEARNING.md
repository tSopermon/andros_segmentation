# Self-Supervised Learning (Masked Autoencoder) Guide

## Overview
When semantic segmentation datasets are exceedingly small, standard supervised learning quickly results in severe overfitting. To counter this, the repository implements an autoassociative **Self-Predictive Learning** pipeline inspired by Masked Autoencoders (MAEs).

Instead of memorizing tiny datasets, the network is forced to "fill in the blanks" on highly obscured inputs, learning generalized global contexts, structural hierarchies, and texture awareness completely independent of labeled ground truth.

Because standard U-Net++ architectures require dense spatial grids, the pipeline employs a dynamic, on-GPU patch masking utility that replaces hidden sectors with `0.0`, allowing CNNs to reap the benefits typical of vision transformers.

## Workflow

### 1. Pre-Training Phase (`pretrain.py`)
The pipeline loads raw RGB images from the training directory (ignoring masks).
It divides images into a grid of patches (e.g. 16x16) and hides a massive portion of them (e.g. 75%).
The U-Net++ receives the obscured image and attempts to reconstruct the native RGB pixels over the hidden patches. The model's loss operates exclusively on the unseen areas via a masked `MSELoss`.

**Masking Evolution:**
- **Random Masking**: Used initially to learn color, low-level texture, and general structure.
- **Object-Centric Masking (Self-Guided)**: Configured by `OBJECT_CENTRIC_EPOCH`, the masking utility seamlessly transitions via a Sobel edge-density heuristic. It actively targets edge-heavy regions (objects) to be masked out, preventing the network from lazily reconstructing flat backgrounds and forcing it to infer complex object anatomy.

### 2. Fine-Tuning Phase (`train.py`)
After pre-training, the `_pretrained.pth` weights contain a robust encoder and decoder. 

By running `train.py` with `TRANSFER_LEARNING: true` and the suffix `PRETRAINED_WEIGHT_SUFFIX: '_pretrained.pth'`, the `apply_transfer_learning()` framework gracefully absorbs the pretrained backbone into the segmentation model. 

It identically maps 1:1 shapes (the entire architecture) and safely discards the `[N, 3, H, W]` RGB reconstruction head, initializing an untrained `[N, NUM_CLASSES, H, W]` segmentation head in its place. Because the system inherently understands boundaries and shapes, it will converge on the few available labels considerably faster yielding superior generalizability.

## Execution Requirements

In `config/config.yaml`:

1. Define your pre-training strategy:
    ```yaml
    # -----------------------------------------------------------------------------
    # Pre-training / Self-Predictive Learning
    # -----------------------------------------------------------------------------
    PRETRAIN_EPOCHS: 100
    MASK_RATIO: 0.75
    PATCH_SIZE: 16
    # Epoch to transition from random mask to object-centric masking
    OBJECT_CENTRIC_EPOCH: 50
    ```
2. Run Pre-Training:
    ```bash
    python pretrain.py --config config/config.yaml
    ```
3. Enable downstream inheritance in `config/config.yaml`:
    ```yaml
    TRANSFER_LEARNING: true
    PRETRAINED_CHECKPOINT_DIR: 'checkpoints/'
    PRETRAINED_WEIGHT_SUFFIX: '_pretrained.pth'
    ```
4. Run Fine-Tuning:
    ```bash
    python train.py --config config/config.yaml
    ```

## Expected Behavior and Loss Convergence

When running `pretrain.py` on a standard CNN architecture like U-Net++, you will likely observe the Masked Mean-Squared Error (MSE) drop extremely rapidly—sometimes reaching values as low as `0.0005` or `0.0002` within just a few epochs. 

**Is this expected? Yes.** 
Unlike Vision Transformers (ViTs) natively used in MAE papers which treat patches completely independently, standard CNNs employ dense convolutions with overlapping sliding windows. This means the receptive field of the network allows information from visible patches to "bleed" or "leak" into the masked patches as the convolutions traverse the spatial grid. 

Because of this architectural trait, the CNN essentially performs an advanced interpolation (inpainting) leveraging nearby pixels, which represents a significantly easier pretext task than it is for a ViT. While the loss converges quickly, the network is still forced to learn highly valuable low-level edge detection, texture recognition, and local structural context to interpolate successfully.

## Configuring Pre-training Based on Your Dataset

To maximize the representation learning quality and prevent the network from taking interpolation "shortcuts", you should configure the masks based on the geometric complexity of your images:

1. **`PATCH_SIZE`**: 
   - **Default (`16`)**: Best for standard medium-resolution imagery.
   - **Larger (`32` or `64`)**: If your dataset consists of very high-resolution images or massive, contiguous objects, the CNN can easily interpolate across a 16x16 missing block using surrounding data. Increasing the patch size forces the network to infer larger structural components, reducing the CNN's ability to cheat via local pixel bleeding.

2. **`MASK_RATIO`**:
   - **Default (`0.75`)**: The standard optimal ratio.
   - **Higher (`0.80` - `0.90`)**: If your MSE loss is converging *too* quickly (e.g., `< 0.0001` in 1-2 epochs), it indicates the task is too easy. Masking out 85-90% of the patches drastically reduces the visible context, severely limiting the CNN's interpolation shortcuts and forcing it to learn holistic global concepts.
   - **Lower (`0.50`)**: Only recommended if the objects are extremely sparse and the network completely fails to reconstruct anything.

3. **`OBJECT_CENTRIC_EPOCH`**:
   - Begin with random masking to let the network learn raw color/texture heuristics.
   - Transition to object-centric masking (e.g., halfway through your `PRETRAIN_EPOCHS`) to dynamically concentrate the patches over edge-dense foreground elements. This is absolutely critical for datasets with huge, homogeneous backgrounds (e.g., satellite imagery, clear skies, empty oceans), completely neutralizing the model's ability to achieve a low MSE by simply predicting flat colors.

---

## Comparison: Self-Supervised vs. Semi-Supervised (Self-Training)

While both techniques leverage unlabeled data, they serve different purposes in the training pipeline:

| Feature | Self-Supervised (MAE) | Semi-Supervised (Self-Training) |
| :--- | :--- | :--- |
| **Logic** | Reconstruct masked RGB pixels. | Predict semantic labels (Pseudo-Labels). |
| **Labels Needed** | None (Zero Labels). | Small set of Ground Truth Labels. |
| **Stage** | Pre-training (Initial backbone formation). | Fine-tuning refinement (Boosting performance). |
| **Architecture** | Enforced symmetry (Encoder + Decoder). | Teacher-Student (Frozen + Active). |
| **Documentation** | `SELF_SUPERVISED_LEARNING.md` | `SELF_TRAINING.md` |

For optimal results on very small datasets, the ideal workflow is to **Pre-train (SSL)** $\rightarrow$ **Fine-tune (Supervised)** $\rightarrow$ **Refine (Self-Training)**.
