# Manual Transfer Learning Workflow Guide

This guide details how to perform a 3-Stage, 4-Step transfer learning workflow (Pre-training $\rightarrow$ Head Swap $\rightarrow$ Frozen Fine-Tuning $\rightarrow$ Unfrozen Fine-Tuning) using the **existing** codebase without modifying any Python files. 

By leveraging the built-in configuration options (`config/config.yaml`), you can manually pause, re-configure, and resume training to achieve the exact architecture swapping, freezing, and learning rate adjustments required for fine-tuning on a small dataset.

---

## Stage 1: Pre-training

First, train the model from scratch on your first dataset to learn robust feature representations.

1.  **Configure Dataset:** Ensure your dataset points to the dataset.
2.  **Configure Settings (`config/config.yaml`):**
    *   Set `DATA_AUGMENTATION: true` (or false, depending on your large dataset needs).
    *   Set `TRANSFER_LEARNING: false`
    *   Set `FREEZE_ENCODER: false`
    *   Set `LEARNING_RATE: 0.001` (or your standard starting LR).
3.  **Run Training:**
    ```bash
    python train.py --config config/config.yaml
    ```
4.  **Result:** Your fully trained 7-class models will be saved in the `checkpoints/` directory (e.g., `checkpoints/UNet_best.pth`). Copy this directory somewhere safe (e.g., `checkpoints_stage1/`) so it isn't overwritten.

> **Alternative Stage 1:** If you lack a large labeled dataset, you can substitute Stage 1 with self-supervised feature extraction (Masked Autoencoder). Run `python pretrain.py` on your unlabeled images to generate a highly capable backbone, then proceed immediately to Stage 2. See [SELF_SUPERVISED_LEARNING.md](SELF_SUPERVISED_LEARNING.md) for dedicated SSL instructions.

---

## Stage 2: Head Swap & Frozen Fine-Tuning

Now, shift to the 120-sample dataset. We will load the pre-trained weights, let the system automatically swap the 7-class head for an 8-class head, and train **only** the new head to prevent random gradients from destroying the pre-trained backbone.

1.  **Configure Dataset:** Update your dataset path to point to the 120-sample dataset.
2.  **Configure Settings (`config/config.yaml`):**
    *   **The Swap & Load:** Set `TRANSFER_LEARNING: true` and point `PRETRAINED_CHECKPOINT_DIR` to your saved Stage 1 checkpoints (e.g., `checkpoints_stage1/`). The system will automatically detect the shape mismatch in the classifier head, load the backbone weights, and randomly initialize a new 8-class head.
    *   **Freeze the Backbone:** Set `FREEZE_ENCODER: true`. This ensures only the new, uninitialized layers receive updates.
    *   **Heavy Augmentation:** Set `DATA_AUGMENTATION: true` to prevent overfitting on the small dataset.
    *   **Learning Rate:** Keep a standard learning rate (e.g., `LEARNING_RATE: 0.001`). Since the backbone is frozen, this only applies to the new head.
    *   **Epochs:** Set `MAX_EPOCHS` to a small number (e.g., 5 to 10) just to let the new head settle.
3.  **Run Training:**
    ```bash
    python train.py --config config/config.yaml
    ```
4.  **Result:** The model now has a partially trained 8-class head attached to the pre-trained 7-class backbone. The new weights are saved in `checkpoints/`. Again, copy this to a safe location (e.g., `checkpoints_stage2/`).

---

## Stage 3: Unfrozen Fine-Tuning (Differential LR Approximation)

Finally, unfreeze the entire network and train it end-to-end so the backbone can gently adapt to the new 8th class.

Since we cannot modify the optimizer code to use natively split parameter groups (true differential learning rates), we achieve the same goal sequentially: the head was trained at a high LR in Stage 2, and now the *entire* network (head + backbone) is trained at a very small LR.

1.  **Configure Settings (`config/config.yaml`):**
    *   **Keep Architecture & Augmentation:** `NUM_CLASSES: 8`, `DATA_AUGMENTATION: true`.
    *   **Load Stage 2 Weights:** Set `TRANSFER_LEARNING: true` and point `PRETRAINED_CHECKPOINT_DIR` to the checkpoints from Step 2 (`checkpoints_stage2/`).
    *   **Unfreeze the Backbone:** Set `FREEZE_ENCODER: false`. Gradients will now flow through the entire model.
    *   **Tiny Learning Rate:** Set `LEARNING_RATE` to a much smaller value (e.g., `LEARNING_RATE: 0.0001` or `0.00001` - about 10x to 100x smaller than Stage 2). This prevents the backbone weights from drifting too far away from their robust pre-trained state.
    *   **Epochs:** Set `MAX_EPOCHS` to your desired length to finish fine-tuning.
3.  **Run Training:**
    ```bash
    python train.py --config config/config.yaml
    ```

### Summary of Manual Configs

| Phase | `NUM_CLASSES` | `TRANSFER_LEARNING` (`PRETRAINED_CHECKPOINT_DIR`) | `FREEZE_ENCODER` | `LEARNING_RATE` | `DATA_AUGMENTATION` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Pre-train)** | 7 | `false` | `false` | Normal (e.g., `1e-3`) | Up to workflow |
| **Stage 2 (Head Only)** | 8 | `true` (Points to Stage 1) | `true` | Normal (e.g., `1e-3`) | `true` |
| **Stage 3 (Full Tune)** | 8 | `true` (Points to Stage 2) | `false` | Tiny (e.g., `1e-4` or `1e-5`) | `true` |