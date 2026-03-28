# Data Loading and Training Process

## Data Loading Strategy

The system uses a custom `SegmentationDataset` class (`utils/dataset.py`) to handle image and mask loading.

### 1. Dataset Initialization
- **Configuration**: The dataset path is defined in `config/config.yaml` (`DATASET_PATH`). The flag `PRE_SPLIT_DATASET` can be set to bypass automatic splitting.
- **Structure**: The system expects the following directory structure:
  - `train/image/` & `train/mask/`
  - `val/image/` & `val/mask/` (If `PRE_SPLIT_DATASET: true`)
  - `test/image/` & `test/mask/`
- **File Matching**: Images and masks are matched by filename. Extensions `.jpg`, `.jpeg`, `.png`, `.tif`, and `.tiff` are supported.
- **Class Mapping**: A `label_mapping` dictionary (derived from `counts` or fixed) ensures mask pixel values map to the correct training class indices (0 to N-1). When `PRE_SPLIT_DATASET` is enabled, the validation set is also scanned for classes.

### 2. Item Retrieval (`__getitem__`)
On-the-fly processing occurs when a batch is requested:
1. **Image Loading**: Images are loaded via OpenCV and converted from BGR to RGB.
2. **Mask Loading**: Masks are loaded in grayscale.
3. **Augmentation**: If enabled (`val` or `train` with augmentations), Albumentations transforms are applied (e.g., matching flips, crops, or color changes) to both image and mask. 
   - **Validation**: Use `python evaluation/visualize_augmentation.py` to generate a visual report of applied augmentations in `outputs/debug/augmentation_validation.png`.
4. **Normalization**: Images are normalized (typically to `[0, 1]` or ImageNet stats via transforms) and converted to PyTorch tensors `(C, H, W)`. Masks become Long tensors `(H, W)`.

### 3. Dual Stream Loading (Self-Training)
When `SELF_TRAINING: true` is enabled, the system switches to the `DualStreamDataset`.
-   **Structure**: It maintains a synchronous stream of labeled `(image, mask)` pairs and unlabeled `image` samples.
-   **Batching**: To maintain GPU memory limits, if `BATCH_SIZE=N`, the loader yields `N/2` labeled pairs and `N/2` unlabeled samples per step.
-   **Augmentation Split**: The unlabeled stream produces two versions of the same image: a **weakly augmented** version for the Teacher (to ensure stable pseudo-labels) and a **strongly augmented** version for the Student (to force robust feature learning).

## Model Training Workflow

Training is orchestrated by `train.py`, featuring K-Fold Cross-Validation, Ensembling, and Mix-Precision training.

### 1. Model Initialization (`models/model_zoo.py`)
- **Factory Pattern**: `get_models()` instantiates architectures based on `MODEL_NAMES` (e.g., DeepLabV3, UNet).
- **Backends**: 
  - **SMP**: Wraps `segmentation_models_pytorch` for standard architectures (ResNet/EfficientNet backbones).
  - **Originals**: Supports paper-faithful implementations (e.g., `DeepLabV2_original`) via custom code or `torchvision` wrappers.
- **Weights**: Pretrained weights (e.g., ImageNet, COCO) are loaded if configured.

### 2. Training Loop
The training process (`train.py`) follows these steps:
1. **Split Strategy**: 
   - **Pre-split Data:** If `PRE_SPLIT_DATASET: true`, training runs entirely on the `train/` directory and validation entirely on the `val/` directory. `K_FOLDS` is forced to 1, and `ENSEMBLE` is disabled.
   - **Stratified K-Fold:** If `PRE_SPLIT_DATASET: false`, it uses Stratified K-Fold (or simple train_test_split if `K_FOLDS=1`) to split the `train/` data into training/validation folds to ensure class balance.
   - Calculates **Class Weights** based on pixel frequency in the training fold to handle class imbalance.
2. **Optimization**:
   - **Loss**: configurable `DiceBCE`, `Dice`, `Focal`, or `CrossEntropy` (weighted). Also includes **`DiceFocal`**: A highly tunable combination of Dice and Focal loss configurable via `DICE_WEIGHT` and `FOCAL_WEIGHT` in config.
   - **Optimizer**: Adam with `ExponentialLR` scheduling.
   - **Precision**: Uses `torch.amp` (Automatic Mixed Precision) for memory efficiency and speed.
3. **Validation & Monitoring**:
   - Evaluates on the held-out fold every epoch.
   - Tracks metrics: **mIoU**, Precision, Recall, F1.
   - **Early Stopping**: Triggered if validation mIoU stops improving for `PATIENCE` epochs.
4. **Checkpointing**: Saves the best model state (`_best.pth`) for each fold.
5. **Transfer Learning (Fine-Tuning)**:
    - If `TRANSFER_LEARNING: true` is set, the system attempts to load weights from the specified `PRETRAINED_CHECKPOINT_DIR`.
    - **Shape Mismatch Handling**: The loader automatically filters out layers with mismatched tensor shapes (e.g. if the pretrained model had 5 classes and the new dataset has 2, the classifier layer is safely skipped while others are loaded).
    - **Encoder Freezing**: If `FREEZE_ENCODER: true` is enabled, the backbone/encoder weights are frozen (`requires_grad=False`), forcing the model to only train the decoder/classifier layers.
5. **Semi-Supervised Learning (Self-Training)**:
    - If `SELF_TRAINING: true`, a frozen **Teacher** model is instantiated from an existing best checkpoint.
    - The Teacher generates pseudo-labels for the unlabeled data stream.
    - **Confidence Thresholding**: Only pixels where the Teacher's softmax probability exceeds `PSEUDO_LABEL_THRESHOLD` are kept as labels.
    - **Masking**: Uncertain pixels are assigned `IGNORE_INDEX` (`-1`), which masks them out of the loss calculation for the Student.

### 3. Ensembling & Retraining
- **Ensemble**: Optionally averages predictions from all K-fold best models on the test set.
- **Production Retrain**: Automatically selects the best-performing fold config and retrains a final model on the **full dataset** (Train + Val) to maximize performance for production use.
