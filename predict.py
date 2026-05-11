#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm
import logging

from utils.config_loader import load_config
from models.model_zoo import get_models
from evaluation.mask_utils import save_mask
from utils.transforms import get_val_transform
from evaluation.visualization import apply_color_mask, get_class_colors
from utils.logging_config import configure_logging

def main():
    parser = argparse.ArgumentParser(description='Predict segmentation masks for an image or directory.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image file or directory containing images.')
    parser.add_argument('--output', type=str, default='predictions', help='Directory to save the predicted masks (default: ./predictions).')
    parser.add_argument('--model', type=str, required=True, help='Model architecture name (e.g., UNetPlusPlus, UNet, DeepLabV3).')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint (.pth). Defaults to checkpoints/{model}_best.pth')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file.')
    parser.add_argument('--no-overlays', action='store_true', help='Skip generating semi-transparent mask overlays.')
    parser.add_argument('--alpha', type=float, default=0.4, help='Opacity of the mask overlay (0.0 to 1.0).')
    args = parser.parse_args()

    # Determine input path
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Collect images
    image_paths = []
    if input_path.is_file():
        image_paths.append(input_path)
    elif input_path.is_dir():
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        for f in input_path.iterdir():
            if f.is_file() and f.suffix.lower() in valid_exts:
                image_paths.append(f)
    else:
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

    if not image_paths:
        print(f"No valid images found in {input_path}")
        return

    # Load config
    config = load_config(args.config)
    configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))
    logger = logging.getLogger(__name__)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Determine checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join('checkpoints', f"{args.model}_best.pth")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Detect number of classes
    detected_classes = None
    for key in checkpoint.keys():
        if any(k in key for k in ['segmentation_head.0.weight', 'final_conv.weight', 'head.weight', 'classifier.4.weight']):
            detected_classes = checkpoint[key].shape[0]
            break
    
    if detected_classes is None:
        for key, val in checkpoint.items():
            if ('head' in key or 'final' in key) and val.ndim == 4:
                detected_classes = val.shape[0]
                break

    NUM_CLASSES = detected_classes if detected_classes else config.get('NUM_CLASSES', 8)
    logger.info(f"Detected {NUM_CLASSES} classes from checkpoint.")

    # Initialize model
    # Temporarily set environment variables if needed by model_zoo
    old_env = {k: os.environ.get(k) for k in ('USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL', 'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
    try:
        os.environ['USE_UNET_ORIGINAL'] = 'true'
        os.environ['USE_DEEPLABV1_ORIGINAL'] = 'true'
        os.environ['USE_DEEPLABV2_ORIGINAL'] = 'true'
        os.environ['USE_DEEPLABV3_ORIGINAL'] = 'true'
        os.environ['USE_MAXVIT_UNET'] = 'true'
        
        BACKBONE = config.get('BACKBONE', 'resnet101')
        models_dict = get_models(NUM_CLASSES, backbone=BACKBONE, specific_model=args.model)
        if args.model not in models_dict:
            raise ValueError(f"Model '{args.model}' could not be initialized. Available models: {list(models_dict.keys())}")
        model = models_dict[args.model]
    finally:
        for k, v in old_env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # We won't use val_transform because it CenterCrops and destroys arbitrary image edges.
    # Instead, we define a transform that only normalizes and converts to tensor.
    base_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    colors = get_class_colors(NUM_CLASSES)

    # Output directory
    os.makedirs(args.output, exist_ok=True)
    masks_dir = os.path.join(args.output, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    if not args.no_overlays:
        overlays_dir = os.path.join(args.output, "overlays")
        os.makedirs(overlays_dir, exist_ok=True)

    logger.info(f"Processing {len(image_paths)} images...")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting"):
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                logger.warning(f"Failed to read image: {img_path}")
                continue
            
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            original_h, original_w = image.shape[:2]
            
            # Pad image dimensions to a multiple of 32 (required by UNet/DeepLab encoder-decoder structures)
            pad_h = (32 - original_h % 32) % 32
            pad_w = (32 - original_w % 32) % 32
            
            padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            
            augmented = base_transform(image=padded_image)
            img_tensor = augmented['image'].unsqueeze(0).to(device)

            output = model(img_tensor)
            if hasattr(output, 'detach'):
                output = output.detach().cpu().numpy()
            else:
                output = output[0].detach().cpu().numpy()
                
            if output.ndim == 4:
                pred = np.argmax(output[0], axis=0)
            elif output.ndim == 3:
                pred = np.argmax(output, axis=0)
            else:
                pred = output.squeeze()

            # Crop prediction back to original image size
            pred = pred[:original_h, :original_w]

            base_name = img_path.stem
            mask_filename = f"{base_name}_mask.png"
            mask_save_path = os.path.join(masks_dir, mask_filename)
            
            save_mask(pred, mask_save_path, NUM_CLASSES)
            
            if not args.no_overlays:
                overlay_rgb = apply_color_mask(image, pred, colors, alpha=args.alpha)
                overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                overlay_filename = f"{base_name}_overlay.png"
                overlay_save_path = os.path.join(overlays_dir, overlay_filename)
                cv2.imwrite(overlay_save_path, overlay_bgr)
                
    logger.info(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()
