#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from utils.config_loader import load_config
from models.model_zoo import get_models
from evaluation.mask_utils import save_mask
from evaluation.visualization import apply_color_mask, get_class_colors
from utils.logging_config import configure_logging

def sliding_window_inference(model, image_tensor, num_classes, patch_size=512, overlap=0.5, logger=None):
    """
    Perform sliding window inference on a large image tensor.
    image_tensor: (1, C, H, W)
    """
    _, _, h, w = image_tensor.shape
    stride = int(patch_size * (1 - overlap))
    if stride == 0:
        stride = patch_size
        
    if logger:
        logger.debug(f"Sliding window: patch_size={patch_size}, overlap={overlap}, stride={stride}")
    
    # Pad image to ensure the sliding window can cover the entire image
    # and to ensure the image is at least patch_size x patch_size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    
    padded_image = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, ph, pw = padded_image.shape
    
    # Initialize output and counter tensors on the same device
    device = image_tensor.device
    output_probs = torch.zeros((1, num_classes, ph, pw), device=device)
    count_map = torch.zeros((1, 1, ph, pw), device=device)
    
    # Generate coordinates for patches
    y_coords = list(range(0, ph - patch_size + 1, stride))
    x_coords = list(range(0, pw - patch_size + 1, stride))
    
    # Ensure the last patch covers the edge if not exactly reached
    if not y_coords or y_coords[-1] != ph - patch_size:
        y_coords.append(ph - patch_size)
    if not x_coords or x_coords[-1] != pw - patch_size:
        x_coords.append(pw - patch_size)
        
    total_patches = len(y_coords) * len(x_coords)
    
    if logger:
        logger.info(f"  -> Image split into {total_patches} overlapping patches.")
    
    patch_pbar = tqdm(total=total_patches, desc="  -> Processing Patches", leave=False)
    
    for y in y_coords:
        for x in x_coords:
            patch = padded_image[:, :, y:y+patch_size, x:x+patch_size]
            with torch.no_grad():
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    logits = model(patch)
                # If model returns a dict (like some torchvision models)
                if isinstance(logits, dict):
                    logits = logits['out']
                elif isinstance(logits, tuple):
                    logits = logits[0]
                    
                # Convert logits to probabilities
                probs = F.softmax(logits, dim=1)
                
            output_probs[:, :, y:y+patch_size, x:x+patch_size] += probs
            count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
            patch_pbar.update(1)
            
    patch_pbar.close()
    
    # Average the predictions
    output_probs /= count_map
    
    # Crop back to original dimensions
    final_output = output_probs[:, :, :h, :w]
    return final_output


def main():
    parser = argparse.ArgumentParser(description='Predict segmentation masks using Sliding Window Inference.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image file or directory.')
    parser.add_argument('--output', type=str, default='predictions', help='Directory to save the predicted masks.')
    parser.add_argument('--model', type=str, required=True, help='Model architecture name (e.g., UNetPlusPlus).')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint (.pth).')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML file.')
    parser.add_argument('--no-overlays', action='store_true', help='Skip generating semi-transparent mask overlays.')
    parser.add_argument('--alpha', type=float, default=0.4, help='Opacity of the mask overlay (0.0 to 1.0).')
    
    # New sliding window arguments
    parser.add_argument('--patch-size', type=int, default=512, help='Size of the patches for sliding window inference.')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap fraction between patches (0.0 to 0.99).')
    
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
    logger.info(f"System initialized. Using device: {device}")

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
    logger.info(f"Detected {NUM_CLASSES} target classes.")

    # Initialize model
    old_env = {k: os.environ.get(k) for k in ('USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL', 'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
    try:
        os.environ['USE_UNET_ORIGINAL'] = 'true'
        os.environ['USE_DEEPLABV1_ORIGINAL'] = 'true'
        os.environ['USE_DEEPLABV2_ORIGINAL'] = 'true'
        os.environ['USE_DEEPLABV3_ORIGINAL'] = 'true'
        os.environ['USE_MAXVIT_UNET'] = 'true'
        
        BACKBONE = config.get('BACKBONE', 'resnet101')
        logger.info(f"Building Model Factory: {args.model} with {BACKBONE} backbone...")
        models_dict = get_models(NUM_CLASSES, backbone=BACKBONE, specific_model=args.model)
        if args.model not in models_dict:
            raise ValueError(f"Model '{args.model}' could not be initialized.")
        model = models_dict[args.model]
    finally:
        for k, v in old_env.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    logger.info("Model loaded and moved to GPU memory successfully.")

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Normalization matches the training distribution
    base_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    colors = get_class_colors(NUM_CLASSES)

    # Output directory setup
    os.makedirs(args.output, exist_ok=True)
    masks_dir = os.path.join(args.output, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    if not args.no_overlays:
        overlays_dir = os.path.join(args.output, "overlays")
        os.makedirs(overlays_dir, exist_ok=True)

    logger.info("==================================================")
    logger.info(f"Starting Inference Pipeline on {len(image_paths)} images")
    logger.info(f"Strategy: Patch-based Sliding Window")
    logger.info(f"Patch Size: {args.patch_size}x{args.patch_size} | Overlap: {args.overlap*100}%")
    logger.info("==================================================")
    
    for img_path in image_paths:
        base_name = img_path.stem
        logger.info(f"Reading: {img_path.name}")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logger.warning(f"  -> Failed to read image, skipping.")
            continue
        
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]
        
        logger.info(f"  -> Original dimensions: {original_w}x{original_h}")
        
        # Transform and convert to tensor
        augmented = base_transform(image=image)
        img_tensor = augmented['image'].unsqueeze(0).to(device)

        # Sliding Window Inference
        output_probs = sliding_window_inference(
            model=model,
            image_tensor=img_tensor,
            num_classes=NUM_CLASSES,
            patch_size=args.patch_size,
            overlap=args.overlap,
            logger=logger
        )
        
        # Process output (1, C, H, W) -> (H, W)
        pred = torch.argmax(output_probs, dim=1).squeeze().cpu().numpy()

        # Save masks
        mask_filename = f"{base_name}_mask.png"
        mask_save_path = os.path.join(masks_dir, mask_filename)
        save_mask(pred, mask_save_path, NUM_CLASSES)
        logger.info(f"  -> Saved semantic mask: {mask_filename}")
        
        # Save overlays
        if not args.no_overlays:
            overlay_rgb = apply_color_mask(image, pred, colors, alpha=args.alpha)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            overlay_filename = f"{base_name}_overlay.png"
            overlay_save_path = os.path.join(overlays_dir, overlay_filename)
            cv2.imwrite(overlay_save_path, overlay_bgr)
            logger.info(f"  -> Saved blended overlay: {overlay_filename}")
            
    logger.info("==================================================")
    logger.info(f"Process complete! Results available in: {args.output}")

if __name__ == "__main__":
    main()
