import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import torch
import yaml
import numpy as np
from tqdm import tqdm
from utils.config_loader import load_config
from utils.model_selection import get_selected_model_names
from models.model_zoo import get_models
from evaluation.mask_utils import get_test_dataset, save_mask
from utils.transforms import get_val_transform
from evaluation.visualization import apply_color_mask, get_class_colors
import cv2

import argparse

# Paths
# CONFIG_PATH will be set via argparse
CHECKPOINTS_DIR = 'checkpoints'
OUTPUTS_DIR = 'outputs'

# Model names and corresponding checkpoint files
MODEL_CHECKPOINTS = {
	'DeepLabV3': 'DeepLabV3_best.pth',
	'DeepLabV3Plus': 'DeepLabV3Plus_best.pth',
	'UNet': 'UNet_best.pth',
	'UNetPlusPlus': 'UNetPlusPlus_best.pth',
	'UNet_original': 'UNet_original_best.pth',
}

from pathlib import Path
import logging
from utils.logging_config import configure_logging


def main():
	"""
	Run all trained models on the test set and generate segmentation masks for each image.
	Masks are saved in model-specific output folders with a color palette for visualization.
	"""
	parser = argparse.ArgumentParser(description='Generate segmentation masks.')
	parser.add_argument('--config', default='config/config.yaml', help='Path to config YAML file')
	parser.add_argument('--no-overlays', action='store_true', help='Skip generating semi-transparent mask overlays')
	parser.add_argument('--alpha', type=float, default=0.4, help='Opacity of the mask overlay (0.0 to 1.0)')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config = load_config(args.config)
	dataset_path = Path(config['DATASET_PATH'])

	# Determine model set from config. We temporarily set env vars so model_zoo registers
	# originals when required, but restore the environment after obtaining models.
	model_set = config.get('MODEL_SET', 'standard')

	# Build the checkpoint mapping lookup
	local_model_checkpoints_map = dict(MODEL_CHECKPOINTS)
	local_model_checkpoints_map['DeepLabV1_original'] = 'DeepLabV1_original_best.pth'
	local_model_checkpoints_map['DeepLabV2_original'] = 'DeepLabV2_original_best.pth'
	local_model_checkpoints_map['DeepLabV3_original'] = 'DeepLabV3_original_best.pth'
	local_model_checkpoints_map['MaxViTSmallUNet'] = 'MaxViTSmallUNet_best.pth'

	# Selection controlled by config only: choose the set of models to generate masks for
	selected_models = get_selected_model_names(config)
	local_model_checkpoints = {
		k: local_model_checkpoints_map[k]
		for k in selected_models
		if k in local_model_checkpoints_map
	}

	# Determine which subsets to include (always include test by default)
	include_all = config.get('GENERATE_FOR_ALL_SETS', False)
	include_lowres = config.get('INCLUDE_LOWRES', False)
	subsets = ['test']
	if include_all:
		subsets = ['train', 'val', 'test']

	# Build image list (path, subset, base_name) for inference
	image_entries = []  # tuples of (image_path: Path, subset: str, base_name: str)
	# Collect masks dirs to infer number of classes
	mask_dirs = []
	for subset in subsets:
		img_dir = dataset_path / subset / ('Image' if (dataset_path / subset / 'Image').exists() else 'image')
		mask_dir = dataset_path / subset / ('Mask' if (dataset_path / subset / 'Mask').exists() else 'mask')
		if img_dir.exists():
			files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
			for fname in files:
				base = os.path.splitext(fname)[0]
				image_entries.append((img_dir / fname, subset, base))
		if mask_dir.exists():
			mask_dirs.append(mask_dir)

	# Include lowres images if requested (look under lowres/image and lowres/mask)
	if include_lowres:
		lowres_img_dir = dataset_path / 'lowres' / ('Image' if (dataset_path / 'lowres' / 'Image').exists() else 'image')
		lowres_mask_dir = dataset_path / 'lowres' / ('Mask' if (dataset_path / 'lowres' / 'Mask').exists() else 'mask')
		if lowres_img_dir.exists():
			files = sorted([f for f in os.listdir(lowres_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
			for fname in files:
				base = os.path.splitext(fname)[0]
				image_entries.append((lowres_img_dir / fname, 'lowres', base))
		if lowres_mask_dir.exists():
			mask_dirs.append(lowres_mask_dir)

	# Fallback: if nothing found in the above, use the original test dataset helper
	if len(image_entries) == 0:
		test_dataset = get_test_dataset(config)
		image_entries = [(Path(test_dataset.image_dir) / f, 'test', os.path.splitext(f)[0]) for f in test_dataset.image_files]
		mask_dirs.append(Path(dataset_path) / 'test' / ('Mask' if (Path(dataset_path) / 'test' / 'Mask').exists() else 'mask'))

	# NUM_CLASSES logic is now handled per-model inside the loop based on checkpoints.
	# We still keep a fallback for color palette generation if needed.
	NUM_CLASSES_FALLBACK = config.get('NUM_CLASSES', 8)

	configure_logging(level=config.get('LOGGING_LEVEL', 'INFO'))
	logger = logging.getLogger(__name__)
	# Ensure required checkpoints exist for the requested models
	missing = []
	for m, c in local_model_checkpoints.items():
		path = os.path.join(CHECKPOINTS_DIR, c)
		if not os.path.exists(path):
			logger.error('No such file: %s', path)
			missing.append(path)
	if len(missing) > 0:
		raise SystemExit(2)

	for model_name, ckpt_file in local_model_checkpoints.items():
		logger.info('\nGenerating masks for %s...', model_name)
		# Prepare output directory
		model_output_dir = os.path.join(OUTPUTS_DIR, model_name, 'masks')
		os.makedirs(model_output_dir, exist_ok=True)
		saved_root_preview = False

		checkpoint_path = os.path.join(CHECKPOINTS_DIR, ckpt_file)
		if not os.path.exists(checkpoint_path):
			logger.warning('Missing checkpoint for %s: %s', model_name, checkpoint_path)
			continue

		# Load checkpoint to detect classes
		checkpoint = torch.load(checkpoint_path, map_location='cpu')
		
		# Detect classes from state_dict
		detected_classes = None
		# SMP models: segmentation_head.0.weight shape is [out_channels, in_channels, 1, 1]
		# UNet_original: final_conv.weight
		# MaxViT: head.weight
		# DeepLab wrapper: base.classifier.x.weight
		for key in checkpoint.keys():
			if any(k in key for k in ['segmentation_head.0.weight', 'final_conv.weight', 'head.weight', 'classifier.4.weight']):
				detected_classes = checkpoint[key].shape[0]
				break
		
		if detected_classes is None:
			# Last resort: try any conv weight in a 'head' or 'final' layer
			for key, val in checkpoint.items():
				if ('head' in key or 'final' in key) and val.ndim == 4:
					detected_classes = val.shape[0]
					break
		
		NUM_CLASSES = detected_classes if detected_classes else NUM_CLASSES_FALLBACK
		logger.info('Detected %d classes from checkpoint for %s', NUM_CLASSES, model_name)

		# Build model specifically for this checkpoint's class count
		old_env = {k: os.environ.get(k) for k in ('USE_UNET_ORIGINAL', 'USE_DEEPLABV1_ORIGINAL', 'USE_DEEPLABV2_ORIGINAL', 'USE_DEEPLABV3_ORIGINAL', 'USE_MAXVIT_UNET')}
		try:
			if model_set in ('originals', 'all'):
				os.environ['USE_UNET_ORIGINAL'] = 'true'
				os.environ['USE_DEEPLABV1_ORIGINAL'] = 'true'
				os.environ['USE_DEEPLABV2_ORIGINAL'] = 'true'
				os.environ['USE_DEEPLABV3_ORIGINAL'] = 'true'
				os.environ['USE_MAXVIT_UNET'] = 'true'
			else:
				os.environ['USE_DEEPLABV1_ORIGINAL'] = str(config.get('USE_DEEPLABV1_ORIGINAL', False)).lower()
				os.environ['USE_DEEPLABV2_ORIGINAL'] = str(config.get('USE_DEEPLABV2_ORIGINAL', False)).lower()
				os.environ['USE_DEEPLABV3_ORIGINAL'] = str(config.get('USE_DEEPLABV3_ORIGINAL', False)).lower()
				os.environ['USE_MAXVIT_UNET'] = str(config.get('USE_MAXVIT_UNET', False)).lower()

			BACKBONE = config.get('BACKBONE', 'resnet101')
			model = get_models(NUM_CLASSES, backbone=BACKBONE, specific_model=model_name)[model_name]
		finally:
			for k, v in old_env.items():
				if v is None: os.environ.pop(k, None)
				else: os.environ[k] = v

		model.load_state_dict(checkpoint)
		model.to(device)
		model.eval()

		val_transform = get_val_transform(config['IMAGE_SIZE'])
		colors = get_class_colors(NUM_CLASSES)
		with torch.no_grad():
			for i, (img_path, subset_name, base_name) in enumerate(tqdm(image_entries)):
				# Read & preprocess image
				image_bgr = cv2.imread(str(img_path))
				if image_bgr is None:
					logger.warning('Failed to read image: %s', img_path)
					continue
				image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
				augmented = val_transform(image=image)
				img_tensor = augmented['image'].unsqueeze(0).to(device)

				output = model(img_tensor)
				if hasattr(output, 'detach'):
					output = output.detach().cpu().numpy()
				else:
					output = output[0].detach().cpu().numpy()
				# Assume output shape: (B, C, H, W) or (B, H, W)
				if output.ndim == 4:
					pred = np.argmax(output[0], axis=0)
				elif output.ndim == 3:
					pred = np.argmax(output, axis=0)
				else:
					pred = output.squeeze()
				# Warn if mask is all zeros (likely model issue)
				if np.all(pred == 0):
					logger.warning("All-zero mask for %s in %s", str(img_path), model_name)
				# Save mask with color palette inside subset-specific folder (no prefix)
				subset_dir = os.path.join(model_output_dir, subset_name)
				os.makedirs(subset_dir, exist_ok=True)
				mask_filename = base_name + '_mask.png'
				save_path = os.path.join(subset_dir, mask_filename)
				save_mask(pred, save_path, NUM_CLASSES)
				# Also save one preview mask at the model root masks folder for compatibility with tests
				if (not saved_root_preview):
					preview_path = os.path.join(model_output_dir, mask_filename)
					try:
						save_mask(pred, preview_path, NUM_CLASSES)
						saved_root_preview = True
					except Exception:
						pass

				# Generate and save overlay if requested
				if not args.no_overlays:
					overlay_dir = os.path.join(OUTPUTS_DIR, model_name, 'overlays', subset_name)
					os.makedirs(overlay_dir, exist_ok=True)
					
					# apply_color_mask expects RGB image
					overlay_rgb = apply_color_mask(image, pred, colors, alpha=args.alpha)
					# Convert to BGR for saving with OpenCV
					overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
					
					overlay_filename = base_name + '_overlay.png'
					cv2.imwrite(os.path.join(overlay_dir, overlay_filename), overlay_bgr)


if __name__ == '__main__':
	main()
 
