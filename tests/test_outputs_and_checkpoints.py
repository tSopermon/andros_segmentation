import os
import numpy as np
import tempfile
from PIL import Image
from utils.config_loader import load_config
from utils.model_selection import get_selected_model_names

def test_output_mask_generation():
    # Simulate mask output
    mask = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "masks")
        os.makedirs(out_dir)
        out_path = os.path.join(out_dir, "mask.png")
        Image.fromarray(mask).save(out_path)
        assert os.path.exists(out_path)
        img = Image.open(out_path)
        assert img.size == (8, 8)

def test_checkpoint_naming_convention():
    """Verifies that checkpoints exist for all models currently selected in config.yaml."""
    config = load_config("config/config.yaml")
    selected_models = get_selected_model_names(config)
    ckpt_dir = "checkpoints"
    
    # We only assert if there are selected models
    for name in selected_models:
        # Known exception: DeepLabV3_original does not have a local .pth under our naming convention 
        # as it loads from torch hub. But in generate_masks.py we mapped it to a name.
        # However, for the standard setup, we check <ModelName>_best.pth
        pth_name = f"{name}_best.pth"
        path = os.path.join(ckpt_dir, pth_name)
        assert os.path.exists(path), f"Checkpoint for active model {name} missing at {path}"
