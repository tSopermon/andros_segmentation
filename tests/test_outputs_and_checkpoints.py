import os
import numpy as np
import tempfile
from PIL import Image

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
    ckpt_dir = "checkpoints"
    expected_names = ["DeepLabV3_best.pth", "DeepLabV3Plus_best.pth", "UNet_best.pth", "UNetPlusPlus_best.pth"]
    for name in expected_names:
        path = os.path.join(ckpt_dir, name)
        assert os.path.exists(path), f"Checkpoint {name} missing"
