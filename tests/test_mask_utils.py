import numpy as np
import tempfile
from evaluation.mask_utils import save_mask
from PIL import Image

def test_save_mask_creates_png():
    mask = np.random.randint(0, 2, (8, 8), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/mask.png"
        save_mask(mask, save_path, num_classes=2)
        img = Image.open(save_path)
        assert img.mode == 'P'
        assert img.size == (8, 8)
        arr = np.array(img)
        assert np.array_equal(arr, mask)

def test_save_mask_palette_length():
    mask = np.zeros((4, 4), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = f"{tmpdir}/mask.png"
        save_mask(mask, save_path, num_classes=5)
        img = Image.open(save_path)
        palette = img.getpalette()
        assert len(palette) >= 15  # 3*num_classes
