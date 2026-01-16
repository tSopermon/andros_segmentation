import numpy as np
from utils.transforms import get_train_transform, get_val_transform

def test_train_transform():
    transform = get_train_transform(32, use_augmentation=True)
    assert callable(transform)
    sample = {
        'image': np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
        'mask': np.random.randint(0, 2, (32, 32), dtype=np.uint8)
    }
    augmented = transform(image=sample['image'], mask=sample['mask'])
    assert 'image' in augmented
    assert 'mask' in augmented
    assert augmented['image'].shape[1:] == (32, 32)
    assert augmented['mask'].shape == (32, 32)

def test_val_transform():
    transform = get_val_transform(32)
    assert callable(transform)
    sample = {
        'image': np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8),
        'mask': np.random.randint(0, 2, (32, 32), dtype=np.uint8)
    }
    augmented = transform(image=sample['image'], mask=sample['mask'])
    assert 'image' in augmented
    assert 'mask' in augmented
    assert augmented['image'].shape[1:] == (32, 32)
    assert augmented['mask'].shape == (32, 32)

    def test_transform_invalid_input():
        from utils.transforms import get_train_transform
        import pytest
        transform = get_train_transform(32)
        # Pass None as input
        with pytest.raises(Exception):
            transform(None)
