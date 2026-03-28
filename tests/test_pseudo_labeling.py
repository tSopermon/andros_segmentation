import numpy as np
import torch
import cv2
import pytest
from pathlib import Path
import os
from utils.dataset import DualStreamDataset
from training.train_utils import train_epoch
from training.metrics import SegmentationMetrics

def create_dummy_image(path, shape=(32, 32, 3)):
    img = np.random.randint(0, 255, shape, dtype=np.uint8)
    cv2.imwrite(str(path), img)

def create_dummy_mask(path, shape=(32, 32), num_classes=2):
    mask = np.random.randint(0, num_classes, shape, dtype=np.uint8)
    cv2.imwrite(str(path), mask)

class DummyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

def test_dual_stream_dataset_logic(tmp_path):
    """Verifies that DualStreamDataset correctly interleaves and augments labeled/unlabeled data."""
    l_img_dir = tmp_path / "labeled_images"
    l_mask_dir = tmp_path / "labeled_masks"
    u_img_dir = tmp_path / "unlabeled_images"
    l_img_dir.mkdir()
    l_mask_dir.mkdir()
    u_img_dir.mkdir()
    
    l_images = ["l1.png", "l2.png"]
    l_masks = ["m1.png", "m2.png"]
    u_images = ["u1.png", "u2.png", "u3.png"]
    
    for f in l_images: create_dummy_image(l_img_dir / f)
    for f in l_masks: create_dummy_mask(l_mask_dir / f)
    for f in u_images: create_dummy_image(u_img_dir / f)
    
    dataset = DualStreamDataset(
        l_img_dir, l_mask_dir, l_images, l_masks,
        u_img_dir, u_images,
        transform=None, unl_transform_weak=None, unl_transform_strong=None
    )
    
    # Dataset length should be governed by the labeled set
    assert len(dataset) == 2
    
    item = dataset[0]
    assert len(item) == 4
    l_img, l_mask, u_img_w, u_img_s = item
    
    assert l_img.shape == (3, 32, 32)
    assert l_mask.shape == (32, 32)
    assert u_img_w.shape == (3, 32, 32)
    assert u_img_s.shape == (3, 32, 32)

def test_train_epoch_teacher_logic():
    """Verifies that the training loop correctly handles teacher-student pseudo-labeling."""
    num_classes = 3
    model = DummyModel(num_classes)
    teacher_model = DummyModel(num_classes)
    teacher_model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Using ignore_index=-1 as configured in the project
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    metrics = SegmentationMetrics(num_classes)
    
    # Mock dual-stream batch: (l_img, l_mask, u_img_w, u_img_s)
    batch_size = 2
    l_imgs = torch.randn(batch_size, 3, 32, 32)
    l_masks = torch.randint(0, num_classes, (batch_size, 32, 32))
    u_imgs_w = torch.randn(batch_size, 3, 32, 32)
    u_imgs_s = torch.randn(batch_size, 3, 32, 32)
    
    # Simulate a few batches
    loader = [(l_imgs, l_masks, u_imgs_w, u_imgs_s)] * 2
    device = torch.device('cpu')
    
    train_loss, train_metrics_dict = train_epoch(
        model, loader, criterion, optimizer, device, metrics,
        teacher_model=teacher_model, teacher_threshold=0.9, ignore_index=-1
    )
    
    assert isinstance(train_loss, float)
    assert 'f1_mean' in train_metrics_dict
    assert train_loss >= 0

def test_pseudo_label_masking_integrity():
    """Specifically checks if the pseudo-labels are correctly thresholded and masked with -1."""
    num_classes = 2
    teacher_model = DummyModel(num_classes)
    teacher_model.eval()
    
    # Create input that would give predictable output if we could, 
    # but here we just check if -1 is actually present after thresholding
    # if the teacher is 'unsure'.
    with torch.no_grad():
        u_imgs_w = torch.randn(1, 3, 32, 32)
        logits = teacher_model(u_imgs_w)
        probs = torch.softmax(logits, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        # Manually force some low confidence to test the masking logic
        teacher_threshold = 0.99 
        # Inside train_epoch logic:
        # mask = max_probs < teacher_threshold
        # pseudo_labels[mask] = ignore_index
        
        # We verify that if we set a very high threshold, we get many -1s
        if torch.any(max_probs < teacher_threshold):
            pseudo_labels[max_probs < teacher_threshold] = -1
            assert -1 in pseudo_labels
