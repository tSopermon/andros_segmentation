import numpy as np
import torch
import pytest
from training.metrics import SegmentationMetrics

def test_metrics_shape():
    num_classes = 3
    metrics = SegmentationMetrics(num_classes)
    assert metrics.tp.shape == (num_classes,)
    assert metrics.fp.shape == (num_classes,)
    assert metrics.fn.shape == (num_classes,)
    assert metrics.intersection.shape == (num_classes,)
    assert metrics.union.shape == (num_classes,)

def test_metrics_update_and_compute():
    num_classes = 2
    metrics = SegmentationMetrics(num_classes)
    # Simulate predictions and targets
    pred = torch.zeros((2, num_classes, 4, 4))
    pred[:, 1, :, :] = 1.0  # Class 1 logits higher
    target = torch.ones((2, 4, 4), dtype=torch.long)  # All class 1
    metrics.update(pred, target)
    result = metrics.compute_metrics()
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result
    assert 'iou' in result
    assert result['precision'].shape == (num_classes,)
    assert result['recall'].shape == (num_classes,)
    assert result['f1'].shape == (num_classes,)
    assert result['iou'].shape == (num_classes,)
    # Class 1 should have perfect scores
    assert np.isclose(result['precision'][1], 1.0)
    assert np.isclose(result['recall'][1], 1.0)
    assert np.isclose(result['f1'][1], 1.0)
    assert np.isclose(result['iou'][1], 1.0)

def test_metrics_reset():
    num_classes = 2
    metrics = SegmentationMetrics(num_classes)
    pred = torch.zeros((2, num_classes, 4, 4))
    target = torch.ones((2, 4, 4), dtype=torch.long)
    metrics.update(pred, target)
    metrics.reset()
    assert np.all(metrics.tp == 0)
    assert np.all(metrics.fp == 0)
    assert np.all(metrics.fn == 0)

def test_metrics_empty_input():
    num_classes = 2
    metrics = SegmentationMetrics(num_classes)
    pred = torch.empty((0, num_classes, 4, 4))
    target = torch.empty((0, 4, 4), dtype=torch.long)
    metrics.update(pred, target)
    result = metrics.compute_metrics()
    # Accept any output, just check correct type/shape or scalar
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            assert v.shape == (num_classes,)
        else:
            assert np.isscalar(v)

def test_metrics_averages():
    num_classes = 2
    metrics = SegmentationMetrics(num_classes)
    # Batch 1: 10 pixels of class 0, all correct
    # (1, 2, 10, 1) -> N, C, H, W
    pred1 = torch.zeros((1, num_classes, 10, 1))
    pred1[:, 0, :, :] = 10.0 # High logit for class 0
    target1 = torch.zeros((1, 10, 1), dtype=torch.long)
    metrics.update(pred1, target1)
    
    # Batch 2: 2 pixels of class 1, all wrong (predicted as class 0)
    pred2 = torch.zeros((1, num_classes, 2, 1))
    pred2[:, 0, :, :] = 10.0 # High logit for class 0
    target2 = torch.ones((1, 2, 1), dtype=torch.long)
    metrics.update(pred2, target2)
    
    result = metrics.compute_metrics()
    
    # Class 0: TP=10, FP=2, FN=0 -> Intersection=10, Union=12 -> IoU = 10/12 = 0.8333
    # Class 1: TP=0, FP=0, FN=2 -> Intersection=0, Union=2 -> IoU = 0.0
    # Macro IoU = (0.8333 + 0.0) / 2 = 0.4166
    # Weighted IoU = (0.8333 * 10 + 0.0 * 2) / 12 = 0.6944
    # Micro IoU = Sum(Inter) / Sum(Union) = 10 / (12 + 2) = 10/14 = 0.7142
    
    assert np.isclose(result['iou_mean'], 0.416666, atol=1e-4)
    assert np.isclose(result['iou_weighted'], 0.694444, atol=1e-4)
    assert np.isclose(result['iou_micro'], 0.714285, atol=1e-4)
    assert 'f1_weighted' in result
    assert 'precision_micro' in result

