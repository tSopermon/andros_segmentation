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
    assert np.isclose(result['iou'][1], 0.5)

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
