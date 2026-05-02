import pytest
import numpy as np
import torch
from evaluation import plots


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Return a tensor with shape (N, C, H, W)
        N, C, H, W = x.shape[0], 8, x.shape[2], x.shape[3]
        return torch.randn(N, C, H, W)


def test_plot_metric_vs_class_frequency(tmp_path):
    # Setup dummy data
    all_test_results = {
        'DummyModel': {
            'iou': np.random.rand(8),
            'f1': np.random.rand(8)
        }
    }
    label_mapping = {i: i for i in range(8)}
    # Create dummy target masks as a numpy array (N, H, W) — NOT a DataLoader
    num_samples = 4
    height, width = 32, 32
    all_targets = np.stack(
        [np.random.randint(0, 8, (height, width)) for _ in range(num_samples)],
        axis=0
    )
    # Should not raise and should create PNGs
    plots.plot_metric_vs_class_frequency(
        all_test_results, all_targets, label_mapping, output_dir=tmp_path
    )
    assert (tmp_path / 'iou_vs_class_frequency.png').exists()
    assert (tmp_path / 'f1_vs_class_frequency.png').exists()