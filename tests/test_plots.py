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
    # Create dummy test_loader
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, idx):
            return torch.zeros(3, 32, 32), torch.randint(0, 8, (32, 32))
    test_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    # Should not raise and should create PNGs
    plots.plot_metric_vs_class_frequency(all_test_results, test_loader, label_mapping, output_dir=tmp_path)
    assert (tmp_path / 'iou_vs_class_frequency.png').exists()
    assert (tmp_path / 'f1_vs_class_frequency.png').exists()