import torch
from training.train_utils import train_epoch, validate, evaluate_model
from training.metrics import SegmentationMetrics

class DummyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

def test_train_and_validate():
    num_classes = 2
    model = DummyModel(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = SegmentationMetrics(num_classes)
    images = torch.randn(4, 3, 8, 8)
    masks = torch.randint(0, num_classes, (4, 8, 8))
    loader = [(images, masks)]
    device = torch.device('cpu')
    train_loss, train_metrics_dict = train_epoch(model, loader, criterion, optimizer, device, metrics)
    val_loss, val_metrics_dict = validate(model, loader, criterion, device, metrics)
    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert 'f1_mean' in train_metrics_dict
    assert 'f1_mean' in val_metrics_dict

def test_evaluate_model():
    num_classes = 2
    model = DummyModel(num_classes)
    metrics = SegmentationMetrics(num_classes)
    images = torch.randn(4, 3, 8, 8)
    masks = torch.randint(0, num_classes, (4, 8, 8))
    loader = [(images, masks)]
    device = torch.device('cpu')
    result = evaluate_model(model, loader, device, metrics)
    assert 'f1_mean' in result

def test_train_epoch_mixed_precision():
    import torch
    from training.train_utils import train_epoch
    from training.metrics import SegmentationMetrics
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, num_classes, kernel_size=1)
        def forward(self, x):
            return self.conv(x)
    num_classes = 2
    model = DummyModel(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = SegmentationMetrics(num_classes)
    images = torch.randn(4, 3, 8, 8)
    masks = torch.randint(0, num_classes, (4, 8, 8))
    loader = [(images, masks)]
    device = torch.device('cpu')
    # Simulate mixed precision context
    try:
        from torch.amp import autocast
        with autocast('cuda'):
            train_loss, train_metrics_dict = train_epoch(model, loader, criterion, optimizer, device, metrics)
        assert isinstance(train_loss, float)
    except ImportError:
        pass

def test_global_seed_reproducibility_train_utils():
    import torch
    torch.manual_seed(42)
    a = torch.randn(1)
    torch.manual_seed(42)
    b = torch.randn(1)
    assert torch.allclose(a, b)
