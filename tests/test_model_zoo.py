from models.model_zoo import get_models

def test_get_models():
    num_classes = 2
    models = get_models(num_classes)
    assert isinstance(models, dict)
    assert set(models.keys()) == {'DeepLabV3', 'DeepLabV3Plus', 'UNet', 'UNetPlusPlus', 'UNet_original'}
    for model in models.values():
        assert hasattr(model, 'forward')

def test_get_models_invalid_backbone():
    from models.model_zoo import get_models
    import pytest
    with pytest.raises(Exception):
        get_models(num_classes=2, backbone="nonexistent_backbone")

def test_model_forward_pass():
    from models.model_zoo import get_models
    import torch
    models = get_models(num_classes=2)
    dummy_input = torch.randn(2, 3, 224, 224)  # Larger batch and spatial size
    for name, model in models.items():
        out = model(dummy_input)
        assert out.shape[0] == 2
