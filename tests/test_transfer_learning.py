import torch
import torch.nn as nn
import os
import pytest
from training.train_utils import apply_transfer_learning, freeze_encoder_if_requested
from models.model_zoo import get_models

class DummyModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3),
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        return self.classifier(self.encoder(x))

def test_apply_transfer_learning_shape_mismatch(tmp_path):
    device = torch.device('cpu')
    
    # Create pretrained model with 5 classes
    pretrained_model = DummyModel(3, 5)
    ckpt_path = tmp_path / "pretrained.pth"
    torch.save(pretrained_model.state_dict(), ckpt_path)
    
    # Create new model with 2 classes
    new_model = DummyModel(3, 2)
    
    # Assert initial weights are different
    assert not torch.equal(new_model.encoder[0].weight, pretrained_model.encoder[0].weight)
    
    # Apply transfer learning
    apply_transfer_learning(new_model, str(ckpt_path), device)
    
    # Assert encoder weights are now equal
    assert torch.equal(new_model.encoder[0].weight, pretrained_model.encoder[0].weight)
    
    # Assert classifier weights are NOT equal (due to shape mismatch)
    assert new_model.classifier.weight.shape != pretrained_model.classifier.weight.shape

def test_freeze_encoder_if_requested():
    model = DummyModel(3, 2)
    
    # Initially requires grad
    assert model.encoder[0].weight.requires_grad == True
    
    # Freeze
    freeze_encoder_if_requested(model, True)
    
    # Check
    assert model.encoder[0].weight.requires_grad == False
    assert model.classifier.weight.requires_grad == True

    # Test UNet_original
    models_dict = get_models(2, specific_model='UNet_original')
    if 'UNet_original' in models_dict:
        unet = models_dict['UNet_original']
        # Check initial state
        assert unet.down_conv[0].first.weight.requires_grad == True
        # Freeze
        freeze_encoder_if_requested(unet, True)
        assert unet.down_conv[0].first.weight.requires_grad == False
        assert unet.middle_conv.first.weight.requires_grad == False
        assert unet.final_conv.weight.requires_grad == True
