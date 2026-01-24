#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima (Adapted)
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19
#
# Adapted to use Torchvision ResNet backbone for better initialization and training stability.

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.resnet import ResNet101_Weights, ResNet50_Weights

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Module):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    
    Now uses torchvision.models.resnet101 with pretrained weights.
    """

    def __init__(self, n_classes, n_blocks=None, atrous_rates=None, backbone='resnet50'):
        """
        Args:
            n_classes (int): Number of output classes.
            n_blocks (list): Deprecated/Ignored. Kept for backward compatibility.
            atrous_rates (list): List of atrous rates for ASPP. Defaults to [6, 12, 18, 24].
            backbone (str): 'resnet50' or 'resnet101'.
        """
        super(DeepLabV2, self).__init__()
        
        if atrous_rates is None:
            atrous_rates = [6, 12, 18, 24]
            
        if backbone == 'resnet101':
            resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, True, True])
            aspp_in = 2048
        else:
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, True, True])
            aspp_in = 2048
        
        # We need everything up to layer4
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # ResNet101 layer4 output channels is 2048, ResNet18 is 512
        self.aspp = _ASPP(aspp_in, n_classes, atrous_rates)

    def forward(self, x):
        input_size = x.shape[2:]  # Store input spatial size (H, W)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.aspp(x)
        
        # Upsample output to match input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)