"""
https://github.com/rulixiang/deeplab-pytorch/blob/master/models/DeepLabV1_LargeFOV.py

Implementation of DeepLabV1 with Large Field-of-View (LargeFOV) as described in:
"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs"
by Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
Published in ICLR 2015

DeepLab-LargeFOV is a specific variant of the DeepLabV1 architecture designed to improve
semantic segmentation performance by increasing the receptive field of the network.
DeepLab-LargeFOV is approximately equivalent to DeepLabV1 with atrous convolution with rate=12 in the
last two convolution layers.
"""


import torch
import torch.nn as nn
import torchvision


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation)

class Block(nn.Module):

    def __init__(self, in_planes=None, out_planes=None, depth=2, padding=1, dilation=1, pool_stride=2):
        
        super(Block, self).__init__()
        self.depth = depth
        self.conv1 = conv3x3(in_planes, out_planes, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        if self.depth==3:
            self.conv3 = conv3x3(out_planes, out_planes, padding=padding, dilation=dilation)
            self.relu3 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=1)

    def forward(self,x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)

        if self.depth==3:
            x = self.conv3(x)
            x = self.relu3(x)

        x = self.pool(x)

        return x

class DeepLabV1_LargeFOV(nn.Module):

    def __init__(self, n_classes=21, init_weights=True,):

        super(DeepLabV1_LargeFOV, self).__init__()
        #self.config = config
        planes = [3, 64, 128, 256, 512, 512]

        self.block_1 = Block(planes[0], planes[1])
        self.block_2 = Block(planes[1], planes[2])
        self.block_3 = Block(planes[2], planes[3], depth=3)
        self.block_4 = Block(planes[3], planes[4], depth=3, pool_stride=1)
        self.block_5 = Block(planes[4], planes[5], depth=3, dilation=2, padding=2, pool_stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = conv3x3(in_planes=512, out_planes=1024, padding=12, dilation=12)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(0.5)

        self.conv7 = conv1x1(in_planes=1024, out_planes=1024, padding=0)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(0.5)

        self.conv8 = conv1x1(in_planes=1024, out_planes=n_classes, padding=0)

        if init_weights is True:
            self._init_weights()

    def forward(self, x):
        input_size = x.shape[2:]  # Store input spatial size (H, W)
        
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.avg_pool(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.drop6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        x = self.drop7(x)

        x = self.conv8(x)
        
        # Upsample output to match input resolution
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        return None
            
