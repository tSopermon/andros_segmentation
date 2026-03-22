"""
Standalone semantic segmentation model:
- Encoder: timm MaxViT-S (features_only)
- Decoder: simple UNet-style upsampling + skip concatenation
- Head: 1x1 conv

Repo references (exact permalinks in PRLAB21/MaxViT-UNet; for traceability only):
- Encoder (repo's own MaxViT backbone, MMSeg-style):
  https://github.com/PRLAB21/MaxViT-UNet/blob/c822cbd283e8af45276e4888b771591250836012/mmseg/models/backbones/maxvit_encoder.py
- Decoder (repo's own MaxViTDecoder head, MMSeg-style):
  https://github.com/PRLAB21/MaxViT-UNet/blob/c822cbd283e8af45276e4888b771591250836012/mmseg/models/decode_heads/maxvit_decoder.py
- Config wiring (MMSeg):
  https://github.com/PRLAB21/MaxViT-UNet/blob/c822cbd283e8af45276e4888b771591250836012/configs/_base_/models/maxvit_unet.py

Encoder source-of-truth (your choice):
- timm (installed dependency). To lock exact encoder source, pin timm version in requirements.

Usage:
    model = MaxViTSmallUNet(num_classes=7, pretrained=True, encoder_name="maxvit_small_tf_224")
    logits = model(images)  # [B, 7, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, act=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True) if act in (nn.ReLU, nn.ReLU6) else act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class UpBlock(nn.Module):
    """Upsample -> concat skip -> conv -> conv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBNAct(out_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv2(self.conv1(x))
        return x


class MaxViTSmallUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 7, pretrained: bool = True, encoder_name: str = "maxvit_small_tf_224"):
        super().__init__()
        
        if timm is None:
            raise ImportError("timm is not installed. Please install it using 'pip install timm'.")

        # timm encoder returns list of feature maps (low->high level): [f1,f2,f3,f4]
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        enc_ch = self.encoder.feature_info.channels()

        # decoder widths (tune if you want)
        dec_c3, dec_c2, dec_c1 = 256, 128, 64

        self.bridge = ConvBNAct(enc_ch[-1], dec_c3)
        self.up3 = UpBlock(in_ch=dec_c3, skip_ch=enc_ch[-2], out_ch=dec_c2)
        self.up2 = UpBlock(in_ch=dec_c2, skip_ch=enc_ch[-3], out_ch=dec_c1)
        self.up1 = UpBlock(in_ch=dec_c1, skip_ch=enc_ch[-4], out_ch=dec_c1)

        self.head = nn.Conv2d(dec_c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_h, orig_w = x.shape[-2:]
        
        # MaxViT window attention requires feature maps to be divisible by window size (7)
        # down to the deepest feature map. 
        # With a downsampling ratio of 32 across encoder stages, input dims must be divisible by 32 * 7 = 224.
        divisible_by = 224
        pad_h = (divisible_by - orig_h % divisible_by) % divisible_by
        pad_w = (divisible_by - orig_w % divisible_by) % divisible_by
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        f1, f2, f3, f4 = self.encoder(x)

        x = self.bridge(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        logits = self.head(x)
        
        # To match exact requested input dims, interpolate if the padded/downsampled dims strictly mismatched sizes
        # In a standard unet architecture we want it to output identically sized spatial outputs as standard unet.
        if logits.shape[-2:] != (orig_h, orig_w):
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
        return logits
