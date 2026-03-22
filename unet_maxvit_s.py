"""
Standalone semantic segmentation model:
- Encoder: timm MaxViT-S (features_only)
- Decoder: simple UNet-style upsampling + skip concatenation
- Head: 1x1 conv to num_classes=7

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
import timm


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
    def __init__(self, num_classes: int = 7, pretrained: bool = True, encoder_name: str = "maxvit_small_tf_224"):
        super().__init__()

        # timm encoder returns list of feature maps (low->high level): [f1,f2,f3,f4]
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
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

        self.head = nn.Conv2d(dec_c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)

        x = self.bridge(f4)
        x = self.up3(x, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        logits = self.head(x)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits
    

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from maxvit_s_unet_7cls import MaxViTSmallUNet


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss()  # add ignore_index=255 if your masks use void label
    total_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            logits = model(images)
            loss = ce(logits, masks)
            loss.backward()
            optimizer.step()
        else:
            with autocast():
                logits = model(images)
                loss = ce(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += float(loss.detach())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_miou(model, loader, device, num_classes=7):
    model.eval()
    # confusion matrix approach
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        preds = torch.argmax(logits, dim=1)  # [B,H,W]

        # flatten
        preds = preds.view(-1)
        masks = masks.view(-1)

        # if you have ignore label, filter here:
        # keep = masks != 255
        # preds, masks = preds[keep], masks[keep]

        k = (masks >= 0) & (masks < num_classes)
        inds = num_classes * masks[k] + preds[k]
        conf += torch.bincount(inds, minlength=num_classes**2).view(num_classes, num_classes)

    # IoU per class
    tp = torch.diag(conf).float()
    fp = conf.sum(0).float() - tp
    fn = conf.sum(1).float() - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom))
    miou = iou.mean().item()
    return miou, iou.detach().cpu().tolist()


def main(train_loader, val_loader, epochs=50, lr=3e-4, device="cuda"):
    device = torch.device(device)
    model = MaxViTSmallUNet(num_classes=7, pretrained=True, encoder_name="maxvit_small_tf_224").to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()  # comment out if you don't want AMP

    best_miou = -1.0
    patience = 10
    bad = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        miou, per_class_iou = evaluate_miou(model, val_loader, device, num_classes=7)

        print(f"epoch={epoch} loss={loss:.4f} mIoU={miou:.4f} per_class={per_class_iou}")

        if miou > best_miou:
            best_miou = miou
            bad = 0
            torch.save({"model": model.state_dict()}, "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch={epoch} (patience={patience}). Best mIoU={best_miou:.4f}")
                break