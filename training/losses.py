import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.weight = weight # Alpha parameter per class
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits shape: (N, C, H, W)
        # targets shape: (N, H, W)

        # Calculate unweighted cross entropy to extract pure probabilities
        ce_loss_unweighted = F.cross_entropy(
            logits, targets, reduction='none', ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss_unweighted)

        # Calculate weighted cross entropy for the loss value
        ce_loss_weighted = F.cross_entropy(
            logits, targets, weight=self.weight, reduction='none', ignore_index=self.ignore_index
        )

        # Calculate Focal Loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted

        if self.reduction == 'mean':
            if self.weight is not None:
                if self.ignore_index >= 0:
                    # Prevent IndexError by zeroing out ignored indices before lookup
                    safe_targets = torch.where(targets != self.ignore_index, targets, 0)
                    weights = self.weight[safe_targets]
                    weights = weights.masked_fill(targets == self.ignore_index, 0.0)
                else:
                    weights = self.weight[targets]
                return focal_loss.sum() / torch.clamp(weights.sum(), min=1e-8)
            else:
                if self.ignore_index >= 0:
                    valid_mask = (targets != self.ignore_index)
                    return focal_loss.sum() / torch.clamp(valid_mask.sum().float(), min=1e-8)
                else:
                    return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W)
            targets: (B, H, W) or (B, C, H, W) one-hot
        """
        num_classes = logits.shape[1]
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot if they aren't already
        if targets.dim() == 3:
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        else:
            targets_one_hot = targets.float()

        # Flatten (B, C, H, W) -> (B, C, H*W)
        probs_flat = probs.view(probs.size(0), num_classes, -1)
        targets_flat = targets_one_hot.view(targets_one_hot.size(0), num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes, ignoring ignore_index if specified
        # Note: ignore_index logic for Dice is tricky if completely missing from target,
        # but standard approach is simple mean over classes.
        return 1.0 - dice.mean()

class DiceBCELoss(nn.Module):
    """
    Combined Dice and Cross Entropy Loss.
    Named 'BCE' to match common terminology, but uses CrossEntropy for multi-class support.
    loss = DiceLoss + CrossEntropyLoss
    """
    def __init__(self, weight=None, smooth=1e-6, ignore_index=-100):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        ce_loss = self.ce(logits, targets)
        return dice_loss + ce_loss
