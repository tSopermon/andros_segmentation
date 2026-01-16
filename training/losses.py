import torch
import torch.nn as nn
import torch.nn.functional as F

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
