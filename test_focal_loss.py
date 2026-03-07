import torch
from training.losses import FocalLoss

loss_fn = FocalLoss(weight=torch.tensor([0.2, 0.8]), ignore_index=-100)
logits = torch.randn(2, 2, 5, 5)
targets = torch.randint(0, 2, (2, 5, 5))
targets[0, 0, 0] = -100 # Introduce ignore_index
try:
    loss = loss_fn(logits, targets)
    print("Success:", loss.item())
except Exception as e:
    print("Error:", type(e).__name__, "-", e)
