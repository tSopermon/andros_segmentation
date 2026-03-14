def train_epoch(model, train_loader, criterion, optimizer, device, metrics=None, epoch=None, max_epochs=None, lr=None, phase='Training'):
    """
    Run one training epoch.

    Args:
        model (torch.nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to use.
        metrics (object, optional): Metrics object.
        epoch (int, optional): Current epoch.
        max_epochs (int, optional): Total epochs.
        lr (float, optional): Learning rate.
        phase (str): Phase name for progress bar.

    Returns:
        tuple: (average loss, metrics dict or None)
    """
    return _run_epoch(model, train_loader, criterion, device, optimizer, metrics, epoch, max_epochs, lr, phase, is_train=True)
import torch
from tqdm import tqdm
import os
import logging

logger = logging.getLogger(__name__)

def _run_epoch(model, loader, criterion, device, optimizer=None, metrics=None, epoch=None, max_epochs=None, lr=None, phase='Training', is_train=True):
    """
    Internal helper to run one epoch of training or validation.

    Args:
        model (torch.nn.Module): Model to train or evaluate.
        loader (DataLoader): DataLoader for the dataset.
        criterion (callable): Loss function.
        device (torch.device): Device to use.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training. Required if is_train=True.
        metrics (object, optional): Metrics object with reset, update, and compute_metrics methods.
        epoch (int, optional): Current epoch number.
        max_epochs (int, optional): Total number of epochs.
        lr (float, optional): Learning rate.
        phase (str): Phase name for progress bar.
        is_train (bool): Whether to run in training mode.

    Returns:
        tuple: (average loss, metrics dict or None)
    """
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    if metrics is not None:
        metrics.reset()
    desc = f"{phase} | Epoch {epoch}/{max_epochs} | LR {lr:.2e}" if epoch and max_epochs and lr is not None else phase
    pbar = tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)
    scaler = torch.amp.GradScaler(device="cuda") if device.type == 'cuda' else None
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        if is_train:
            optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, masks)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            with torch.set_grad_enabled(is_train):
                outputs = model(images)
                loss = criterion(outputs, masks)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item()
        if metrics is not None:
            out_detach = outputs.detach() if is_train else outputs
            mask_detach = masks.detach() if is_train else masks
            metrics.update(out_detach, mask_detach)
            metrics_dict = metrics.compute_metrics()
            postfix = {'loss': f'{loss.item():.4f}'}
            postfix.update({
                'f1': f"{metrics_dict.get('f1_mean', 0):.4f}",
                'IoU': f"{metrics_dict.get('iou_mean', 0):.4f}",
                'Prec': f"{metrics_dict.get('precision_mean', 0):.4f}",
                'Rec': f"{metrics_dict.get('recall_mean', 0):.4f}"
            })
            pbar.set_postfix(postfix)
        else:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    avg_loss = total_loss / max(1, len(loader))
    if metrics is not None:
        metrics_dict = metrics.compute_metrics()
        return avg_loss, metrics_dict
    return avg_loss, None

def validate(model, val_loader, criterion, device, metrics=None, epoch=None, max_epochs=None, lr=None, phase='Validation'):
    """
    Run one validation epoch.

    Args:
        model (torch.nn.Module): Model to validate.
        val_loader (DataLoader): Validation data loader.
        criterion (callable): Loss function.
        device (torch.device): Device to use.
        metrics (object, optional): Metrics object.
        epoch (int, optional): Current epoch.
        max_epochs (int, optional): Total epochs.
        lr (float, optional): Learning rate.
        phase (str): Phase name for progress bar.

    Returns:
        tuple: (average loss, metrics dict or None)
    """
    return _run_epoch(model, val_loader, criterion, device, None, metrics, epoch, max_epochs, lr, phase, is_train=False)

def evaluate_model(model, loader, device, metrics):
    """
    Evaluate a model on a dataset and compute metrics.

    Args:
        model (torch.nn.Module): Model to evaluate.
        loader (DataLoader): DataLoader for evaluation.
        device (torch.device): Device to use.
        metrics (object): Metrics object with reset, update, and compute_metrics methods.

    Returns:
        dict: Computed metrics.
    """
    model.eval()
    metrics.reset()
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            metrics.update(outputs, masks)
    return metrics.compute_metrics()

# Transfer Learning Helpers
def apply_transfer_learning(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Transfer Learning requested, but checkpoint not found: {checkpoint_path}")
        return
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        
        # Filter out mismatched shapes
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        mismatched_keys = set(state_dict.keys()) - set(filtered_dict.keys())
        
        model.load_state_dict(filtered_dict, strict=False)
        logger.info(f"Loaded pretrained weights from {checkpoint_path}. Skipped {len(mismatched_keys)} mismatched layers (e.g. final classifier).")
    except Exception as e:
        logger.error(f"Failed to load checkpoint '{checkpoint_path}': {e}")

def freeze_encoder_if_requested(model, freeze: bool):
    if not freeze:
        return
    # Determine the encoder/backbone part based on architecture
    encoder = None
    if hasattr(model, 'encoder'):
        # SMP models
        encoder = model.encoder
    elif hasattr(model, 'down_conv') and hasattr(model, 'middle_conv'):
        # UNet_original (actual implementation)
        model.down_conv.requires_grad_(False)
        model.middle_conv.requires_grad_(False)
        logger.info("Froze UNet_original encoder layers (down_conv, middle_conv).")
        return
    elif hasattr(model, 'backbone'):
        # Original DeepLab wrappers may have backbone
        encoder = model.backbone
    elif hasattr(model, 'base') and hasattr(model.base, 'backbone'):
        # DeepLabV3_original wrapper
        encoder = model.base.backbone
        
    if encoder is not None:
        encoder.requires_grad_(False)
        logger.info("Froze encoder weights.")
    else:
        logger.warning("Could not identify encoder to freeze for this model.")