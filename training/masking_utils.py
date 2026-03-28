import torch
import torch.nn.functional as F

def generate_random_mask(images: torch.Tensor, mask_ratio: float, patch_size: int) -> torch.Tensor:
    """
    Generates a patch-based binary mask for MAE-style self-supervised learning.
    
    Args:
        images: Input tensor of shape (B, C, H, W)
        mask_ratio: Float representing the proportion of patches to mask (e.g., 0.75)
        patch_size: Integer size of the square patch (e.g., 16)
        
    Returns:
        mask: Boolean tensor of shape (B, 1, H, W) where True means the pixel is masked.
    """
    B, C, H, W = images.shape
    num_patches_y = H // patch_size
    num_patches_x = W // patch_size
    
    # Generate random noise for each patch
    noise = torch.rand(B, 1, num_patches_y, num_patches_x, device=images.device)
    
    # Calculate the threshold for masking
    k = int((num_patches_y * num_patches_x) * mask_ratio)
    if k == 0:
        return torch.zeros((B, 1, H, W), dtype=torch.bool, device=images.device)
        
    # Get top-k threshold per item in batch
    # topk returns the largest k elements. The mask ratio defines how many patches to mask (1/True).
    threshold = torch.topk(noise.view(B, -1), k, dim=-1, largest=True).values[:, -1].view(B, 1, 1, 1)
    
    # 1/True for patches to mask, 0/False for visible patches
    mask_patches = (noise >= threshold).float()
    
    # Upscale mask to full image size
    mask = F.interpolate(mask_patches, size=(H, W), mode='nearest')
    return mask.bool()

def generate_object_centric_mask(images: torch.Tensor, mask_ratio: float, patch_size: int) -> torch.Tensor:
    """
    Generates an object-centric patch mask. Uses a Sobel filter to find edges,
    biasing the mask noise to cover high-edge areas (objects).
    """
    B, C, H, W = images.shape
    num_patches_y = H // patch_size
    num_patches_x = W // patch_size
    
    # Calculate grayscale intensities
    gray = images.mean(dim=1, keepdim=True)
    
    # Sobel filters for edge detection
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=images.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=images.device).view(1, 1, 3, 3)
    
    # Apply filters (pad to maintain size)
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    
    # Compute edge magnitude
    edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
    
    # Pool edges to patch size to get edge density per patch
    patch_edges = F.avg_pool2d(edges, kernel_size=patch_size, stride=patch_size)
    
    # Normalize patch edges to use as bias (0 to 1)
    patch_edges_min = patch_edges.view(B, -1).min(dim=-1, keepdim=True)[0].view(B, 1, 1, 1)
    patch_edges_max = patch_edges.view(B, -1).max(dim=-1, keepdim=True)[0].view(B, 1, 1, 1)
    edge_bias = (patch_edges - patch_edges_min) / (patch_edges_max - patch_edges_min + 1e-6)
    
    # Generate random noise and combine with edge bias
    # Higher combined score means higher probability of being masked
    # Weight the edge bias strongly so it dominates the random noise
    noise = torch.rand(B, 1, num_patches_y, num_patches_x, device=images.device)
    combined_score = noise + 2.0 * edge_bias
    
    # Calculate threshold
    k = int((num_patches_y * num_patches_x) * mask_ratio)
    if k == 0:
        return torch.zeros((B, 1, H, W), dtype=torch.bool, device=images.device)
        
    threshold = torch.topk(combined_score.view(B, -1), k, dim=-1, largest=True).values[:, -1].view(B, 1, 1, 1)
    mask_patches = (combined_score >= threshold).float()
    
    mask = F.interpolate(mask_patches, size=(H, W), mode='nearest')
    return mask.bool()
