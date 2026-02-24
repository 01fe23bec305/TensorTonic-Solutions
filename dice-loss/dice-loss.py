import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    
    p: predicted mask (probabilities or binary), shape (N, ...)  
    y: ground truth mask (binary), same shape as p  
    eps: smoothing constant for numerical stability
    
    Returns: scalar Dice loss
    """
    
    # Convert to numpy arrays (safe if lists passed)
    p = np.asarray(p)
    y = np.asarray(y)
    
    # Flatten to compute global Dice (works for any dimension)
    p_flat = p.reshape(-1)
    y_flat = y.reshape(-1)
    
    # Compute intersection and sums
    intersection = np.sum(p_flat * y_flat)
    sum_p = np.sum(p_flat)
    sum_y = np.sum(y_flat)
    
    # Dice coefficient
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)
    
    # Dice loss
    return 1 - dice