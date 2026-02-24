import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    
    p: predicted probabilities, shape (N,)
    y: binary labels {0,1}, shape (N,)
    gamma: focusing parameter (>=0)
    
    Returns: mean focal loss
    """
    
    # Convert to numpy arrays (safe if lists are passed)
    p = np.asarray(p)
    y = np.asarray(y)
    
    # Compute focal loss
    loss = -((1 - p) ** gamma) * y * np.log(p) \
           - (p ** gamma) * (1 - y) * np.log(1 - p)
    
    return np.mean(loss)