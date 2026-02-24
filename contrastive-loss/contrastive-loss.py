import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    
    # Convert to numpy arrays (handles list input safely)
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)
    
    # Ensure batch dimension
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    if y.ndim == 0:
        y = y.reshape(1)
    
    # Euclidean distance
    d = np.linalg.norm(a - b, axis=1)
    
    # Loss components
    positive_loss = y * (d ** 2)
    negative_loss = (1 - y) * (np.maximum(0, margin - d) ** 2)
    
    loss = positive_loss + negative_loss
    
    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")