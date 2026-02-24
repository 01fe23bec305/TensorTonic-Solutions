import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    
    anchor, positive, negative:
        shape (D,) or (N, D)
    margin: float
    
    Returns: float (mean triplet loss if batch)
    """
    
    anchor = np.asarray(anchor)
    positive = np.asarray(positive)
    negative = np.asarray(negative)
    
    # Ensure batch dimension
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)
    
    # Squared L2 distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)
    
    # Triplet loss
    losses = np.maximum(0.0, d_ap - d_an + margin)
    
    return float(np.mean(losses))