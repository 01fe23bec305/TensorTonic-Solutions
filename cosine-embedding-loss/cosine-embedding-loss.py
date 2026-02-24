import numpy as np

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    
    x1, x2 : array-like vectors
    label  : 1 (similar) or -1 (dissimilar)
    margin : float >= 0
    
    Returns: float
    """
    
    # Convert to numpy arrays
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    
    # Compute cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    
    cos_sim = dot / (norm1 * norm2)
    
    # Compute loss based on label
    if label == 1:
        # Similar pair
        loss = 1 - cos_sim
    elif label == -1:
        # Dissimilar pair
        loss = max(0.0, cos_sim - margin)
    else:
        raise ValueError("label must be 1 or -1")
    
    return float(loss)