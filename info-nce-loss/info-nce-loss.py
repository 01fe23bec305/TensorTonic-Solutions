import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    
    N = Z1.shape[0]
    
    # Similarity matrix
    S = (Z1 @ Z2.T) / temperature
    
    # Log-sum-exp stabilization
    S_max = np.max(S, axis=1, keepdims=True)
    S_shifted = S - S_max
    
    exp_S = np.exp(S_shifted)
    
    # log( exp(S_ii) / sum(exp(S_ij)) )
    log_probs = np.diag(S_shifted) - np.log(np.sum(exp_S, axis=1))
    
    loss = -np.mean(log_probs)
    
    return float(loss)