import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).

    p : array-like, shape (N,) - true probability distribution
    q : array-like, shape (N,) - predicted probability distribution
    eps : float - numerical stability epsilon

    Returns: float
    """
    p = np.asarray(p)
    q = np.asarray(q)

    # Numerical stability
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # KL divergence
    kl = np.sum(p * np.log(p / q))

    return float(kl)