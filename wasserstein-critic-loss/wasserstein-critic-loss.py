import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    
    real_scores : array-like - critic outputs for real samples
    fake_scores : array-like - critic outputs for fake samples
    
    Returns: float
    """
    
    real_scores = np.asarray(real_scores)
    fake_scores = np.asarray(fake_scores)
    
    loss = np.mean(fake_scores) - np.mean(real_scores)
    
    return float(loss)