import numpy as np

def binary_focal_loss(predictions, targets, alpha=0.25, gamma=2.0, eps=1e-12):
    """
    Compute mean binary focal loss.

    :param predictions: array-like of predicted probabilities (shape: N,)
    :param targets: array-like of binary labels {0,1} (shape: N,)
    :param alpha: balancing factor
    :param gamma: focusing parameter
    :param eps: numerical stability constant
    :return: float (mean focal loss)
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Clip predictions to avoid log(0)
    predictions = np.clip(predictions, eps, 1.0 - eps)

    # p_t: probability of the true class
    p_t = np.where(targets == 1, predictions, 1 - predictions)

    # Focal loss
    loss = -alpha * (1 - p_t) ** gamma * np.log(p_t)

    return float(np.mean(loss))