import math

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.

    :param predictions: list or iterable of predicted probabilities (length K)
    :param target: index of the true class
    :param epsilon: label smoothing factor
    :return: scalar loss value
    """
    K = len(predictions)
    loss = 0.0

    for i, p_i in enumerate(predictions):
        if i == target:
            q_i = (1 - epsilon) + epsilon / K
        else:
            q_i = epsilon / K

        loss -= q_i * math.log(p_i)

    return loss