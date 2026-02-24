import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    
    # Convert to numpy arrays (fix for list input)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    N = y_true.shape[0]
    
    correct_class_probs = y_pred[np.arange(N), y_true]
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss