import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    
    y_true : array-like
    y_pred : array-like
    delta  : threshold parameter
    
    Returns: mean Huber loss (float)
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    error = y_true - y_pred
    abs_error = np.abs(error)
    
    # Quadratic part (|e| <= delta)
    quadratic = 0.5 * error**2
    
    # Linear part (|e| > delta)
    linear = delta * (abs_error - 0.5 * delta)
    
    # Combine using condition
    loss = np.where(abs_error <= delta, quadratic, linear)
    
    return float(np.mean(loss))