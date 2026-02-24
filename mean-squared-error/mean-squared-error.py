import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    
    # Convert to numpy arrays (handles list input safely)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # Compute MSE
    mse = np.mean((y_pred - y_true) ** 2)
    
    return mse