import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Numerical stability
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    N = y_pred.shape[0]

    log_probs = -np.log(y_pred[np.arange(N), y_true])
    return np.mean(log_probs)