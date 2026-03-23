import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    # pass
    n_sample, n_features = np.shape(X)
    w = np.zeros(n_features)
    b = 0

    for _ in range(steps):
        z = np.dot(X, w) + b
        y_pred = _sigmoid(z)

        cost = 1/n_sample*np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

        del_w = -1/n_sample*np.sum(np.dot(X.T,(y_pred-y)),axis = 0)
        del_b = -1/n_sample*np.sum(y_pred-y)

        w += lr*del_w
        b += lr*del_b


    return w,b