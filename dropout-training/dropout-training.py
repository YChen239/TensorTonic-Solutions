import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    if rng is None:
        rng = np.random

    # Keep probability
    keep_prob = 1 - p
    # conver to array
    x = np.array(x)
    
    # Sample mask: 1 with prob keep_prob, 0 otherwise
    if x.ndim == 1:
        mask = (rng.random(x.shape[0]) < keep_prob)
    if x.ndim == 2:
        mask = (rng.random((x.shape[0],x.shape[1])) < keep_prob)

    # Dropout pattern and output
    dropout_pattern = mask / keep_prob
    output = x * dropout_pattern

    return output, dropout_pattern
            
            