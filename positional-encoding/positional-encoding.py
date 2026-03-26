import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.power(base, np.arange(0, d_model, 2) / d_model)
    print(div_term)
    
    # Apply sine to even indices and cosine to odd indices
    PE[:, 0::2] = np.sin(position / div_term)
    if d_model%2:
        PE[:, 1::2] = np.cos(position / div_term[:-1])
    else:
        PE[:, 1::2] = np.cos(position / div_term)

    return PE