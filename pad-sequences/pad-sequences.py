import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)
    ret = np.full((len(seqs),max_len), pad_value)
    for i in range(len(seqs)):
        for j in range(min(len(seqs[i]),max_len)):
            ret[i][j] = seqs[i][j]
    return ret
    