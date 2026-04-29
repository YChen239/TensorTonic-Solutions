import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    """
    Returns: list of length k with tuples (train_idx, val_idx)

    Args:
        N (int): number of samples
        k (int): number of folds
        shuffle (bool): whether to shuffle indices
        rng (np.random.Generator or None): random generator
    """
    if k <= 1 or k > N:
        raise ValueError("k must be in range [2, N]")

    indices = np.arange(N)

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)

    # Compute fold sizes (handle uneven splits)
    fold_sizes = np.full(k, N // k)
    fold_sizes[:N % k] += 1  # distribute remainder

    splits = []
    current = 0

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))

        splits.append((train_idx, val_idx))
        current = stop

    return splits
