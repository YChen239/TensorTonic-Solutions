def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    if len(y_true) == 0:
        return 0

    correct = sum(1 for yp, yt in zip(y_true, y_pred) if yp==yt)
    return correct/len(y_true)