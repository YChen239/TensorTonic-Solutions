def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    for _ in range(steps):
        del_x = (2*a*x+b)
        x -= lr*del_x
    return x