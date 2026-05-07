import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    if op == "add":
        return (torch.tensor(x)+torch.tensor(y)).tolist()
    if op == "multiply":
        return (torch.tensor(x)*torch.tensor(y)).tolist()
    if op == "matmul":
        return (torch.tensor(x)@torch.tensor(y)).tolist()
    if op == "power":
        return (torch.tensor(x)**torch.tensor(y)).tolist()
    if op == "max":
        return (torch.maximum(torch.tensor(x), torch.tensor(y))).tolist()