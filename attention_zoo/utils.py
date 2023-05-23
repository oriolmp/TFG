# This file implements several auxiliary methods that are useful for some of our provided implementations
import torch
from torch import Tensor
from einops import rearrange
import math

def iterative_inv(X: Tensor, iters: int = 6) -> Tensor:
    # This method computes the iterative penrose-inverse of a matrix
    device = X.device

    abs_x: Tensor = torch.abs(X)
    col: Tensor = abs_x.sum(dim = -1)
    row: Tensor = abs_x.sum(dim = -2)
    z: Tensor = rearrange(X, '... i j -> ... j i') / (torch.max(col[~torch.isnan(col)]) * torch.max(row[~torch.isnan(row)])) # ensure non nan values are used

    I: Tensor = torch.eye(X.shape[-1], device = device)
    I: Tensor = rearrange(I, 'i j -> () i j')

    out = z
    for i in range(iters):
        xz: Tensor = X @ z
        # print(f'xz {i}: {torch.isnan(xz).any()}')
        z: Tensor = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
        # print(f'z {i}: {torch.isnan(z).any()}')
        if not torch.isnan(z).any():
            out = z
        else:
            break     

    return out

def init_(tensor:Tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor