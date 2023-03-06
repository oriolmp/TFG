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
    z: Tensor = rearrange(X, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I: Tensor = torch.eye(X.shape[-1], device = device)
    I: Tensor = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz: Tensor = X @ z
        z: Tensor = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))
    return z

def init_(tensor:Tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor