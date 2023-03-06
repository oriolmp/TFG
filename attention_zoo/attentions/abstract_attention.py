import torch.nn as nn
from torch import Tensor
import abc
from einops import rearrange
import torch

class AbstractAttention(nn.Module):
    def __init__(self, n:int, h:int, in_feat:int, out_feat:int):
        super().__init__()
        self.n = n
        self.h = h
        self.in_feat = in_feat
        self.out_feat = out_feat

        # Initialize the three projection matrices producing Q,K and V
        self.q_proj = nn.Linear(self.in_feat, self.out_feat)
        self.k_proj = nn.Linear(self.in_feat, self.out_feat)
        self.v_proj = nn.Linear(self.in_feat, self.out_feat)


    @abc.abstractmethod
    def apply_attention(self, Q:Tensor, K:Tensor, V:Tensor, debug:bool=False, mask:Tensor=None) -> Tensor:
        # This is an abstract method used for each of the subclasses to implement the specific attention logic
        # Methods will return the output of the attention, and the attention matrices if debug flag is set to True (if possible)
        pass

    def forward(self, x1:Tensor, x2:Tensor=None, x3:Tensor=None) -> Tensor:
        # This method takes as input x1, and optionally x2 and x3. It then applies the corresponding attention mechanism.
        # If x2 and x3 are specified, it then assumes that the projections are all applied over the single input x (normal attention)
        # If x2 and x3 are specified, this corresponds to a cross-attention method, where x1, x2 and x3, are used to obtain Q,K,V, respectively.
        # All the input matrices must of shape (B x N x F), this being Batch, Heads, Num of tokens, and num of Features
        Q: Tensor = self.q_proj(x1)
        K: Tensor = self.k_proj(x2) if x2 is not None else self.k_proj(x1)
        V: Tensor = self.v_proj(x3) if x3 is not None else self.v_proj(x1)

        # Transform the input into BxHxNxF
        Q,K,V = map(lambda t: rearrange(t, 'b n (h f) -> b h n f', h=self.h), (Q, K, V))

        out = self.apply_attention(Q, K, V)
        return out
