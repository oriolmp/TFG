# This implementation is based on https://github.com/scaomath/galerkin-transformer

import copy
from typing import Tuple

import torch
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor
from torch import nn, einsum

from attentions.abstract_attention import AbstractAttention


class GalerkinAttention(AbstractAttention):
    def __init__(self, hpars: DictConfig, n: int, h: int, in_feat: int, out_feat: int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat=out_feat)
        self.model_params = hpars.model
        self.n_head = h
        self.dim_head = out_feat // h

        # Key and Value LN
        self.norm_k = nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(self.dim_head)) for _ in range(self.n_head)])
        self.norm_v = nn.ModuleList(
            [copy.deepcopy(nn.LayerNorm(self.dim_head)) for _ in range(self.n_head)])

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask=None) -> Tuple[Tensor, Tensor]:
        b, h, n, f = Q.shape

        # Normalize the k and value matrices (one layer normalization per head)
        K = torch.stack([norm(x) for norm, x in
                         zip(self.norm_k, (K[:, i, ...] for i in range(n)))], dim=1)
        V = torch.stack([norm(x) for norm, x in
                         zip(self.norm_v, (V[:, i, ...] for i in range(n)))], dim=1)

        # K @ V: shape b x h x d x d
        kv_scores = einsum('b h n d, b h n f -> b h d f', K, V)

        output = einsum('b h n d, b h d f -> b h n f', Q, kv_scores)

        # Apply the final normalization
        output /= n

        # Merge the multiple heads into one
        output = rearrange(output, 'b h n d -> b n (h d)')

        return output, None
