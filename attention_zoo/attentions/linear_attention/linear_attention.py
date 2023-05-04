# This implementation is based on https://github.com/lucidrains/linear-attention-transformer

import torch
from torch import einsum
from attentions.abstract_attention import AbstractAttention
from einops import rearrange
from torch import Tensor
from typing import Tuple
from omegaconf import DictConfig

class LinearAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars.model

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask=None) -> Tuple[Tensor, Tensor]:
        dim = Q.shape[-1]

        if mask is not None:
            mask_value = -torch.finfo(Q.dtype).max
            mask = mask[:, None, :, None]
            K = K.masked_fill_(~mask, mask_value)
            V = V.masked_fill_(~mask, 0.)
            del mask

        Q = Q.softmax(dim=-1)  # Apply the normalization to the query matrix
        K = K.softmax(dim=-2)  # Apply the normalization to the query matrix

        Q = Q * dim ** -0.5  # Scale the query matrix

        context = einsum('b h n d, b h n e -> b h d e', K, V)  # Apply p_k(K)^T V
        output = einsum('b h n d, b h d e -> b h n e', Q, context)  # Finally apply p_q(Q) (p_k(K)^T V)

        # Merge the multiple heads into one
        output = rearrange(output, 'b h n d -> b n (h d)')

        return output, None if not debug else einsum('... n d, ... m d -> ... n m', Q, K)
