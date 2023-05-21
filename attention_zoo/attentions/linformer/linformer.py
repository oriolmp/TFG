import torch
from attentions.abstract_attention import AbstractAttention
from einops import rearrange
from torch import Tensor
from torch import nn
from typing import Tuple
from omegaconf import DictConfig

class LinformerAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars.model

        self.to_k = nn.Linear(n+1, self.model_params.proj_feats, bias=False)
        self.to_v = nn.Linear(n+1, self.model_params.proj_feats, bias=False)


    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask=None) -> Tuple[Tensor, Tensor]:
        b, h, n, f = Q.shape

        # Reshape to apply the projection
        K = rearrange(K, 'b h n f -> b (f h) n')
        V = rearrange(V, 'b h n f -> b (f h) n')

        # Apply the low dimension projection of both the key and the values matrices
        K = self.to_k(K)
        V = self.to_v(V)

        # Reshape back into BxHxNxF
        K = rearrange(K, 'b (f h) k -> b h k f', f=f)
        V = rearrange(V, 'b (f h) k -> b h k f', f=f)


        # attention
        dots = torch.einsum('b h n d, b h k d -> b h n k', Q, K) * (f ** -0.5)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h n k, b h k d -> b h n d', attn, V)

        # Merge the multiple heads into one
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out, None if not debug else attn
