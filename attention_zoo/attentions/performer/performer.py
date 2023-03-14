import math
import torch
import torch.nn as nn
from attentions.abstract_attention import AbstractAttention
from einops import rearrange
from performer_pytorch import FastAttention
from torch import Tensor
from typing import Tuple
from omegaconf import DictConfig


class PerformerAttention(AbstractAttention):
    def __init__(self, hpars: DictConfig, n: int, h: int, in_feat: int, out_feat: int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat=out_feat)
        self.model_params = hpars
        self.kernel_type = self.model_params.kernel_type
        self.dim_head = out_feat // h

        if self.kernel_type == "relu":
            self.attn_fn = FastAttention(dim_heads=self.dim_head, nb_features=self.model_params.rp_dim, causal=False,
                                         kernel_fn=nn.ReLU())
        elif self.kernel_type == "exp":
            self.attn_fn = FastAttention(dim_heads=self.dim_head, nb_features=self.model_params.rp_dim, causal=False,
                                         kernel_fn=torch.exp)

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask=None) -> Tuple[Tensor, Tensor]:
        b, h, n, f = Q.shape

        if mask is None:
            mask = torch.ones(1, n, device=Q.device).bool()

        output = self.attn_fn(
            Q / math.sqrt(math.sqrt(f)),
            K / math.sqrt(math.sqrt(f)) * mask[:, None, :, None],
            V * mask[:, None, :, None])

        # Merge the multiple heads into one
        output = rearrange(output, 'b h n d -> b n (h d)')

        # The use of the FastAttention does not return the attention matrix
        return output, None
