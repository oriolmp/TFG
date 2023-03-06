# Script based on the original implementation: https://github.com/OpenNLPLab/cosFormer

import torch
import numpy as np
from torch import nn
from attention_zoo.attentions.abstract_attention import AbstractAttention
from einops import rearrange
from torch import Tensor
from typing import Tuple
from omegaconf import DictConfig


class CosformerAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars
        self.eps = self.model_params.eps

    def get_index(self, seq_len:int):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        # Query (q), Key (k) and Value (v) matrices are of shape: (b x h x n x d)

        b, h, tgt_len, d = Q.size()  # tgt_len, bsz, embed_dim = query.size()
        src_len = K.size(2)  # this is N

        # multihead reshape
        Q = rearrange(Q, 'b h n d -> (b h) n d')
        K = rearrange(K, 'b h n d -> (b h) n d')
        V = rearrange(V, 'b h n d -> (b h) n d')

        # cos transform
        m = max(src_len, tgt_len)

        # get index and send to cuda
        weight_index = self.get_index(m).to(Q)

        # (N * h, L, 2 * d)
        q_ = torch.cat(
            [Q * torch.sin(weight_index[:, :tgt_len, :] / m), Q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)

        # (N * h, S, 2 * d)
        k_ = torch.cat(
            [K * torch.sin(weight_index[:, :src_len, :] / m), K * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        kv_ = torch.einsum('nld,nlm->ndm', k_, V)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', q_, torch.sum(k_, axis=1)), self.eps)
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', q_, kv_, z_)

        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = rearrange(attn_output, '(b h) n d -> b n (h d)', b=b)

        return attn_output, None
