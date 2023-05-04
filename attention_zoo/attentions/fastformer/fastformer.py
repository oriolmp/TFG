# This implementation is based on https://github.com/lucidrains/fast-transformer-pytorch

from attentions.abstract_attention import AbstractAttention
from einops import rearrange, reduce
from torch import Tensor
from torch import nn, einsum
from typing import Tuple
from omegaconf import DictConfig

class FastFormerAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars.model
        self.dim_head = in_feat // h

        self.to_q_attn_logits = nn.Linear(self.dim_head, 1,
                                          bias=False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(self.dim_head, 1, bias=False)  # for projecting keys to key attention logits

        # final transformation of values to "r" as in the paper
        self.to_r = nn.Linear(self.dim_head, self.dim_head)

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False) -> None:
        b,h,n,f = Q.shape

        # calculate query attention logits
        q_attn_logits = rearrange(self.to_q_attn_logits(Q), 'b h n () -> b h n') * (f ** -0.5)
        q_attn = q_attn_logits.softmax(dim=-1)

        # calculate global query token
        global_q = einsum('b h n, b h n d -> b h d', q_attn, Q)
        global_q = rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token
        K = K * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension
        if self.model_params.use_rotary_emb:
            K = reduce(K, 'b h n (d r) -> b h n d', 'sum', r=2)

        # now calculate key attention logits
        k_attn_logits = rearrange(self.to_k_attn_logits(K), 'b h n () -> b h n') * (self.dim_head ** -0.5)
        k_attn = k_attn_logits.softmax(dim=-1)

        # calculate global key token
        global_k = einsum('b h n, b h n d -> b h d', k_attn, K)
        global_k = rearrange(global_k, 'b h d -> b h () d')

        # bias the values
        u = V * global_k

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension
        if self.model_params.use_rotary_emb:
            u = reduce(u, 'b h n (d r) -> b h n d', 'sum', r=2)

        # transformation step
        r = self.to_r(u)

        # paper then says to add the queries as a residual
        r = r + Q

        # Merge the multiple heads into one
        r = rearrange(r, 'b h n d -> b n (h d)')

        # Not sure what we consider the attention matrix in this case
        # Global q and k, or attn_q and attn_k??
        return r, global_k
