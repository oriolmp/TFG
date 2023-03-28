# This implementation is based on https://github.com/lucidrains/nystrom-attention

from math import ceil
import torch
import torch.nn.functional as F
from attentions.abstract_attention import AbstractAttention
from einops import rearrange, reduce
from torch import Tensor
from torch import einsum
from utils import iterative_inv
from typing import Tuple
from omegaconf import DictConfig


# -------------------------------------------
# main attention class
class NystromformerAttention(AbstractAttention):
    def __init__(self, hpars: DictConfig, n: int, h: int, in_feat: int, out_feat: int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat=out_feat)
        self.model_params = hpars

        self.n_orig = None
        self.eps = self.model_params.eps
        self.num_landmarks = self.model_params.num_landmarks
        self.pinv_iterations = self.model_params.pinv_iterations

    # This is an optional function that if overwritten, it adds the necessary padding
    def pad_input(self, x: Tensor) -> Tensor:
        b, n, d, f, m = *x.shape, self.num_landmarks
        remainder = n % m

        self.original_dim = n
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

        self.n_orig = n

        return x

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask=None) -> Tuple[Tensor, Tensor]:
        b, h, n, d_head, m, iters, eps = *Q.shape, self.num_landmarks, self.pinv_iterations, self.eps

        # If necessary, add padding to the embeddings to be divisible
        # Q,K,V = map(lambda t: self.pad_input(t), (Q, K, V))

        # set masked positions to 0 in queries, keys, values
        if mask is not None:
            mask = rearrange(mask, 'b n -> b () n')
            Q, K, V = map(lambda t: t * mask[..., None], (Q, K, V))
        Q *= (d_head ** -0.5)

        # generate landmarks by sum reduction, and then calculate mean using the mask
        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(Q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(K, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean
        divisor = l
        if mask is not None:
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)
        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities
        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, Q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, K)

        # masking
        if mask is not None:
            mask_value = -torch.finfo(Q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = iterative_inv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ V)

        # Merge the multiple heads into one
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = out[:, -n:, :]  # Select only last n to get rid of the padded ones

        return out, None if not debug else (attn1 @ attn2_inv @ attn3)
