# This implementation is based on https://github.com/lucidrains/nystrom-attention

import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_zoo.attentions.abstract_attention import AbstractAttention
from einops import rearrange
from torch import Tensor
from torch import einsum
from attention_zoo.utils import iterative_inv
from typing import Tuple
from omegaconf import DictConfig

class CustomNystromAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars
        self.n = n
        self.eps = self.model_params.eps

        use_sample_perc = self.model_params.use_sample_perc
        if use_sample_perc:
            self.num_samples = int(self.n * (int(self.model_params.sampling_perc) / 100))
            assert False, "we want the other branch => don't specify  sampling_perc"
        else:
            self.num_samples = self.model_params.num_samples

        self.pinv_iterations = self.model_params.pinv_iterations

        # Other options
        self.symmetry = False  # self.model_params.symmetrization
        self.activation = self.model_params.activation_function
        self.use_final_norm = self.model_params.final_normalization

        if self.use_final_norm:
            self.final_norm = nn.LayerNorm(in_feat)

        # Parameters of the uniform sketching
        self.accumulation = 1
        self.nb_features = self.model_params.num_samples
        self.use_uniform_sketching = self.model_params.use_uniform_sketching

    # Compute a uniform sketch. This involves computing the sketching matrix
    @torch.no_grad()
    def uniform_sketching(self, Q:Tensor, K:Tensor, n:int, nb_rows:int, nb_columns:int, non_padding_num:int, device):
        b, h, n, p = Q.shape

        # Compute the sketching matrix
        total = nb_rows * nb_columns  # 1 x dim
        S = torch.rand(total, device=device)
        S = torch.einsum("b,d->bd", non_padding_num, S).long()
        S = S.reshape(-1, nb_rows, nb_columns)

        # Apply the sketch matrix S to the query and key matrices q and k
        Q = Q.transpose(1, 2)[torch.arange(b)[:, None, None], S].permute(0, 3, 1, 2, 4)  # bmdhp -> bhmdp
        K = K.transpose(1, 2)[torch.arange(b)[:, None, None], S].permute(0, 3, 1, 2, 4)  # bmdhp -> bhmdp

        Q = Q.squeeze(2)
        K = K.squeeze(2)

        return Q, K

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        b, h, n, d_head, m, iters, eps = *Q.shape, self.num_samples, self.pinv_iterations, self.eps

        # Apply symmetrization if needed
        if self.symmetry:
            assert False, "We don't want symmetry in this case"
            K = Q

        # Compute the matrix q_hat and k_hat
        if self.use_uniform_sketching:
            if mask is None:
                mask = torch.ones(1, self.original_n, device=q.device).bool()
                mask = rearrange(mask, 'b n -> b () n')

            non_padding_num = torch.squeeze(mask.sum(-1), 0)  # b
            Q_hat, K_hat = self.uniform_sketching(q, k, self.original_n, self.accumulation, self.nb_features,
                                                  non_padding_num, q.device)  # bmd
        else:
            Q_hat = Q[:, :, :m, :]
            K_hat = K[:, :, :m, :]

        # similarities (this holds for our case as well)
        einops_eq = '... i d, ... j d -> ... i j'  # this performs the matrix multiplication over the last dimension
        sim1 = einsum(einops_eq, Q, K_hat)  # Q x K_hat
        del Q
        sim3 = einsum(einops_eq, Q_hat, K)  # Q_hat x K
        del K
        sim2 = einsum(einops_eq, Q_hat, K_hat)  # Q_hat x K_hat

        # Apply the activation
        if self.activation == 'relu':
            attn1, attn2, attn3 = map(lambda t: F.relu(t), (sim1, sim2, sim3))
        elif self.activation == 'celu':
            attn1, attn2, attn3 = map(lambda t: F.celu(t), (sim1, sim2, sim3))
        else:
            attn1, attn2, attn3 = sim1, sim2, sim3

        attn2_inv = iterative_inv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ V)

        # Merge the multiple heads into one
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply the additional final normalization rela style BUT use vanilla layernorm
        if self.use_final_norm:
            out = self.final_norm(out)
        del V

        return out, None if not debug else (attn1 @ attn2_inv @ attn3)
