import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from einops import rearrange, reduce
from attentions.abstract_attention import AbstractAttention
from torch import Tensor
from typing import Tuple
from omegaconf import DictConfig

class CustomFullAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars
        self.dim = out_feat

        self.symmetry = self.model_params.symmetrization
        assert self.symmetry==False
        self.activation = self.model_params.activation_function
        self.use_final_normalization = self.model_params.final_normalization

        if self.use_final_normalization:
            self.final_norm = nn.LayerNorm(out_feat)

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        # Apply symmetrization if needed
        if self.symmetry:
            K = Q

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply the corresponding activation function
        if self.activation == 'softmax':
            scores = F.softmax(scores, dim=-1)
        elif self.activation == 'relu':
            scores = F.relu(scores)
        elif self.activation == 'celu':
            scores = F.celu(scores)

        output = torch.matmul(scores, V)

        # Merge the multiple heads into one
        output = rearrange(output, 'b h n d -> b n (h d)')

        if self.use_final_normalization:
            output = self.final_norm(output)

        return output, None if not debug else scores