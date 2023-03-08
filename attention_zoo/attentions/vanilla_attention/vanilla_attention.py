import math
import torch
import torch.nn.functional as F
from attentions.abstract_attention import AbstractAttention
from einops import rearrange
from torch import Tensor
from typing import Tuple
from omegaconf import DictConfig

class VanillaAttention(AbstractAttention):
    def __init__(self, hpars:DictConfig, n:int, h:int, in_feat:int, out_feat:int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat = out_feat)
        self.model_params = hpars
        self.dim = out_feat

    def apply_attention(self, Q:Tensor, K:Tensor, V:Tensor, debug:bool=False, mask:Tensor=None) -> Tuple[Tensor, Tensor]:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim)
        scores= F.softmax(scores, dim=-1)

        output= torch.matmul(scores, V)

        # Merge the multiple heads into one
        output = rearrange(output, 'b h n d -> b n (h d)')

        return output, scores if debug else None
