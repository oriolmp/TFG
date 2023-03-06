# This implementation is greatly inspired by https://github.com/pkuzengqi/Skyformer

from functools import partial
import torch
from attention_zoo.attentions.abstract_attention import AbstractAttention
from einops import rearrange
from torch import Tensor
from torch import nn
from attention_zoo.utils import iterative_inv
from collections.abc import Callable
from typing import Tuple
from omegaconf import DictConfig

# Constants
DATA_NORMALIZER = (32 ** -0.25)


class Kernel(nn.Module):
    # This class implements a general kernel function. Inside we implement a series of possible kernel functions that can be chosen by the user
    def __init__(self, kernel_type: str):
        super(Kernel, self).__init__()
        self.kernel_type: str = kernel_type

    def kernel_sketch(self, Q: Tensor, K: Tensor, *, kernel_fn: Callable, sketching_matrix: Tensor, random_sign,
                      normalize_data: bool = False, eps: float = 1e-4):
        # This method applies the kernel sketch on two input matrices Q and K
        b, h, n, p = Q.shape

        X = torch.cat([Q, K], dim=2)

        XS = X.transpose(1, 2)[torch.arange(b)[:, None, None], sketching_matrix].permute(0, 3, 1, 2,
                                                                                         4)  # bmdhp -> bhmdp

        # Compute the kernel only on a subst of elements, so that the complexity is still linear (we use sketching)
        # Q_S | K_S
        AS = kernel_fn(X, XS, True, random_sign)

        return AS.type_as(Q)

    def compute_kernel(self, X1: Tensor, X2: Tensor = None, X2_accu: Tensor = False,
                       random_sign= None) -> Tensor:
        random_sign = torch.tensor(random_sign, device=X1.device)

        if self.kernel_type == 'SM':
            return self.__kernel_SM(X1, X2, X2_accu, random_sign)
        elif self.kernel_type == 'RS_SM':
            return self.__kernel_RS_SM(X1, X2, X2_accu, random_sign)
        elif self.kernel_type == 'RS_SM':
            return self.__kernel_RS_SM1(X1, X2, X2_accu, random_sign)
        elif self.kernel_type == 'RELU':
            return self.__kernel_RELU(X1, X2, X2_accu, random_sign)
        elif self.kernel_type == 'RBF':
            return self.__kernel_RS_RBF(X1, X2, X2_accu, random_sign)

    def __kernel_SM(self, X1: Tensor, X2: Tensor = None, X2_accu: Tensor = False, random_sign= None) -> Tensor:
        if X2 is None:
            X2 = X1
            X2_accu = False
        if X2_accu:
            product = torch.einsum('...np,...mdp->...mnd', X1, X2)
            result = torch.exp(product)

            result = result.sum(dim=2)
        else:
            product = torch.einsum('...np,...dp->...nd', X1, X2)
            result = torch.exp(product)

        return result

    def __kernel_RS_SM(self, X1: Tensor, X2: Tensor = None, X2_accu: Tensor = False,
                       random_sign = None) -> Tensor:
        # RS for random sign
        if X2 is None:
            X2 = X1
            X2_accu = False
        if X2_accu:
            product = torch.einsum('...np,...mdp->...mnd', X1, X2)

            result = torch.exp(product)
            result = torch.transpose(result, 2, 3)  # nmd
            result = result * random_sign
            result = result.sum(dim=3)

        else:
            product = torch.einsum('...np,...dp->...nd', X1, X2)
            result = torch.exp(product)

        return result

    def __kernel_RS_SM1(self, X1: Tensor, X2: Tensor = None, X2_accu: Tensor = False,
                        random_sign = None) -> Tensor:

        if X2 is None:
            X2 = X1
            X2_accu = False
        if X2_accu:
            product = torch.einsum('...np,...mdp->...nmd', X1, X2)
            result = torch.exp(product)
            result = torch.einsum('bhnmd,...bmd->...bhnd', result, random_sign)
        else:
            product = torch.einsum('...np,...dp->...nd', X1, X2)
            result = torch.exp(product)
        return result

    def __kernel_RELU(self, X1: Tensor, X2: Tensor = None, X2_accu: Tensor = False,
                      random_sign = None) -> Tensor:
        aux = torch.einsum('...np,...mdp->...mnd', X1, X2)
        aux = torch.squeeze(aux, 2)
        return nn.functional.relu(aux)

    def __kernel_RS_RBF(self, X1: Tensor, X2: Tensor = None, X2_accu: Tensor = False,
                        random_sign = None) -> Tensor:

        # todo
        if X2 is None:
            X2 = X1
            X2_accu = False

        diag_X1 = (X1 * X1).sum(-1) * 0.5
        diag_X1 = diag_X1.unsqueeze(dim=-1)
        diag_X2 = (X2 * X2).sum(-1) * 0.5
        diag_X2 = diag_X2.unsqueeze(dim=-2)

        if X2_accu:
            diag_X1 = diag_X1.unsqueeze(dim=-3)
            product = torch.einsum('...np,...mdp->...mnd', X1, X2) - diag_X1 - diag_X2
            result = torch.exp(product)
            result = torch.transpose(result, 2, 3)  # nmd
            result = torch.einsum('bhnmd,bmd->bhnd', result, random_sign)
        else:
            product = torch.einsum('...np,...dp->...nd', X1, X2) - diag_X1 - diag_X2  # shape: b x h x 2n x d
            result = torch.exp(product)
        return result


class SkyformerAttention(AbstractAttention):
    def __init__(self, hpars: DictConfig, n: int, h: int, in_feat: int, out_feat: int) -> None:
        super().__init__(n=n, h=h, in_feat=in_feat, out_feat=out_feat)
        self.model_params = hpars
        self.accumulation = self.model_params.accumulation
        self.nb_features = self.model_params.num_feats
        self.kernel = Kernel('RBF')

    @torch.no_grad()
    def uniform_sketching(self, n, nb_rows, nb_columns, non_padding_num, device):
        # This method computes the uniform sketching matrix of a given input matrix size
        total = nb_rows * nb_columns  # 1 x dim
        S = torch.rand(total, device=device)

        S = torch.einsum("b,d->bd", non_padding_num, S).long()
        S[:, total // 2:] = S[:, total // 2:] + n
        S = S.reshape(-1, nb_rows, nb_columns)

        random_sign = torch.ones(S.shape)

        return S, random_sign

    def apply_attention(self, Q: Tensor, K: Tensor, V: Tensor, debug: bool = False, mask: Tensor = None) -> Tuple[
        Tensor, Tensor]:
        # get the shapes
        b, h, n, d = Q.shape

        if mask is None:
            mask = torch.ones(1, n, device=Q.device).bool()

        # apply the mask if exists
        mask = rearrange(mask, 'b n -> b () n')
        Q = Q * (mask[..., None] * DATA_NORMALIZER)
        K = K * (mask[..., None] * DATA_NORMALIZER)
        V = V * mask[..., None]

        non_padding_num = torch.squeeze(mask.sum(-1), 0)  # b

        # Compute the uniform sketching matrix S
        self.sketching_matrix, self.random_sign = self.uniform_sketching(
            n, self.accumulation, self.nb_features, non_padding_num, Q.device)  # bmd

        self.random_sign = torch.tensor(self.random_sign, device=Q.device)

        # Create the kernel function
        create_kernel_sketch = partial(self.kernel.kernel_sketch, kernel_fn=self.kernel.compute_kernel,
                                       sketching_matrix=self.sketching_matrix, random_sign=self.random_sign)

        # Complete the kernel matrix
        AS = create_kernel_sketch(Q, K)  # b,h,2n, nb_feats (BS)

        # Extract the completed query matrix Q_S
        Q = AS[:, :, :n]  # b, h, n, nb_feat

        # Extract the completed key matrix  K_S
        K = AS[:, :, n:]

        # Apply the sketching matrix (shape: b x h x 2n x d -> b x h x d x d)
        STAS = AS.transpose(1, 2)[torch.arange(b)[:, None, None], self.sketching_matrix]  # bnhd -> bmdhd
        STAS = torch.einsum('bmdhe,bmd->bhde', STAS, self.random_sign)  # bmdhd -> bhdd

        shape_identify = torch.eye(STAS.shape[-1], device=Q.device)

        STAS = STAS + 1e-1 * shape_identify

        ##################################################################
        D_STAS_inv = 1 / STAS.sum(-1)
        D_STAS_inv = torch.sqrt(D_STAS_inv)
        STAS = torch.einsum("...d,...de,...e->...de", D_STAS_inv, STAS, D_STAS_inv)  # S^T @ B @ S

        STAS_inv = iterative_inv(STAS, 6)  # (S^T @ B @ S)^{-1}    # shape: b x h x d x d

        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv) @ STAS_inv  # shape: b x h x n x d
        K = torch.einsum("...nd,...d->...nd", K, D_STAS_inv)
        ##################################################################

        # This order of computation avoid fully materializing the attention scores
        context = torch.einsum('...nd,...ne->...de', K, V)
        out = torch.matmul(Q, context)  # output of shape batch x head x num_tokens x head_dim

        # Merge the multiple heads into one
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out, None if not debug else torch.einsum('bhnd,bhmd->bhnm', Q, K)
