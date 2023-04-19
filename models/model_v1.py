import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from attention_zoo.base_attention import BaseAttention
import math

from omegaconf import DictConfig, OmegaConf
import hydra

# Define architecture of the video transformer
# It will be composed of the following parts:
#   - Linear map
#   - N x encoder block (multihead attention + LN + MLP)
#   - MLP


class PatchTokenization(nn.Module):
    """
        Video to Patch Embedding.
        Applies Conv2d to transform the input video
        (batch, chanels, frames, width, height) -> (batches, frames*patches, embed_dim)
    """
    def __init__(self, img_size: int, patch_size: int, frames: int, in_chans: int, embed_dim: int):
        super().__init__()
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.frames = frames

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, T, _, W = x.shape
        x = rearrange(x, 'b c f w h -> (b f) c w h') # shape: (batch x frames, channels, img_w, img_h)
        x = self.proj(x)    # shape: (batch x frame, frames, patches_w, patches_h
        W = x.size(-1)
        x = rearrange(x, '(b f) d wp hp -> b (wp hp f) d', f=self.frames)     # shape: (batch, patches_w x patches_h x frames, channels) = (B, N, C)
        return x, T, W


class MultiHeadAttention(nn.Module):
    """
        Computes the Q, K, V values from the patch embeddings and computes attention.
        To select which attention we want to apply, the cfg file must have the variable
        ATTENTION set to one of the available models in the library attention_zoo.
    """
    def __init__(self, cfg: OmegaConf, dim: int, num_heads: int, num_patches: int, proj_drop: float, attn_drop: float):
        super().__init__()
        self.num_heads = num_heads
        self.attention = BaseAttention.init_att_module(cfg, in_feat=dim, out_feat=dim, n=num_patches, h=num_heads)
        # self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)  
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        _, _, D = x.shape
        qkv = self.qkv(x) # shape: (b, n, d * 3) = (b, n, d * a)
        qkv = rearrange(qkv, 'b n (a H d1) -> b n a H d1', H=self.num_heads, d1=D//self.num_heads) # shape: (b, n, 3, heads, D / heads)
        qkv = rearrange(qkv, 'b n a H d1 -> a b H n d1') # shape: (3, b, heads, n, D / heads)
        q, k, v = qkv[0], qkv[1], qkv[2]

        output = self.attention.apply_attention(Q=q, K=k, V=v)

        return output



class MLP(nn.Module):
    """
        Applies 2 linear layers after attention is computed.
        Should have as output a tensor of the same size as the patch tokens.
    """
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x 
 

class Block(nn.Module):
    """
        Implements the NL + MultiheadAttention + NL + MLP block
    """
    def __init__(self, cfg: OmegaConf, dim: int, num_heads: int, num_patches: int, mlp_ratio: float, proj_drop: float,
                 attn_drop:float, act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(cfg, dim, num_heads, num_patches, proj_drop, attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0] # [out, scores]
        x = x + self.mlp(self.norm2(x))
        return x
    
class PositionalEncoding(nn.Module):
    """
        Implements sinuisoidal positional encodiing.
        Source code can be found here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 9801):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_length, embedding_dim]`` 
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Model(nn.Module):
    """
    Model class with PatchTokenization + (MuliHeadAttention + MLP) x L + classification head
    """
    def __init__(self, cfg: OmegaConf, mlp_ratio: float = 4., proj_drop: float = 0., attn_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, dropout: float = 0.):
        super().__init__()

        self.num_classes = cfg.model.NUM_CLASSES
        self.depth = cfg.model.DEPTH
        self.patch_size = cfg.model.PATCH_SIZE
        self.num_heads = cfg.model.HEADS

        self.img_size = cfg.dataset.FRAME_SIZE
        self.in_chans = cfg.dataset.IN_CHANNELS
        self.num_frames = cfg.dataset.NUM_FRAMES
        self.batch_size = cfg.dataset.BATCH_SIZE

        self.num_features = self.embed_dim = (self.patch_size * self.patch_size) * self.in_chans
        self.dropout = nn.Dropout(dropout)

        self.patch_embed= PatchTokenization(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            frames=self.num_frames,
            embed_dim=self.embed_dim)

        self.num_patches = self.patch_embed.num_patches * self.patch_embed.frames

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Learnable positional embedding
        # self.pos_embed = nn.Parameter(torch.zeros(self.batch_size, self.num_patches+1, self.embed_dim))

        # Sinusoidal positional embedding
        self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=self.num_patches+1)

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(cfg, self.embed_dim, self.num_heads, self.num_patches, mlp_ratio, proj_drop, attn_drop, act_layer=nn.GELU, norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        x, _, _ = self.patch_embed(x)
        # add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # shape: (1, 1, embed) -> (batches, 1, embed)
        x = torch.cat((cls_tokens, x), dim=1) # shape: (batch, frames * patches + 1, embed)

        # add positional/temporal embedding
        # x = x + self.pos_embed # learnable
        x = self.pos_embed.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        
        # TODO: You are averaging all the patches of each of the videos.
        # TODO: Problem: You applied attention frame-wise. Also, this is what we use the CLS token for!
        x = x[:, -1] # Get cls token to classify
        x = self.head(x)
        return x

# Debugging code
'''
if __name__ == '__main__':
    # Define an input of shape 2x3x10x32x32 (BxCxTxHxW
    x = torch.randn(2, 3, 10, 32, 32)

    # Check the patchification
    patch_embed = PatchTokenization(img_size=32, patch_size=4, in_chans=3, embed_dim=48)
    print(patch_embed(x)[0].shape)
'''