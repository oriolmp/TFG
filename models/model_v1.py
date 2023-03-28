import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from attention_zoo.base_attention import BaseAttention

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
        (batch, chanels, frames, width, height) -> (frames, patches, embed_dim)
    """
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        _, _, T, _, W = x.shape
        x = rearrange(x, 'b c t w h -> (b t) c w h')
        x = self.proj(x)    # shape: BT x F x patches_w x patches_h
        W = x.size(-1)
        x = rearrange(x, 'b c w h -> b (w h) c')     # TODO: Check because this has shape (BT, total_num_patches, F)
        return x, T, W


class MultiHeadAttention(nn.Module):
    """
        Computes the Q, K, V values from the patch embeddings and computes attention.
        To select which attention we want to apply, the cfg file must have the variable
        ATTENTION set to one of the available models in the library attention_zoo.
    """
    def __init__(self, cfg: OmegaConf, dim: int, num_heads: int, proj_drop: float, attn_drop: float):
        super().__init__()
        self.num_heads = num_heads
        self.attention = BaseAttention.init_att_module(cfg, in_feat=dim, out_feat=dim, n=dim, h=dim)
        # self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)  # (B, N, C) -> (B, N, C * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        _, _, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (c h w) -> b n c h w', h=self.num_heads, w=C//self.num_heads)
        qkv = rearrange(qkv, 'b n c h w -> c b h n w')
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
    def __init__(self, cfg: OmegaConf, dim: int, num_heads: int, mlp_ratio: float, proj_drop: float,
                 attn_drop:float, act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(cfg, dim, num_heads, proj_drop, attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0] # [out, scores]
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    """
    Model class with PatchTokenization + (MuliHeadAttention + MLP) x L + classification head
    """
    def __init__(self, cfg: OmegaConf, mlp_ratio: float = 4., proj_drop: float = 0., attn_drop: float = 0.,
                 norm_layer: nn.Module = nn.LayerNorm, num_frames: int = 30, dropout: float = 0.):
        super().__init__()

        self.num_classes = cfg.NUM_CLASSES
        self.img_size = cfg.FRAME_SIZE
        self.patch_size = cfg.PATCH_SIZE
        self.in_chans = cfg.IN_CHANNELS
        self.depth = cfg.DEPTH
        self.num_heads = cfg.HEADS

        self.num_features = self.embed_dim = (self.patch_size * self.patch_size) * self.in_chans
        self.dropout = nn.Dropout(dropout)

        self.num_frames = num_frames # TODO: How do we handle variable number of frames

        self.patch_embed= PatchTokenization(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim)

        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(num_frames, num_patches+1, self.embed_dim))
        # self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(cfg, self.embed_dim, self.num_heads, mlp_ratio, proj_drop, attn_drop, act_layer=nn.GELU, norm_layer=norm_layer)
            for i in range(self.depth)])
        self.norm = norm_layer(self.embed_dim)

        # Classifier head
        self.head = nn.Linear(self.embed_dim, self.num_classes)

    def forward(self, x):
        x, _, _ = self.patch_embed(x)
        # add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # (1, 1, embed) -> (30, 1, embed)
        x = torch.cat((cls_tokens, x), dim=1) # (batch x frames, patches, embed) -> (batch x frames, patches + 1, embed)    # TODO: We want video-level sequences, so shape B x TP+1 x F
        # add positional/temporal embedding
        x = x + self.pos_embed
        for block in self.blocks:
            x = block.forward(x)
        x = rearrange(x, '(b f) p e -> b f p e', f=self.num_frames) # (batch x frames, patches, embed) -> (batch, frames, patch, embed)

        # TODO: You are averaging all the patches of each of the videos.
        # TODO: Problem: You applied attention frame-wise. Also, this is what we use the CLS token for!
        x = torch.mean(x, [1,2])
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