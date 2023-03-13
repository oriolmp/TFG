import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from omegaconf import DictConfig, OmegaConf
import hydra

# Define architecture of the video transformer
# It will be composed of the following parts:
#   - Linear map
#   - N x encoder block (multihead attention + LN + MLP)
#   - MLP

@hydra.main(version=None, config_path='./configs', config_name='model_v1')
def cfg_setup(cfg: DictConfig):
    return cfg

class PatchTokenization(nn.Module):
    """ 
        Image to Patch Embedding
        Usually embed_dim = patch_size^2 * in_chans
    """
    def __init__(self, img_size=480, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = rearrange(x, 'b c h w -> b (h w) c') 
        return x, T, W


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)  # (B, N, C) -> (B, N, C * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, head, C/head, W, H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)  # @ operator? 
        # attn = attn * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
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
    def __init__(self, dim, num_heads, mlp_ratio=4., proj_drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads, proj_drop, attn_drop)
        self.temporal_fc = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer, drop=proj_drop)

    def forward(self, x):


class Model_v1(nn.Module):
    """
    First model backbone
    """
    def __init__(self, img_size=480, patch_size=16, in_chans=3, embed_dim=768, num_classes=97, num_heads=4, depth=2):
        super.__init__()
        self.depth = depth
        self.heads = 4
        self.num_classes = num_classes
        
        self.patch_embed = PatchTokenization(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )

        

