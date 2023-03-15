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
    def __init__(self, config, dim, num_heads=8, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attention = BaseAttention.init_att_module(cfg, in_feat=dim, out_feat=dim, n=dim, h=dim)
        # self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)  # (B, N, C) -> (B, N, C * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        print(f'qkv: {self.qkv(x).shape}')
        qkv = rearrange(qkv, 'b n (c h w) -> b n c h w', h=self.num_heads, w=C//self.num_heads)
        print(f'qkv reshaped: {qkv.shape}')
        qkv = rearrange(qkv, 'b n c h w -> c b h n w')
        print(f'qkv reshaped and permuted: {qkv.shape}')
        q, k, v = qkv[0], qkv[1], qkv[2]

        output = self.attention.apply_attention(Q=q, K=k, V=v)
        return output


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
    def __init__(self, cfg, dim, num_heads, mlp_ratio=4., proj_drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(cfg, dim, num_heads, proj_drop, attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    """
    Model class with PatchTokenization + (MuliHeadAttention + MLP) x L + MLP
    """
    def __init__(self, cfg, img_size=240, patch_size=16, in_chans=3, embed_dim=768, num_classes=97, depth=2, num_heads=4, mlp_ratio=4.,
                 proj_drop=0., attn_drop=0., norm_layer=nn.LayerNorm, num_frames=30, dropout=0.):
        super().__init__()
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.patch_embed= PatchTokenization(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(num_frames, num_patches+1, embed_dim))
        # self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
                                       
        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(cfg, embed_dim, num_heads, mlp_ratio, proj_drop, attn_drop, act_layer=nn.GELU, norm_layer=norm_layer)
            for i in range(self.depth)])                            
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x, T, W = self.patch_embed(x)
        
        # add class token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) # (1, 1, embed) -> (30, 1, embed)
        x = torch.cat((cls_tokens, x), dim=1) # (batch x frames, patches, embed) -> (batch x frames, patches + 1, embed)
    
        # add positional/temporal embedding
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block.forward(x)
        x = rearrange(x, '(b f) p e -> b f p e', f=self.num_frames) # (batch x frames, patches, embed) -> (batch, frames, patch, embed)
        x = torch.mean(x, [1,2])
        x = self.head(x)
        return x   





