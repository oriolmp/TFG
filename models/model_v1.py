import torch
import torch.nn as nn

# Define architecture of the video transformer
# It will be composed of the following parts:
#   - Linear map
#   - N x encoder block (multihead attention + LN + MLP)
#   - MLP


class LinearMap(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


class MultiHeadAttention(nn.module):
    def __init__(self, dim, num_heads=8, proj_drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)  # (W, H, C) -> (W, H, C * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        W, H, C = x.shape()
        qkv = self.qkv(x).reshape(W, H, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (3, head, C/head, W, H)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.mul(q, k.transpose(-2, -1))  # sometimes they use @ operator? 
        # attn = attn * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.mul(attn, v).reshape(W, H, C)
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




    
    





