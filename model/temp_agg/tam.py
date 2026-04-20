# Temporal information aggregation module
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import repeat
import collections.abc

from timm.models.layers import DropPath, trunc_normal_

import math

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
attention_map = None

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
    
class TAgAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_dropout=0.1, proj_dropout=0.0, learned_bias = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        # multi-head attention
        self.q_proj = nn.Linear(dim, dim, bias=True) # Learned query bias for represent quality
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, source, x_mask = None, source_mask = None):
        """
        Args:
            x (torch.Tensor): [B, N, L, D]
            source (torch.Tensor): [B, N, S, D]
            x_mask (torch.Tensor): [(B, N), L] (optional)
            source_mask (torch.Tensor): [(B, N), S] (optional)
        """
        query, key, value = x, source, source
        B, N, L, C = x.shape
        S = source.shape[2]

        query = self.q_proj(query).view(-1, L, self.num_heads, self.head_dim ) #+ self.temporal_query_bias # [-1, L, (H, D)]
        key = self.k_proj(key).view(-1, S, self.num_heads, self.head_dim ) # [-1, L, (H, D)]
        value = self.v_proj(value).view(-1, S, self.num_heads, self.head_dim) # [-1, L, (H, D)]

        # # Original skip connection from channel1
        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        if source_mask is not None:
            QK.masked_fill_(~(x_mask[:, :, None, None] * source_mask[:, None, :, None]), float('-inf'))

        attn = QK * (1.0 / (self.head_dim ** 0.5))

        attn = torch.softmax(attn, dim=2)
        attn = self.attn_dropout(attn)
        queried_values = torch.einsum("nlsh,nshd->nlhd", attn, value).reshape(B, N, L, C)

        return self.proj_dropout(self.proj(queried_values)), attn
    
 
class TAg_Singlelayer(nn.Module):
    def __init__(self, embed_dim = 768, num_heads = 8, attn_dropout = 0., proj_dropout = 0., learned_bias = True) :
        super().__init__()
        # self.patch_embed = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size, stride=stride, in_chans=in_chans, embed_dim=embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)


        self.attn = TAgAttention(embed_dim, num_heads, attn_dropout, proj_dropout, learned_bias=learned_bias)
        self.mlp = nn.Sequential(
            nn.Linear(2*embed_dim, 2*embed_dim, bias=False),  # first projection
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*embed_dim, embed_dim, bias=False),  # original output layer
        )
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x, source, H, W, mask = None):
        """
        Args: 
            x [B, N, 1, C]
            source [B, N, T, C]
        """
        B, N, T, C = source.shape
        source = self.norm1(source)
        message, attn_map = self.attn(x, source, mask, mask)
        message = self.norm2(message)
        message = self.mlp(torch.cat([x, message], dim=-1))#.squeeze(2), H, W).unsqueeze(2)
        message = self.norm3(message)

        return x + message, attn_map.detach()


class TAg(nn.Module):
    def __init__(self, config):
        super(TAg, self).__init__()
        self.config = config
        self.d_model = config['embed_dim']
        self.nhead = config['num_heads']
        self.attn_dropout = config['attn_dropout']
        self.proj_dropout = config['dropout']
        self.num_layers = config['layers']
        encoder_layer = TAg_Singlelayer(self.d_model,self.nhead,self.attn_dropout,self.proj_dropout, learned_bias=config['learned_bias'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.num_layers)])
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
    def forward(self, feat, T, mask=None):
        """
        Args:
            feat [-1, C, H, W]
        """
        _, C, H, W = feat.shape
        feat = feat.reshape(-1, T, C, H, W)# [B, T, C, H, W]
        feat = feat.flatten(3).permute(0, 3, 1, 2) # [B, N, T, C]

        assert self.d_model == feat.size(3), "the feature number of src and transformer must be equal"
        query = feat[:,:,:1,:]# [B, N, 1, C]
        attn_rtn = None
        for layer in self.layers:
            query, attn_map = layer(query, feat, H, W, mask)
            if attn_rtn is None:
                attn_rtn = attn_map
        query = query.permute(0, 2, 3, 1)

        return query.reshape(-1, C, H, W), attn_rtn.detach().reshape(-1, H*W, T, self.nhead)