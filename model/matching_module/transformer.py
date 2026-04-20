import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .linear_attention import GeneralMaskedAttention
from einops.einops import rearrange
from collections import OrderedDict
from ..utils.position_encoding import RoPEPositionEncodingSine, RoPE1d
import numpy as np
from loguru import logger    


class RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 agg_size0=4,
                 agg_size1=4,
                 no_flash=False,
                 rope=False,
                 npe=None,
                 fp32=False,
                 event_token_prune = False
                 ):
        super(RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead
        self.agg_size0, self.agg_size1 = agg_size0, agg_size1
        self.rope = rope
        self.event_token_prune = event_token_prune

        # aggregate and position encoding
        self.aggregate = nn.Conv2d(d_model, d_model, kernel_size=agg_size0, padding=0, stride=agg_size0, bias=False, groups=d_model) if self.agg_size0 != 1 else nn.Identity()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.agg_size1, stride=self.agg_size1) if self.agg_size1 != 1 else nn.Identity()
        if self.rope:
            self.rope_pos_enc = RoPEPositionEncodingSine(d_model, max_shape=(d_model, d_model), npe=npe, ropefp16 = not fp32)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        self.attention = GeneralMaskedAttention(self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model, 2*d_model, bias=False),  # first projection
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*d_model, d_model, bias=False),  # original output layer
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, hard_mask = None, bias = None):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        bs, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # Aggragate feature
        # hard_keep_decision = None
        query, source = self.norm1(self.aggregate(x).permute(0,2,3,1)), self.norm1(self.max_pool(source).permute(0,2,3,1)) # [N, H, W, C]

        
        # if self.event_token_prune:

        #     contrast = contrast.reshape(-1,1,H0,W0)
        #     contrast = self.max_pool(contrast).permute(0,2,3,1)
        #     # distrib = self.bg_mlp(torch.cat([source, contrast],dim=-1))
        #     distrib = self.bg_mlp(source)
        #     hard_keep_decision = F.gumbel_softmax(distrib, hard=True)[:, :, :, 0:1]

        query, key, value = self.q_proj(query), self.k_proj(source), self.v_proj(source)

        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)
        query, key, value = map(lambda x: rearrange(x, 'n h w c -> n (h w) c'), [query, key, value])

        # multi-head attention handle padding mask
        m = self.attention(query, key, value, hard_mask, bias = bias)
        m = self.merge(m.reshape(bs, -1, self.nhead*self.dim)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, 'b (h w) c -> b c h w', h=H0 // self.agg_size0, w=W0 // self.agg_size0) # [N, C, H0, W0]
        if self.agg_size0 != 1:
            m = torch.nn.functional.interpolate(m, scale_factor=self.agg_size0, mode='bilinear', align_corners=False) # [N, C, H0, W0]

        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()
        
        self.full_config = config
        self.fp32 = not (config['mp'] or config['half'])
        config = config['coarse']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        self.agg_size0, self.agg_size1 = config['agg_size0'], config['agg_size1']
        self.rope = config['rope']


        self_layer = RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], config['rope'], config['npe'], self.fp32)
        cross_layer = RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], False, config['npe'], self.fp32)
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if _ == 'self' else copy.deepcopy(cross_layer) for _ in self.layer_names])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """
        H0, W0, H1, W1 = feat0.size(-2), feat0.size(-1), feat1.size(-2), feat1.size(-1)
        bs = feat0.shape[0]
    

        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):

            if name == 'self':

                feat0 = layer(feat0, feat0)
                feat1 = layer(feat1, feat1)
                
                
            elif name == 'cross':

                feat0 = layer(feat0, feat1)
                feat1 = layer(feat1, feat0)  
         
            else:
                raise KeyError

        return feat0, feat1
    

class EventSpatialTransformer(nn.Module):
    """Event spatial temporal transformer."""

    def __init__(self, config):
        super(EventSpatialTransformer, self).__init__()
        
        self.full_config = config
        self.fp32 = not (config['mp'] or config['half'])
        config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        
        self.agg_size0, self.agg_size1 = config['agg_size0'], config['agg_size1']
        self.rope = config['rope']

        self_layer = RoPE_EncoderLayer(config['d_model'], config['nhead'], config['agg_size0'], config['agg_size1'],
                                            config['no_flash'], config['rope'], config['npe'], self.fp32, event_token_prune=False)
        self_tmp_layer = Full_RoPE_EncoderLayer(config['d_model'], config['nhead'],config['seq_len'],
                                             config['rope'], config['t_npe'], self.fp32)
        
        self.layers = nn.ModuleList([copy.deepcopy(self_layer) if name =='spatial' else copy.deepcopy(self_tmp_layer) for name in self.layer_names ])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ts, feat0, mask0=None, token_manager = None, bias = None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            mask0 (torch.Tensor): [N, L] (optional)
        """

        H0, W0 = feat0.size(-2), feat0.size(-1)
        bs = feat0.shape[0]
        # bias = None
        stage = 0
        
        
        for i, (layer, name) in enumerate(zip(self.layers, self.layer_names)):
            
            
            if name == 'spatial':
                stage = stage + 1
                # if token_manager is not None:
                #     bias = token_manager.construct_attention_bias(prop_t = True)
                mask_history = None
                if token_manager is not None and bs == 8:
                 
                    mask_history = token_manager.get_mask_history()
                
                feat0 = layer(feat0, feat0, mask_history, bias = bias)

                if token_manager is not None and stage <= 1:
                    
                    bias = token_manager.step(rearrange(feat0, '(b t) c h w -> b t (h w) c', t = ts), agg_t = True, stage = stage)
                
            elif name == 'temporal':
                feat0 = layer(ts, feat0, feat0)
            else:
                print(name)
                raise KeyError
        
        return feat0

class Full_RoPE_EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 seq_len,
                 rope=False,
                 npe=None,
                 fp32 = False
                 ):
        super(Full_RoPE_EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        # self.dim = self.d_q // nhead
        self.nhead = nhead
        self.rope = rope
        self.fp32 = fp32
        if self.rope:
            self.rope_pos_enc = RoPE1d(d_model, seq_len=seq_len, npe=npe, ropefp16=not fp32)
        
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)        
        # self.attention = ESTFullAttention(False)
        self.attention = GeneralMaskedAttention(self.nhead, self.dim, fp32)
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(2*d_model, 2*d_model, bias=False),  # first projection
            nn.LeakyReLU(inplace=True),
            nn.Linear(2*d_model, d_model, bias=False),  # original output layer
        )

        # norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, ts, x, source):
        """
        Args:
            x (torch.Tensor): [N, C, H0, W0]
            source (torch.Tensor): [N, C, H1, W1]
            x_mask (torch.Tensor): [N, H0, W0] (optional) (L = H0*W0)
            source_mask (torch.Tensor): [N, H1, W1] (optional) (S = H1*W1)
        """
        _, C, H0, W0 = x.size()
        H1, W1 = source.size(-2), source.size(-1)

        # x = x.reshape(-1, ts, C, H0, W0)
        source = source.reshape(-1, ts, C, H1, W1)
        bs = source.shape[0]

        # Aggragate feature
        query, source = self.norm1(x.reshape(-1, ts, C, H0, W0).permute(0,3,4,1,2)), self.norm1(source.permute(0,3,4,1,2)) # [N, H, W, ts, C]
        query, key, value = self.q_proj(query).reshape(-1, ts, C), self.k_proj(source).reshape(-1, ts, C), self.v_proj(source).reshape(-1, ts, C)
        # query, key, value = map(lambda x: x, [query, key, value])
        
        # Positional encoding        
        if self.rope:
            query = self.rope_pos_enc(query)
            key = self.rope_pos_enc(key)

        # query, key = map(lambda x: rearrange(x, 'n l (nhead d) -> n l nhead d', nhead=self.nhead, d=self.dim), [query, key])
        # value = rearrange(value, 'n l (nhead d) -> n l nhead d', nhead=self.nhead, d=self.dim)
        # multi-head attention 
        m = self.attention(query, key, value)
        m = self.merge(m.reshape(-1, ts, C)) # [N, L, C]

        # Upsample feature
        m = rearrange(m, '(b h w) ts c -> (b ts) c h w',b=bs, h=H0 , w=W0) # [N, C, H0, W0]
        # feed-forward network
        m = self.mlp(torch.cat([x, m], dim=1).permute(0, 2, 3, 1)) # [N, H0, W0, C]
        m = self.norm2(m).permute(0, 3, 1, 2) # [N, C, H0, W0]

        return x + m
