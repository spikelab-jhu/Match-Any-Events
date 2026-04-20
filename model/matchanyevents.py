import torch
import torch.nn as nn
from einops.einops import rearrange

from .temp_agg import build_temporal_phrase
from .matching_module import LocalFeatureTransformer, FinePreprocess, EventSpatialTransformer
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from utils.misc import detect_NaN
from torch.utils.checkpoint import checkpoint

from loguru import logger
from .backbone.model import DinoDPT
import math
import copy

class MatchAnyEvents(nn.Module):
    def __init__(self, config, profiler=None):
        super().__init__()

        self.config = config
        self.profiler = profiler
      
        self.event_backbone = DinoDPT(amp=False, init_weight=False, config=config['dino'])
        self.loftr_coarse = LocalFeatureTransformer(config)
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.linear = nn.Sequential(
            nn.Linear(config['dino']['dim'], config['coarse']['d_model'], bias=False),
        )

        self.fine_preprocess = FinePreprocess(config)
        self.fine_matching = FineMatching(config)
        self.tam = build_temporal_phrase(config)
        self.est = EventSpatialTransformer(config['est'])

        # self.token_pruning_in_dim = config['dino']['dim']
        # prior_prob = 0.01
        # last_fc = nn.Linear(self.token_pruning_in_dim//2, 1, bias=True)
        # init_bias = math.log(prior_prob / (1 - prior_prob))
        # nn.init.constant_(last_fc.bias, init_bias)
        
        # pruning_score_mlp = nn.Sequential(
        #     nn.Linear(self.token_pruning_in_dim, self.token_pruning_in_dim//2, bias=True),
        #     nn.LeakyReLU(inplace = True),
        #     last_fc,
        #     nn.Sigmoid()
        # )
        # self.pruning_predicters = nn.ModuleList([
        #     copy.deepcopy(pruning_score_mlp) for i in range(15)
        # ])


        # self.act_manage_est0 = Manager(epsilon=0.25, score_mlp=self.pruning_predicters) # Shared pruning mlps for each stage 
        # self.act_manage_est1 = Manager(epsilon=0.25, score_mlp=self.pruning_predicters) 
        self.act_manage_est0 = None  
        self.act_manage_est1 = None

    
    def event_early_stage(self, feat_c, Ts, mask = None, token_manager = None, bias = None):

        feat_c = self.est(Ts, feat_c, mask, token_manager, bias)
        feat_c, attn = self.tam(feat_c, Ts)

        return feat_c, attn

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        event_bb_input = data['event']
        Bs, _, H_e, W_e = event_bb_input.shape

        data.update({
            'bs': data['image'].size(0),
            'hw0_i': data['image'].shape[2:], 'hw1_i': data['event'].shape[2:]
        })


        if self.config['architecture']== 'TAg':
            event_bb_input = event_bb_input.reshape(-1, 2, H_e, W_e)
            input = torch.cat([data['image'].reshape(-1, 2, H_e, W_e), event_bb_input], dim=0)
    
            ret_dict =self.event_backbone.patch_embedding(input)
            ret_dict0, ret_dict1 = torch.chunk(ret_dict, 2, dim=0)

            ret_dict0_score, ret_dict1_score = map(lambda x: rearrange(x, '(b t) h w c -> b t (h w) c', b = Bs), [ret_dict0, ret_dict1])
            bias0 = None
            bias1 = None
            if self.act_manage_est0 is not None and self.act_manage_est1 is not None:
                bias0 = self.act_manage_est0.step(ret_dict0_score, agg_t = True, stage=0)
                bias1 = self.act_manage_est1.step(ret_dict1_score, agg_t = True, stage=0)
            
            if self.act_manage_est0 is not None and self.act_manage_est1 is not None:
                data.update({
                    'pounder_loss': 0.25*(self.act_manage_est0.get_ponder_loss() + self.act_manage_est1.get_ponder_loss())
                })
                data.update({
                'certainty_loss': 0.25*(self.act_manage_est0.get_certainty_loss() + self.act_manage_est1.get_certainty_loss())
                })
                data.update({
                    'pruning_vis': self.act_manage_est1.get_mask_history()
                })
            
            ret_dict0, _ = self.event_early_stage(ret_dict0.permute(0,3,1,2), self.config['est']['seq_len'], None, self.act_manage_est0, bias0)
            ret_dict1, attn = self.event_early_stage(ret_dict1.permute(0,3,1,2), self.config['est']['seq_len'], None, self.act_manage_est1, bias1)
            if self.act_manage_est0 is not None and self.act_manage_est1 is not None:
                data['pounder_loss'] = data['pounder_loss'] + 0.25*(self.act_manage_est0.get_ponder_loss() + self.act_manage_est1.get_ponder_loss())
                data['certainty_loss'] = data['certainty_loss'] + 0.25*(self.act_manage_est0.get_certainty_loss() + self.act_manage_est1.get_certainty_loss())

            feat_c0,feat_inter0 = self.event_backbone(ret_dict0.permute(0,2,3,1))
            feat_c1,feat_inter1 = self.event_backbone(ret_dict1.permute(0,2,3,1))

        else:
            input = torch.cat([data['image'], event_bb_input], dim=0)
    
            ret_dict =self.event_backbone.patch_embedding(input)
            feat_c, feat_inter = self.event_backbone(ret_dict)
            feat_c0, feat_c1 = torch.chunk(feat_c, 2, dim=0)

            attn = torch.zeros(Bs, int(H_e * W_e/ self.config['dino']['patch_size']/ self.config['dino']['patch_size']), 1, self.config['est']['seq_len'], 4)

        
        feat_c0 = self.linear(feat_c0.permute(0,2,3,1)).permute(0,3,1,2)
        feat_c1 = self.linear(feat_c1.permute(0,2,3,1)).permute(0,3,1,2)

        mul = self.config['resolution'][0] // self.config['resolution'][1]
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': [feat_c0.shape[2] * mul, feat_c0.shape[3] * mul] ,
            'hw1_f': [feat_c1.shape[2] * mul, feat_c1.shape[3] * mul],
            'attn': attn,
            # 'prune':decision1,
            # 'contrast_supv': contrast1
        })


        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1)
        
        # detect NaN during mixed precision training
        if self.config['replace_nan'] and (torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))):
            detect_NaN(feat_c0, feat_c1)

            
        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data
                                )
        
        feat_inter0.pop()
        feat_inter1.pop()
        feat_inter0.append(feat_c0)
        feat_inter1.append(feat_c1)

        feat_f0 = self.event_backbone.dpt_fine(feat_inter0,  feat_c0.shape[2], feat_c0.shape[3])
        feat_f1 = self.event_backbone.dpt_fine(feat_inter1,  feat_c1.shape[2], feat_c1.shape[3])

        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, data)
        
        # detect NaN during mixed precision training
        if self.config['replace_nan'] and (torch.any(torch.isnan(feat_f0_unfold)) or torch.any(torch.isnan(feat_f1_unfold))):
            detect_NaN(feat_f0_unfold, feat_f1_unfold)
        
    
        # 5. match fine-level            
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        del feat_c0, feat_c1


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)