import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat

from loguru import logger


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.W = self.config['fine_window_size']

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]
        # 2. unfold(crop) all local windows
        feat_f0 = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=0)
        feat_f0 = rearrange(feat_f0, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f1 = F.unfold(feat_f1, kernel_size=(W+2, W+2), stride=stride, padding=1)
        feat_f1 = rearrange(feat_f1, 'n (c ww) l -> n l ww c', ww=(W+2)**2)

        # 3. select only the predicted matches
        feat_f0 = feat_f0[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1 = feat_f1[data['b_ids'], data['j_ids']]

        return feat_f0, feat_f1