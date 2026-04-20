# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpt_blocks import FeatureFusionBlock, _make_scratch

def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[512, 512, 768, 768], 
    ):
        super(DPTHead, self).__init__()
        
        self.norm = nn.LayerNorm(in_channels)
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels if i < 3 else 256,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for i,out_channel in enumerate(out_channels)
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        

        # FIXME: layer dim
        head_features_1 = 256
        
        self.scratch.output_conv1 = nn.Sequential(
            # Depthwise Convolution (Spatial mixing)
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, groups=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, head_features_1, kernel_size=1, stride=1, padding=0),
        )
        self.scratch.output_conv2 = nn.Sequential(
            # Depthwise Convolution (Spatial mixing)
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, groups=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, head_features_1, kernel_size=1, stride=1, padding=0),
        )

        # self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
        #     # nn.ReLU(True),
        #     # nn.Conv2d(head_features_1, head_features_1, kernel_size=1, stride=1, padding=0),
        # )
    
    def forward(self, out_features, final_h, final_w):
        out = []
 
        for i, x in enumerate(out_features):
            # x = self.norm(x)
            # x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x.contiguous())
            x = self.resize_layers[i](x)
            
            out.append(x)
        

        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:]) 
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(final_h), int(final_w)), mode="bilinear", align_corners=True).contiguous()
        # print(path_1.shape)
        out = self.scratch.output_conv2(out)
        
        return out