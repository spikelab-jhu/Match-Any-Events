

import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import gc
import math
from collections import OrderedDict

from typing import Callable, Optional, Tuple, Union

from torch import Tensor

from model.backbone.dpt_head import DPTHead


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)

class MyPatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
    
def init_vit_weights_stable(model: nn.Module, n_layers: int = 12):
    """
    Applies stable initialization to a ViT model.
    Call this ONLY ONCE, before training starts.
    """
    
    # 1. Standard Init (Use apply to visit every layer recursively)
    def _init_basic_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(_init_basic_weights)

    # 2. LayerScale / Depth-Scaling (Iterate top-level named parameters ONCE)
    #    This prevents "Double Initialization" recursion bugs.
    scale_factor = 1.0 / math.sqrt(2.0 * n_layers)
    
    for name, p in model.named_parameters():
        if 'proj.weight' in name or 'fc2.weight' in name:
            # We overwrite the previous init for these specific layers
            nn.init.trunc_normal_(p, std=0.02 * scale_factor)

    print(f"Weights initialized with depth-scaling factor: {scale_factor:.4f}")

class DinoDPT(nn.Module):
    def __init__(self, amp = False, init_weight = True, amp_dtype = torch.float16, config = None):
        super().__init__()

        image_size = config['img_size']
        in_channel = config['in_chan']
        self.patch_size = config['patch_size']
        self.model_type = config['model']
        self.embed_dim = config['dim']

        from .dinov3.vision_transformer import vit_large, vit_small, vit_base
        vit_kwargs = dict(img_size= image_size,
                patch_size= self.patch_size,
                init_values = 1.0,
                layerscale_init = 1.0,
                ffn_layer = "mlp",
                block_chunks = 0,
            )
        if self.model_type == 's':
            vits14 = vit_small(**vit_kwargs)#.eval()
            dpt_channels = [128, 256, 512, 512]
        elif self.model_type == 'b':
            vits14 = vit_base(**vit_kwargs)
            dpt_channels = [256, 512, 1024, 1024]
        else:
            raise NotImplementedError('Model type not implemented')
            

        vits14.patch_embed = MyPatchEmbed(img_size=image_size, patch_size=self.patch_size, in_chans=in_channel, embed_dim=self.embed_dim, flatten_embedding=False)
        
        if init_weight:
            # init_weight = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            # ckpt = torch.load("./pretrained/dinov2_vits14_pretrain.pth", map_location="cpu")
            # init_weight = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth")
            
            if self.model_type == 'b':
                ckpt_path = "./pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            elif self.model_type == 's':
                ckpt_path = "./pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"  # your .pth file
            dino_v3_weight = torch.load(ckpt_path, map_location="cpu")

            filtered_weights = OrderedDict()
            for k, v in dino_v3_weight.items():
                if k.startswith("patch_embed."):
                    print(f"Skipping {k} (shape {v.shape})")
                    continue
                filtered_weights[k] = v
       
            vits14.load_state_dict(filtered_weights, strict=False)

        for param in vits14.parameters():
            param.requires_grad = True

        for param in vits14.patch_embed.parameters():
            param.requires_grad = True

        self.amp = amp
        self.amp_dtype = amp_dtype
        if self.amp:
            vits14 = vits14.to(self.amp_dtype)
        self.vits14 = vits14 # ugly hack to not show parameters to DDP


        self.dpt_head = DPTHead(in_channels = self.embed_dim, features = 256, out_channels=dpt_channels)
        if self.amp:
            self.dpt_head = self.dpt_head.to(self.amp_dtype)
        
    
    def patch_embedding(self, x):
        if self.vits14.device != x.device:
            self.vits14 = self.vits14.to(x.device)
        return self.vits14.embedding(x)
    
    def train(self, mode: bool = True):
        return self.vits14.train(mode)
    
    def forward(self, x):
        B,H,W,C = x.shape
        
        if self.vits14.device != x.device:
            self.vits14 = self.vits14.to(x.device)
        dino_features_14 = self.vits14.forward_features(x)
        features_14 = dino_features_14['x_norm_patchtokens'].permute(0,2,1).reshape(B,self.embed_dim,H, W)

        intermediate = dino_features_14['intermediate']

        return features_14, intermediate#, fine_feature
    
    def dpt_fine(self, intermediate, H, W):
        
        fine_feature = self.dpt_head(intermediate, H * self.patch_size, W * self.patch_size)
        return fine_feature
