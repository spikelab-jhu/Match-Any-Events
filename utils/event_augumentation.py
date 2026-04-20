import torch
import numpy as np
aug_type = ['hori_flip', 'time_inv']

def temporal_augmentation(event, polared = True, valid_start = [0,1,2], T = 8, train = True):
    T_max = 8
    '''Temporal augmentation for a singel event data'''
    c, h, w = event.shape
    if polared:
        event = event.reshape(-1, 2, h, w)
    
    if not train:
        
        return event[valid_start[1]+T_max-T:valid_start[1]+T_max].reshape(-1, h, w)
   
    idx = torch.randint(0, len(valid_start), (1,))
    event = event[valid_start[idx]+T_max-T:valid_start[idx] + T_max]
    
    return event.reshape(-1, h, w)

import torch.nn.functional as F

def build_contrast(event, patch_size):
    B,C,H,W = event.shape
    event = event.reshape(-1,2,H,W)
    event = torch.mean(event,dim=1)
    if event.ndim == 3:  # (B,H,W)
        event = event[:, None, :, :]
    _, _, H, W = event.shape
    mean = F.avg_pool2d(event, kernel_size=patch_size, stride=patch_size)
    mean_sq = F.avg_pool2d(event**2, kernel_size=patch_size, stride=patch_size)
    var = mean_sq - mean**2   # (B,1,H//p,W//p)

    return var.reshape(B, -1, *var.shape[2:])