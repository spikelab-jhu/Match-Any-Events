import numpy as np
import torch


def depth_transform(depth_map, K_src, K_dst, T):
    h,w = depth_map.shape
    depth_map = torch.from_numpy(depth_map)
    K_src = torch.from_numpy(K_src)
    K_dst = torch.from_numpy(K_dst)
    T  = torch.from_numpy(T)

    valid_inf = ~torch.isinf(depth_map.flatten())

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    xy_homogeneous = torch.stack((x.flatten(), y.flatten(), torch.ones_like(x.flatten())), dim=0).to(torch.float64)
    K_src_inv = torch.inverse(K_src)
    points_3D_src = K_src_inv @ xy_homogeneous[:,valid_inf] * depth_map.flatten()[valid_inf]
    points_3D_src_h = torch.cat((points_3D_src, torch.ones(1, points_3D_src.shape[1])), dim=0)

    points_3D_tgt = (torch.inverse(T) @ points_3D_src_h)[:-1]
    pixel_coords_tgt = K_dst @ points_3D_tgt  # (3, H*W)
    pixel_coords_tgt[:2] /= pixel_coords_tgt[2]
    u_tgt = pixel_coords_tgt[0].long()
    v_tgt = pixel_coords_tgt[1].long()
    depth_tgt = points_3D_tgt[2]

    depth_map_tgt = torch.ones_like(depth_map,dtype=torch.float32)*torch.inf
    valid = (u_tgt >= 0) & (u_tgt < w) & (v_tgt >= 0) & (v_tgt < h)
    # depth_map_tgt[v_tgt[valid], u_tgt[valid]] > depth_tgt[valid]
    depth_map_tgt[v_tgt[valid], u_tgt[valid]] = depth_tgt[valid].to(torch.float32)
    
    return depth_map_tgt.numpy()