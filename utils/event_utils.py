import torch
import numpy as np
import torch.nn as nn
import os
from dataset.utils.pose_intp import pose_intp

def get_random_interval(val, tol):
    rand_int_tol = lambda val, tol: np.random.randint(int(val*(1-tol/100)), int(val*(1+tol/100))+1)
    #return rand_int_tol(val, tol)
    return val

def get_pose_t(pose, pose_ts, ts):
    current_pose_idx = np.searchsorted(pose_ts, ts)
    if current_pose_idx==0 or current_pose_idx>=len(pose_ts)-1:
        return None
    intp_ratio = (ts - pose_ts[current_pose_idx-1]) / (pose_ts[current_pose_idx] - pose_ts[current_pose_idx-1]+1e-6)
    pose_current = np.linalg.inv(pose[current_pose_idx-1])
    pose_next = np.linalg.inv(pose[current_pose_idx])
    return pose_intp(pose_current, pose_next, intp_ratio)
    
def ensure_dir(dir):
    os.makedirs(dir, exist_ok=True)

def get_event_polarity(event:dict, polarity = 1):
    if polarity>0:
        index = event['p'] > 0
    else:
        index = event['p'] < 1

    if len(index) == 0:
        return None
    event_updated = {'x':event['x'][index],
                     'y':event['y'][index],
                     't':event['t'][index],
                     'p':event['p'][index]
                     }
    return event_updated

def stack_pos_neg(pos,neg):
    stacked = torch.stack([pos, neg], dim=1).reshape(-1, *pos.shape[1:])
    return stacked

def calc_floor_ceil_delta(x): 
    x_fl = torch.floor(x + 1e-6)
    x_ce = torch.ceil(x - 1e-6)
    x_ce_fake = torch.floor(x) + 1

    dx_ce = x - x_fl
    dx_fl = x_ce_fake - x

    return [x_fl.long(), dx_fl], [x_ce.long(), dx_ce]

def create_update_no_polarity(x, y, t, dt, p, vol_size):
    inds = (vol_size[1]*vol_size[2]) * t\
         + (vol_size[2])*y\
         + x

    vals = dt

    return inds, vals
 
def gen_discretized_event_volume_no_polarity(events, vol_size):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    #t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min))
    t_scaled = (t-t_min) * ((vol_size[0]-1) / (t_max-t_min + 1e-6))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())

    inds_fl, vals_fl = create_update_no_polarity(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size)
        
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update_no_polarity(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size)
    
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)

    return volume

def normalize_event_volume(event_volume):
    event_volume_flat = event_volume.view(-1)
    nonzero = torch.nonzero(event_volume_flat)
    nonzero_values = event_volume_flat[nonzero]
    if nonzero_values.shape[0]:
        lower = torch.kthvalue(nonzero_values,
                               max(int(0.02 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        upper = torch.kthvalue(nonzero_values,
                               max(int(0.98 * nonzero_values.shape[0]), 1),
                               dim=0)[0][0]
        max_val = max(abs(lower), upper)
        event_volume = torch.clamp(event_volume, -max_val, max_val)
        event_volume /= (max_val + 1e-5)
    return event_volume


def create_update(x, y, t, dt, p, vol_size, device="cpu"):
    assert (x>=0).byte().all() 
    assert (x<vol_size[2]).byte().all()
    assert (y>=0).byte().all()
    assert (y<vol_size[1]).byte().all()
    assert (t>=0).byte().all() 

    if not (t < vol_size[0] // 2).byte().all():
        print(t[t >= vol_size[0] // 2])
        print(vol_size)
        raise AssertionError()

    # vol_mul = torch.where(p < 0,
    #                       torch.div(torch.ones(p.shape, dtype=torch.long).to(device) \
    #                                     * vol_size[0],
    #                                 2, rounding_mode='floor'),
    #                       torch.zeros(p.shape, dtype=torch.long).to(device))

    # inds = (vol_size[1]*vol_size[2]) * (t + vol_mul)\
    #      + (vol_size[2])*y\
    #      + x

    # vals = dt

    '''
    My modification
    '''
    inds = (vol_size[1]*vol_size[2]) * t\
         + (vol_size[2])*y\
         + x

    vals = p * dt

    return inds, vals

def gen_discretized_event_volume(events, vol_size, device="cpu"):
    # volume is [t, x, y]
    # events are Nx4
    npts = events.shape[0]
    volume = events.new_zeros(vol_size).to(device)

    x = events[:, 0].long()
    y = events[:, 1].long()
    t = events[:, 2]

    t_min = t.min()
    t_max = t.max()
    t_scaled = (t-t_min) * ((vol_size[0] // 2-1) / (t_max-t_min + 1e-6))

    ts_fl, ts_ce = calc_floor_ceil_delta(t_scaled.squeeze())
    
    inds_fl, vals_fl = create_update(x, y,
                                     ts_fl[0], ts_fl[1],
                                     events[:, 3],
                                     vol_size,
                                     device=device)
    volume.view(-1).put_(inds_fl, vals_fl, accumulate=True)

    inds_ce, vals_ce = create_update(x, y,
                                     ts_ce[0], ts_ce[1],
                                     events[:, 3],
                                     vol_size,
                                     device=device)
    volume.view(-1).put_(inds_ce, vals_ce, accumulate=True)
    return volume