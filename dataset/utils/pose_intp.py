import numpy as np
from .quaternion import Quaternion
import matplotlib.pyplot as plt
import matplotlib as mpl


def quaternion_slerp(start:Quaternion, stop:Quaternion, ratio:int, larger_angle = False):

    t = ratio
    cos_theta = start.q @ stop.q
    if(np.abs(cos_theta)>=0.9): # Linear approximation
        scale0 = 1 - t
        scale1 = t
    else:
        theta = np.arccos(cos_theta)
        sinTheta = np.sin(theta)

        scale0 = np.sin( ( 1 - t ) * theta) / sinTheta
        scale1 = np.sin( ( t * theta) ) / sinTheta

    if(cos_theta < 0): # Choose the smaller arc to go
        scale1 = -scale1

    if larger_angle:
        scale1 = -scale1

    qt = Quaternion(scalar = scale0 * start.scalar() + scale1 * stop.scalar(),
                        vec = scale0 * start.vec() + scale1 * stop.vec()
                        )
    return qt

def transition_lerp(begin, end, ratio):
    
    trans_t = (1 - ratio) * begin + ratio * end 
    return trans_t

def pose_intp(pose_start, pose_stop, ratio):
    rotm_start = pose_start[:3,:3]
    rotm_stop = pose_stop[:3,:3]

    tsl_start = pose_start[:3,3]
    tsl_stop = pose_stop[:3,3]

    q_start = Quaternion()
    q_start.from_rotm(rotm_start)
    q_stop = Quaternion()
    q_stop.from_rotm(rotm_stop)

    q_intp = quaternion_slerp(q_start, q_stop, ratio)
    tsl_intp = transition_lerp(tsl_start, tsl_stop, ratio)
    pose_intp = np.eye(4)
    pose_intp[:3,:3] = q_intp.to_rotm()
    pose_intp[:3,3] = tsl_intp
    return pose_intp

    
    
    



    
