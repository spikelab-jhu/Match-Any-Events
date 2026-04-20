import torch
import h5py
import numpy as np
import yaml
import os
import sys
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R
sys.path.append('./')

import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from representations import EventFrame, Adaptive_interval, TsGenerator, EventVis
from utils.event_utils import get_event_polarity, stack_pos_neg, get_pose_t
from dataset.data_loader import event_norm
import torch.nn.functional as F
from os.path import join as opj



def ts2image(ts):
    #ts = np.average(ts, axis=2)  # Take average over channels
    #ts = np.uint8(255. * ts)
    #ts = cv2.cvtColor(ts, cv2.COLOR_GRAY2BGR)
    ts_out = np.ones(list(ts.shape[:2]) + [3])
    #blue_values = ts[ts[..., 2] > 0][..., 2].T
    #red_values = ts[ts[..., 7] > 0][..., 7].T
    #ts_out[ts[..., 2] > 0] = np.array([np.ones_like(blue_values), 1. - blue_values, 1. - blue_values]).T
    #ts_out[ts[..., 7] > 0] = np.array([1. - red_values, 1. - red_values, np.ones_like(red_values)]).T
    blue_values = ts[ts[..., 3] > 0][..., 3].T
    red_values = ts[ts[..., 8] > 0][..., 8].T
    ts_out[ts[..., 3] > 0] = np.array([np.ones_like(blue_values), 1. - blue_values, 1. - blue_values]).T
    ts_out[ts[..., 8] > 0] = np.array([1. - red_values, 1. - red_values, np.ones_like(red_values)]).T
    ts_out = np.rint(ts_out * 255.).astype(np.uint8)
    return ts_out


def read_calib(calib_file):
    with open(calib_file, "r") as f:
        calib = yaml.safe_load(f)

        # Access cam0
        cam0 = calib["cam0"]
        fx, fy, cx, cy = np.array(cam0["intrinsics"])  # fx, fy, cx, cy
        K0 = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        D0 = np.array(cam0["distortion_coeffs"])
        res0 = np.array(cam0["resolution"])

        # Access cam1
        cam1 = calib["cam1"]
        fx, fy, cx, cy = np.array(cam1["intrinsics"])  # fx, fy, cx, cy
        K1 = np.array([[fx, 0,  cx],
                    [0,  fy, cy],
                    [0,  0,  1]], dtype=np.float32)
        D1 = np.array(cam1["distortion_coeffs"])
        T_cn_cnm1 = np.array(cam1["T_cn_cnm1"])  # 4x4 extrinsic
        res1 = np.array(cam1["resolution"])

        return {'event':{
                    'K':K1,
                    'D':D1,
                    'res':res1
                         },
                'image':{
                    'K':K0,
                    'D':D0,
                    'res':res0
                },
                'i_T_e':np.linalg.inv(T_cn_cnm1)
                }

class EDMDataset(Dataset):
    def __init__(self,root_dir, homo = True, repre = '',
                 config = None):
        self.root_dir = root_dir
        self.homo = homo
        self.src_res = config['resolution']
        self.dst_res = config['val_res']
        # self.ransac_thr = config['ransac_thr']
        self.interval = config['interval_ms'] * 1e3
        self.num_bin = config['num_bin']
        pair_infos = np.loadtxt(opj(self.root_dir, 'sampled_pairs.txt'), dtype=str)
        overlap_score = np.loadtxt(opj(self.root_dir, 'overlapping_score.txt'), dtype=float)

        self.pair_info = [info for _, (info, score) in enumerate(zip(pair_infos, overlap_score)) if (score > config['min_overlap_score']) ]
        self.pose_dir = opj(self.root_dir, 'poses')
        self.calib_dir = opj(self.root_dir, 'calib.yaml')

        assert repre in config['repres']
        self.repre = repre

        self.calibration = read_calib(self.calib_dir)

    def __len__(self):
        return len(self.pair_info)
    
    def read_im(self, dir):
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # Convert to float32 and normalize to 0-1
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img[None,...]).to(torch.float32)
    
    def undistort_resize(self, data, K, D):
        h,w = data.shape[-2:]
        coeff = np.array(self.dst_res)/np.array([h,w])
        K_new = K.copy()
        K_new[0] *= coeff[1]
        K_new[1] *= coeff[0]

        data_np = np.transpose(data.numpy(),(1,2,0))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R=np.eye(3), newCameraMatrix=K,
                                         size=(w, h), m1type=cv2.CV_32FC1)
        undistorted = cv2.remap(data_np, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

        if len(undistorted.shape) == 2:
            undistorted = undistorted[...,None]
        undistorted = np.transpose(undistorted,(2,0,1))

        # Cropping to fit the target ratio
        # h_new = h
        # w_new = w
        # if 1.0*h/w >= 1.0*self.dst_res[0]/self.dst_res[1]:
        #     h_new = int(w * self.dst_res[0]/self.dst_res[1])
        # else:
        #     w_new = int(h * self.dst_res[1]/self.dst_res[0])
        # undistorted = undistorted[:,:h_new-1,:w_new-1]

        data_torch = torch.from_numpy(undistorted)
        data_torch = F.interpolate(data_torch.unsqueeze(0).float(),
                                    size=self.dst_res,
                                    mode='nearest'
                                    ).squeeze(0)

        return data_torch, K_new
    
    def read_pose(self, dir):
        pose = np.loadtxt(dir)
        return pose
    
    def load_h5(self, dir):
        with h5py.File(dir, 'r') as f:
            t = f['t'][:]
            t_stop = t[0]+ self.interval
            t_stop_for_visualize = t[0]+ 30 * 1e3

            idx_stop, idx_stop_vis = np.searchsorted(t, np.array([t_stop, t_stop_for_visualize]))
            if idx_stop>=len(t):
                print("Interval too long, setting to -1")
                idx_stop = -1
            event = {
                'x':torch.from_numpy(f['x'][:idx_stop].copy().astype(float)).to(torch.float32),
                'y':torch.from_numpy(f['y'][:idx_stop].copy().astype(float)).to(torch.float32),
                't':torch.from_numpy(f['t'][:idx_stop].copy().astype(float)).to(torch.float32),
                'p':torch.from_numpy(f['p'][:idx_stop].copy().astype(float)).to(torch.float32) * 2 - 1,
            }
            event_vis = {
                'x':torch.from_numpy(f['x'][:idx_stop_vis].copy().astype(float)).to(torch.float32),
                'y':torch.from_numpy(f['y'][:idx_stop_vis].copy().astype(float)).to(torch.float32),
                't':torch.from_numpy(f['t'][:idx_stop_vis].copy().astype(float)).to(torch.float32),
                'p':torch.from_numpy(f['p'][:idx_stop_vis].copy().astype(float)).to(torch.float32) * 2 - 1,
            }

        import time 
        repre_start = time.perf_counter()
        vis = None
        
        converter = EventVis((3, self.src_res[0], self.src_res[1]))
        vis = converter.convert(event_vis)
        if self.repre == 'event_frame':
            converter = EventFrame((self.src_res[0],self.src_res[1]))
            event_data = converter.convert(event)
        elif self.repre == 'mcts':
            converter = TsGenerator(settings = {"shape": self.src_res, "delta_t": [0.001, 0.003, 0.01, 0.03, 0.1]})
            event_data = converter.convert(event)
            event_data = event_data.permute(2,0,1)
        elif self.repre == 'event_stack':
            event_voxel = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
            data_event_pos = event_voxel.convert(get_event_polarity(event, polarity=1))
            event_voxel_neg = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
            data_event_neg = event_voxel_neg.convert(get_event_polarity(event, polarity=-1))
            event_data = stack_pos_neg(data_event_pos, data_event_neg)
            event_data = event_norm(event_data)
        else:
            pass

        repre_stop = time.perf_counter()

        print(f'build one rep {(repre_stop - repre_start)*1000:.1f} ms')
        if len(event_data.shape) == 2:
            event_data = event_data[None,...]

        return event_data, vis
    
    def __getitem__(self, index):
        name0, name1 = self.pair_info[index]
        parent0 = name0.split('_')[0]
        parent1 = name1.split('_')[0]
        data0 = None
        data1 = None
        vis0 = None
        vis1 = None
        if self.repre =='mcts':
            event_foler = 'event_per_frame_super'
        else:
            event_foler = 'event_per_frame'

        if self.repre == 'reconstruction':
            if self.homo:
                data0 = self.read_im(opj(self.root_dir,parent0, 'event', 'reconstruction', 'renamed', name0.split('.')[0]+'.png'))
                data1 = self.read_im(opj(self.root_dir,parent1, 'event', 'reconstruction', 'renamed', name1.split('.')[0]+'.png'))
                pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
                pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
                K0 = self.calibration['event']['K']
                K1 = self.calibration['event']['K']
                D0 = self.calibration['event']['D']
                D1 = self.calibration['event']['D']
            else:
                data0 = self.read_im(opj(self.root_dir, parent0, 'gt_hetero', name0))
                data1 = self.read_im(opj(self.root_dir,parent1, 'event', 'reconstruction', 'renamed', name1.split('.')[0]+'.png'))
                pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt'))
                pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
                K0 = self.calibration['image']['K']
                K1 = self.calibration['event']['K']
                D0 = self.calibration['image']['D']
                D1 = self.calibration['event']['D']
        
        elif self.repre == 'image':
            data0 = self.read_im(opj(self.root_dir, parent0, 'gt_hetero', name0))
            data1 = self.read_im(opj(self.root_dir, parent1, 'gt_hetero', name1))
            pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt'))
            pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt'))
            K0 = self.calibration['image']['K']
            K1 = self.calibration['image']['K']
            D0 = self.calibration['image']['D']
            D1 = self.calibration['image']['D']
        elif self.homo:
            data0, vis0 = self.load_h5(opj(self.root_dir, parent0, event_foler, name0.split('.')[0]+'.h5'))
            data1, vis1 = self.load_h5(opj(self.root_dir, parent1, event_foler, name1.split('.')[0]+'.h5'))
            pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
            pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
            K0 = self.calibration['event']['K']
            K1 = self.calibration['event']['K']
            D0 = self.calibration['event']['D']
            D1 = self.calibration['event']['D']
        else:
            data0 = self.read_im(opj(self.root_dir, parent0, 'gt_hetero', name0))
            data1, vis1 = self.load_h5(opj(self.root_dir, parent1, event_foler, name1.split('.')[0]+'.h5'))
            pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt'))
            pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
            K0 = self.calibration['image']['K']
            K1 = self.calibration['event']['K']
            D0 = self.calibration['image']['D']
            D1 = self.calibration['event']['D']
        data0_resized, K0_new = self.undistort_resize(data0, K0, D0)
        data1_resized, K1_new = self.undistort_resize(data1, K1, D1)


        if vis0 is not None:
            vis0, _ = self.undistort_resize(vis0, K0, D0)
            vis0 = vis0.squeeze(0)
        else:
            vis0 = data0_resized
        if vis1 is not None:
            vis1, _ = self.undistort_resize(vis1, K0, D0)
            vis1 = vis1.squeeze(0)
        else:
            vis1 = data1_resized

       

        return {
            'image':data0_resized,
            'pose0':torch.from_numpy(pose0),
            'K0':torch.from_numpy(K0_new),
            'event':data1_resized,
            'pose1':torch.from_numpy(pose1),
            'K1':torch.from_numpy(K1_new),
            'T_0to1':torch.from_numpy(np.linalg.inv(pose1) @ pose0).to(torch.float32),
            'D':D0,
            'D1':D1,
            'vis0': vis0,
            'vis1': vis1
        }
    
class M3EDDataset(Dataset):
    def __init__(self,root_dir, homo = True, repre = '',
                 config = None):
        self.root_dir = root_dir
        self.file_name = root_dir.split('/')[-1]
        self.homo = homo
        self.src_res = config['resolution']
        self.dst_res = config['val_res']
        # self.ransac_thr = config['ransac_thr']
        self.interval = config['interval_ms']
        self.num_bin = config['num_bin']
        self.pair_info = np.loadtxt(opj(self.root_dir, 'pair_info.txt'), dtype=int)
        self.data_h5 = opj(self.root_dir, self.file_name+'_data.h5')
        self.gt_h5 = opj(self.root_dir, self.file_name+'_depth_gt.h5')

        assert repre in config['repres']
        self.repre = repre

        self.rec_dir = opj(self.root_dir, 'reconstruction')
    def __len__(self):
        return len(self.pair_info)

    
    def undistort_resize(self, data, K, D):

        h,w = data.shape[-2:]
        coeff = np.array(self.dst_res)/np.array([h,w])
        K_new = K.copy()
        K_new[0] *= coeff[1]
        K_new[1] *= coeff[0]

        data_np = np.transpose(data.numpy(),(1,2,0))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R=np.eye(3), newCameraMatrix=K,
                                         size=(w, h), m1type=cv2.CV_32FC1)
        undistorted = cv2.remap(data_np, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

        if len(undistorted.shape) == 2:
            undistorted = undistorted[...,None]
        undistorted = np.transpose(undistorted,(2,0,1))

        # Cropping to fit the target ratio
        # h_new = h
        # w_new = w
        # if 1.0*h/w >= 1.0*self.dst_res[0]/self.dst_res[1]:
        #     h_new = int(w * self.dst_res[0]/self.dst_res[1])
        # else:
        #     w_new = int(h * self.dst_res[1]/self.dst_res[0])
        # undistorted = undistorted[:,:h_new-1,:w_new-1]

        data_torch = torch.from_numpy(undistorted)
        data_torch = F.interpolate(data_torch.unsqueeze(0).float(),
                                    size=self.dst_res,
                                    mode='nearest'
                                    ).squeeze(0)

        return data_torch, K_new
    
    def read_pose(self, ts_i, ts_e):
        with h5py.File(self.gt_h5) as f:
            depth_ts = f['/ts'][:].copy()
            poses = f['Cn_T_C0'][:].copy()
        pose_image = get_pose_t(poses, depth_ts, ts_i) # C0_T_in
        pose_event = get_pose_t(poses, depth_ts, ts_e) # C0_T_en

        return pose_image, pose_event
    
    def read_calib(self, calib):

        dist = np.array(calib['distortion_coeffs'])
        intri = calib['intrinsics'][:].copy()
        K0_mat = np.array([
            [intri[0], 0, intri[2]],
            [0, intri[1], intri[3]],
            [0, 0, 1]
        ])
        T = calib['T_to_prophesee_left'][:].copy()
        # mapx_d, mapy_d = cv2.initUndistortRectifyMap(K0_mat, dist, np.eye(3), K0_mat, (self.cfg['shape'][1], self.cfg['shape'][0]), cv2.CV_32FC1)
        return K0_mat, dist, T
    # 000000000016
    # 000000005808
    def read_im(self, dir):
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # Convert to float32 and normalize to 0-1
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img[None,...]).to(torch.float32)
    
    
    def load_h5(self, idx_i, idx_e):
        def construct_repre(event, event_vis, ts):
            import time
            repre_start = time.perf_counter()
            vis = None
            if self.homo:
                converter = EventVis((3, self.src_res[0], self.src_res[1]))
                vis = converter.convert(event_vis)
            if self.repre == 'reconstruction':

                event_data = self.read_im(opj(self.rec_dir,'%016d000.png' % ts))
            elif self.repre == 'event_frame':
                converter = EventFrame((self.src_res[0],self.src_res[1]))
                event_data = converter.convert(event)
            elif self.repre == 'mcts':
                converter = TsGenerator(settings = {"shape": self.src_res, "delta_t": [0.001, 0.003, 0.01, 0.03, 0.1]})
                event_data = converter.convert(event)
                event_data = event_data.permute(2,0,1)
            elif self.repre == 'event_stack':
                event_voxel = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
                data_event_pos = event_voxel.convert(get_event_polarity(event, polarity=1))
                event_voxel_neg = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
                data_event_neg = event_voxel_neg.convert(get_event_polarity(event, polarity=-1))
                event_data = stack_pos_neg(data_event_pos, data_event_neg)
                event_data = event_norm(event_data)
            else:
                pass

            repre_stop = time.perf_counter()

            print(f'build one rep {(repre_stop - repre_start)*1000:.1f} ms')
            if len(event_data.shape) == 2:
                event_data = event_data[None,...]
            return event_data, vis
        
        with h5py.File(self.data_h5) as f:
                image_ts = f['/ovc/ts'][:].copy()
                ts_i = image_ts[idx_i]
                ts_e = image_ts[idx_e]
                ms2left = f['/prophesee/left/ms_map_idx'][:].copy()
                if self.repre == 'mcts':
                    e_ref_start, e_ref_stop = int(ms2left[int(ts_i/1e3 - self.interval)]),int(ms2left[int(ts_i/1e3 ) % len(ms2left)])
                    e_idx_start, e_idx_stop = int(ms2left[int(ts_e/1e3 - self.interval)]),\
                        int(ms2left[int(ts_e/1e3) % len(ms2left)])
                else:

                    e_ref_start, e_ref_stop = int(ms2left[int(ts_i/1e3 )]),int(ms2left[int(ts_i/1e3 + self.interval) % len(ms2left)])
                    e_idx_start, e_idx_stop = int(ms2left[int(ts_e/1e3)]),\
                        int(ms2left[int(ts_e/1e3 + self.interval) % len(ms2left)])
                
                vis_e_ref_start, vis_e_ref_stop = int(ms2left[int(ts_i/1e3)]),int(ms2left[int(ts_i/1e3 + 10 ) % len(ms2left)])
                vis_e_idx_start, vis_e_idx_stop = int(ms2left[int(ts_e/1e3)]),\
                    int(ms2left[int(ts_e/1e3+ 10) % len(ms2left)])
                event_for_vis = {'x':torch.from_numpy(f['/prophesee/left/x'][vis_e_idx_start:vis_e_idx_stop-1].copy().astype(float)).to(torch.float32),
                'y': torch.from_numpy(f['/prophesee/left/y'][vis_e_idx_start:vis_e_idx_stop-1].copy().astype(float)).to(torch.float32),
                't': torch.from_numpy(f['/prophesee/left/t'][vis_e_idx_start:vis_e_idx_stop-1].copy().astype(float)).to(torch.float32),
                'p': torch.from_numpy(f['/prophesee/left/p'][vis_e_idx_start:vis_e_idx_stop-1].copy().astype(float)*2-1).to(torch.float32)
                }
                event_paired_for_vis = {'x':torch.from_numpy(f['/prophesee/left/x'][vis_e_ref_start:vis_e_ref_stop-1].copy().astype(float)).to(torch.float32),
                'y': torch.from_numpy(f['/prophesee/left/y'][vis_e_ref_start:vis_e_ref_stop-1].copy().astype(float)).to(torch.float32),
                't': torch.from_numpy(f['/prophesee/left/t'][vis_e_ref_start:vis_e_ref_stop-1].copy().astype(float)).to(torch.float32),
                'p': torch.from_numpy(f['/prophesee/left/p'][vis_e_ref_start:vis_e_ref_stop-1].copy().astype(float)*2-1).to(torch.float32)
                }
                event = {'x':torch.from_numpy(f['/prophesee/left/x'][e_idx_start:e_idx_stop-1].copy().astype(float)).to(torch.float32),
                'y': torch.from_numpy(f['/prophesee/left/y'][e_idx_start:e_idx_stop-1].copy().astype(float)).to(torch.float32),
                't': torch.from_numpy(f['/prophesee/left/t'][e_idx_start:e_idx_stop-1].copy().astype(float)).to(torch.float32),
                'p': torch.from_numpy(f['/prophesee/left/p'][e_idx_start:e_idx_stop-1].copy().astype(float)*2-1).to(torch.float32)
                }
                event_paired = {'x': torch.from_numpy(f['/prophesee/left/x'][e_ref_start:e_ref_stop-1].copy().astype(float)).to(torch.float32),
                'y': torch.from_numpy(f['/prophesee/left/y'][e_ref_start:e_ref_stop-1].copy().astype(float)).to(torch.float32),
                't': torch.from_numpy(f['/prophesee/left/t'][e_ref_start:e_ref_stop-1].copy().astype(float)).to(torch.float32),
                'p': torch.from_numpy(f['/prophesee/left/p'][e_ref_start:e_ref_stop-1].copy().astype(float)*2-1).to(torch.float32)
                }
                

                image = f['/ovc/left/data'][idx_i]
                image_paired = f['/ovc/left/data'][idx_e]

                
        event_data, vis1 = construct_repre(event, event_for_vis, ts_e)
        event_data_paired, vis0 = construct_repre(event_paired, event_paired_for_vis, ts_i)


        return event_data_paired, event_data, ts_i, ts_e, torch.from_numpy(image).to(torch.float32).squeeze(-1).unsqueeze(0)/255, vis0, vis1 # or image event_data
    
    def __getitem__(self, index):
        name0, name1 = self.pair_info[index]
        # parent0 = name0.split('_')[0]
        # parent1 = name1.split('_')[0]
        # data0 = None
        # data1 = None

        if self.repre == 'reconstruction':
            data0, data1, t0, t1, image_paired, vis0, vis1 = self.load_h5(name0, name1)
            pose0, pose1 = self.read_pose(t0, t1)
            with h5py.File(self.data_h5) as f:
                calib_e = f['/prophesee/left/calib']
                K0, D0, T = self.read_calib(calib_e)
                K1 = K0
                D1 = D0
            if not self.homo:
                data0 = image_paired
                with h5py.File(self.data_h5) as f:
                    calib_e = f['/prophesee/left/calib']
                    calib_i = f['/ovc/left/calib']
                    K0, D0, T = self.read_calib(calib_i)
                    K1, D1, _ = self.read_calib(calib_e)
        elif self.repre == 'image':
            pass
        elif self.homo:
            data0, data1, t0, t1, _, vis0, vis1 = self.load_h5(name0, name1)
            pose0, pose1 = self.read_pose(t0, t1)
            with h5py.File(self.data_h5) as f:
                calib_e = f['/prophesee/left/calib']
                K0, D0, _ = self.read_calib(calib_e)
                K1 = K0
                D1 = D0
        else:
            _, data1, t0, t1, data0 , vis0, vis1 = self.load_h5(name0, name1)
            pose0, pose1 = self.read_pose(t0, t1)
            with h5py.File(self.data_h5) as f:
                calib_e = f['/prophesee/left/calib']
                calib_i = f['/ovc/left/calib']
                K0, D0, T = self.read_calib(calib_i)
                K1, D1, _ = self.read_calib(calib_e)

            pose0 = pose0 @ T
        data0_resized, K0_new = self.undistort_resize(data0, K0, D0)
        data1_resized, K1_new = self.undistort_resize(data1, K1, D1)
        if vis0 is not None:
            vis0, _ = self.undistort_resize(vis0, K0, D0)
            vis0 = vis0.squeeze(0)
        else:
            vis0 = torch.empty(0)
        if vis1 is not None:
            vis1, _ = self.undistort_resize(vis1, K0, D0)
            vis1 = vis1.squeeze(0)
        else:
            vis1 = torch.empty(0)

        return {
            'image':data0_resized,
            'pose0':torch.from_numpy(pose0).to(torch.float32),
            'K0':torch.from_numpy(K0_new).to(torch.float32),
            'event':data1_resized,
            'pose1':torch.from_numpy(pose1).to(torch.float32),
            'K1':torch.from_numpy(K1_new).to(torch.float32),
            'T_0to1':torch.from_numpy(np.linalg.inv(pose1) @ pose0).to(torch.float32),
            'vis0': vis0,
            'vis1': vis1
        }


class EDSDataset(Dataset):
    def __init__(self,root_dir, homo = True, repre = '',
                 config = None):
        self.root_dir = root_dir
        self.homo = homo
        self.src_res = [480,640]
        self.dst_res = config['val_res']
        # self.ransac_thr = config['ransac_thr']
        self.interval = config['interval_ms'] * 1e3
        self.num_bin = config['num_bin']
        self.file_name = root_dir.split('/')[-1]
        # self.pair_info = self.parse_eds(opj(os.path.dirname(self.root_dir), 'eds_indices.json'), self.file_name)
        # self.pair_info = [[(506-4) *3,(506+i)*3] for i in range(100)]
        self.pair_info = [[(173-4-i+1800) *3,(255+1800-16-4-i)*3] for i in range(100)]
        # self.pair_info = [[506 *3,(970+7)*3]]
        # self.pair_info = [[506 *3,(970+i)*3] for i in range(100)]
        self.pose = np.genfromtxt(opj(self.root_dir, 'stamped_groundtruth.txt'))
        self.calib_dir = opj(os.path.dirname(self.root_dir), 'calib.txt')

        assert repre in config['repres']
        self.repre = repre

        self.calibration = np.genfromtxt(self.calib_dir)
        self.K = np.array([
            [self.calibration[0],0,self.calibration[2]],
            [0, self.calibration[1], self.calibration[3]],
            [0,0,1]
        ])
        self.D = self.calibration[4:]

        events_t = h5py.File(opj(self.root_dir,'events.h5'), 'r')["t"][:]
        tss = self.get_all_ts()

        if self.repre == 'event_stack':
            self.end_event_idx = np.searchsorted(events_t, np.array(tss) * 1e6 + 4*self.interval )
            self.current_event_idx = np.searchsorted(events_t, np.array(tss)* 1e6)
            self.repre_end_idx = np.searchsorted(events_t, np.array(tss)* 1e6 + self.interval)

        elif self.repre == 'mcts':
            self.end_event_idx = np.searchsorted(events_t, np.array(tss)* 1e6+ 5*self.interval)
            self.current_event_idx = np.searchsorted(events_t, np.array(tss)* 1e6 )
            self.repre_end_idx = np.searchsorted(events_t, np.array(tss)* 1e6 + self.interval)
        else:
            self.end_event_idx = np.searchsorted(events_t, np.array(tss) * 1e6 + self.interval)
            self.current_event_idx = np.searchsorted(events_t, np.array(tss)* 1e6)
            self.repre_end_idx = np.searchsorted(events_t, np.array(tss)* 1e6 + self.interval)

        # self.end_event_idx = np.searchsorted(events_t, np.array(tss) * 1e6 + self.interval )

    def parse_eds(self, json_file, seq_name):
        with open(json_file, "r") as f:
            data = json.load(f)

        index = None
        for item in data:
            if seq_name == item["name"]:
                index = item["indexs"]
        
        return index

    def __len__(self):
        return len(self.pair_info)
    
    def read_im(self, dir):
        # TODO: Change this for hetero
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # Convert to float32 and normalize to 0-1
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img[None,...]).to(torch.float32)
    
    def undistort_resize(self, data, K, D):

        h,w = data.shape[-2:]
        coeff = np.array(self.dst_res)/np.array([h,w])
        K_new = K.copy()
        K_new[0] *= coeff[1]
        K_new[1] *= coeff[0]

        data_np = np.transpose(data.numpy(),(1,2,0))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R=np.eye(3), newCameraMatrix=K,
                                         size=(w, h), m1type=cv2.CV_32FC1)
        undistorted = cv2.remap(data_np, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

        if len(undistorted.shape) == 2:
            undistorted = undistorted[...,None]
        undistorted = np.transpose(undistorted,(2,0,1))

        # Cropping to fit the target ratio
        # h_new = h
        # w_new = w
        # if 1.0*h/w >= 1.0*self.dst_res[0]/self.dst_res[1]:
        #     h_new = int(w * self.dst_res[0]/self.dst_res[1])
        # else:
        #     w_new = int(h * self.dst_res[1]/self.dst_res[0])
        # undistorted = undistorted[:,:h_new-1,:w_new-1]

        data_torch = torch.from_numpy(undistorted)
        data_torch = F.interpolate(data_torch.unsqueeze(0).float(),
                                    size=self.dst_res,
                                    mode='nearest'
                                    ).squeeze(0)

        return data_torch, K_new
    
    def get_all_ts(self):
        
        return [annot[0] for annot in self.pose]
    
    def read_pose(self, index):
        annot = self.pose[index]
        ts = annot[0]
        trans = annot[1:4]
        rot = R.from_quat(annot[4:]).as_matrix()
        pose = np.eye(4)
        pose[:3,:3] = rot
        pose[:3,-1] = trans
        return ts,pose
    
    def load_h5(self, index):
        events = h5py.File(opj(self.root_dir,'events.h5'), 'r')
        current_event_idx = self.current_event_idx[index]
        end_event_idx = self.end_event_idx[index]
        # event_batch = torch.from_numpy(np.vstack([(
        #     events_t[current_event_idx:end_event_idx] - events_t[0]) * 1e-6,
        #     events['x'][current_event_idx:end_event_idx],
        #     events['y'][current_event_idx:end_event_idx],
        #     events['p'][current_event_idx:end_event_idx]]).T).to(torch.float32)

        print(current_event_idx, end_event_idx)
        event_batch = {
                'x':torch.from_numpy(events['x'][current_event_idx:end_event_idx].copy().astype(float)).to(torch.float32),
                'y':torch.from_numpy(events['y'][current_event_idx:end_event_idx].copy().astype(float)).to(torch.float32),
                't':torch.from_numpy(events['t'][current_event_idx:end_event_idx] - events['t'][0]).to(torch.float32),
                'p':torch.from_numpy(events['p'][current_event_idx:end_event_idx].copy().astype(float)).to(torch.float32) * 2 - 1,
            }
        # if self.repre == 'event_frame':
        #     vis_end_event_idx = end_event_idx
        # else:
        vis_end_event_idx = self.repre_end_idx[index]
        event_batch_vis = {
                'x':torch.from_numpy(events['x'][current_event_idx:vis_end_event_idx].copy().astype(float)).to(torch.float32),
                'y':torch.from_numpy(events['y'][current_event_idx:vis_end_event_idx].copy().astype(float)).to(torch.float32),
                't':torch.from_numpy(events['t'][current_event_idx:vis_end_event_idx] - events['t'][0]).to(torch.float32),
                'p':torch.from_numpy(events['p'][current_event_idx:vis_end_event_idx].copy().astype(float)).to(torch.float32) * 2 - 1,
            }
        
        import time 
        repre_start = time.perf_counter()
        converter = EventVis((3, self.src_res[0], self.src_res[1]))
        vis = converter.convert(event_batch_vis)
        if self.repre == 'event_frame':
            converter = EventFrame((self.src_res[0],self.src_res[1]))
            event_data = converter.convert(event_batch)
        elif self.repre == 'mcts':
            converter = TsGenerator(settings = {"shape": self.src_res, "delta_t": [0.001, 0.003, 0.01, 0.03, 0.1]})
            event_data = converter.convert(event_batch)
            event_data = event_data.permute(2,0,1)
        elif self.repre == 'event_stack':
            event_voxel = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
            data_event_pos = event_voxel.convert(get_event_polarity(event_batch, polarity=1))
            event_voxel_neg = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
            data_event_neg = event_voxel_neg.convert(get_event_polarity(event_batch, polarity=-1))
            event_data = stack_pos_neg(data_event_pos, data_event_neg)
            event_data = event_norm(event_data)
        else:
            pass

        repre_stop = time.perf_counter()

        print(f'build one rep {(repre_stop - repre_start)*1000:.1f} ms')
        if len(event_data.shape) == 2:
            event_data = event_data[None,...]

        return event_data, vis
    
    def __getitem__(self, index):
        idx0, idx1 = self.pair_info[index]

        if self.repre == 'reconstruction':
            pass
            # data0 = self.read_im(opj(self.root_dir,parent0, 'event', 'reconstruction', 'renamed', name0.split('.')[0]+'.png'))
            # data1 = self.read_im(opj(self.root_dir,parent1, 'event', 'reconstruction', 'renamed', name1.split('.')[0]+'.png'))
            # pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
            # pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt')) @ self.calibration['i_T_e']
            # K0 = self.calibration['event']['K']
            # K1 = self.calibration['event']['K']
            # D0 = self.calibration['event']['D']
            # D1 = self.calibration['event']['D']
        elif self.repre == 'image':
            pass
            # data0 = self.read_im(opj(self.root_dir, parent0, 'gt_hetero', name0))
            # data1 = self.read_im(opj(self.root_dir, parent1, 'gt_hetero', name1))
            # pose0 = self.read_pose(opj(self.pose_dir, name0.split('.')[0]+'_pose.txt'))
            # pose1 = self.read_pose(opj(self.pose_dir, name1.split('.')[0]+'_pose.txt'))
            # K0 = self.calibration['image']['K']
            # K1 = self.calibration['image']['K']
            # D0 = self.calibration['image']['D']
            # D1 = self.calibration['image']['D']
        elif self.homo:
            ts0, pose0 = self.read_pose(idx0) 
            ts1, pose1 = self.read_pose(idx1)
            data0, vis0 = self.load_h5(idx0)
            data1, vis1 = self.load_h5(idx1)
            
        else:
            pass
        K0 = self.K
        K1 = self.K
        D0 = self.D
        D1 = self.D
        data0_resized, K0_new = self.undistort_resize(data0, K0, D0)
        data1_resized, K1_new = self.undistort_resize(data1, K1, D1)
        if vis0 is not None:
            vis0, _ = self.undistort_resize(vis0, K0, D0)
            vis0 = vis0.squeeze(0)
        else:
            vis0 = torch.empty(0)
        if vis1 is not None:
            vis1, _ = self.undistort_resize(vis1, K0, D0)
            vis1 = vis1.squeeze(0)
        else:
            vis1 = torch.empty(0)

        return {
            'image':data0_resized,
            'pose0':torch.from_numpy(pose0).to(torch.float32),
            'K0':torch.from_numpy(K0_new).to(torch.float32),
            'event':data1_resized,
            'pose1':torch.from_numpy(pose1).to(torch.float32),
            'K1':torch.from_numpy(K1_new).to(torch.float32),
            'T_0to1':torch.from_numpy(np.linalg.inv(pose1) @ pose0).to(torch.float32),
            'D':self.D,
            'vis0': vis0,
            'vis1': vis1
        }
    
class EMegaDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 normalize_event = False,
                 train_data = '',
                 ignore_list = [''],
                 min_overlap_score=0.3,
                 max_overlap_score=0.3,
                 train_res = (336, 560),
                 **kwargs):
        """
        Manage one scene(npz_path) of MegaDepth dataset.
        
        Args:
            root_dir (str): megadepth root directory that has `phoenix`.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test']
            min_overlap_score (float): how much a pair should have in common. In range of [0, 1]. Set to 0 when testing.
            img_resize (int, optional): the longer edge of resized images. None for no resize. 640 is recommended.
                                        This is useful during training with batches and testing with memory intensive algorithms.
            df (int, optional): image size division factor. NOTE: this will change the final image size after img_resize.
            img_padding (bool): If set to 'True', zero-pad the image to squared size. This is useful during training.
            depth_padding (bool): If set to 'True', zero-pad depthmap to (2000, 2000). This is useful during training.
            augment_fn (callable, optional): augments images with pre-defined visual effects.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.scene_id = npz_path.split('/')[-1].split('_')[0]
        self.event_norm = normalize_event
        self.train_data = train_data
        self.resolution = (360, 640)
        self.dst_res = train_res
        # TODO: Test with other representation
        self.repre = 'event_stack'
        self.num_bin = 8

        # prepare scene_info and pair_info
        if mode == 'test' and min_overlap_score != 0:
            print("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = 0
        self.scene_info = np.load(npz_path, allow_pickle=True)

        self.pair_infos = self.scene_info['pair_infos'].copy()


        del self.scene_info['pair_infos']
        self.pair_infos = [pair_info for pair_info in self.pair_infos if (pair_info[1] > min_overlap_score) and (pair_info[1] < max_overlap_score)]

        if self.scene_id in ignore_list:
            self.pair_infos = []

        # parameters for image resizing, padding and depthmap padding

        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        
    def __len__(self):
        return len(self.pair_infos)
    
    def undistort_resize(self, data, K, D):

        h,w = data.shape[-2:]
        coeff = np.array(self.dst_res)/np.array([h,w])
        K_new = K.copy()
        K_new[0] *= coeff[1]
        K_new[1] *= coeff[0]

        data_np = np.transpose(data.numpy(),(1,2,0))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R=np.eye(3), newCameraMatrix=K,
                                         size=(w, h), m1type=cv2.CV_32FC1)
        undistorted = cv2.remap(data_np, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

        if len(undistorted.shape) == 2:
            undistorted = undistorted[...,None]
        undistorted = np.transpose(undistorted,(2,0,1))

        # Cropping to fit the target ratio
        # h_new = h
        # w_new = w
        # if 1.0*h/w >= 1.0*self.dst_res[0]/self.dst_res[1]:
        #     h_new = int(w * self.dst_res[0]/self.dst_res[1])
        # else:
        #     w_new = int(h * self.dst_res[1]/self.dst_res[0])
        # undistorted = undistorted[:,:h_new-1,:w_new-1]

        data_torch = torch.from_numpy(undistorted)
        data_torch = F.interpolate(data_torch.unsqueeze(0).float(),
                                    size=self.dst_res,
                                    mode='nearest'
                                    ).squeeze(0)

        return data_torch, K_new
    
    def load_h5(self, name):

        with h5py.File(opj(self.root_dir, self.scene_id, 'images',  name+'.h5'), 'r') as f:
            event_batch = {'x':torch.from_numpy(f['event/x'][:].copy().astype(float)).to(torch.float32),
                'y': torch.from_numpy(f['event/y'][:].copy().astype(float)).to(torch.float32),
                't': torch.from_numpy(f['event/t'][:].copy().astype(float)).to(torch.float32),
                'p': torch.from_numpy(f['event/p'][:].copy().astype(float)).to(torch.float32) # Polarity already in -1,1
                }
            
            vis_index = torch.searchsorted(event_batch['t'], event_batch['t'][0]+30*1e6) # TODO: Verify
            event_vis = {
                'x':event_batch['x'][:vis_index],
                'y':event_batch['y'][:vis_index],
                't':event_batch['t'][:vis_index],
                'p':event_batch['p'][:vis_index]
            }
            # event = torch.from_numpy(f['event'][:]).to(torch.float32)
            image = torch.from_numpy(f['image'][:].astype(int)).to(torch.float32)/255.0
            pose = torch.from_numpy(f['pose'][:]).to(torch.float32)
            depth = torch.from_numpy(f['depth'][:]).to(torch.float32)
            K = (f['K'][:])

        converter = EventVis((3, self.resolution[0], self.resolution[1]),classic=True)
        vis = converter.convert(event_vis)
        if self.repre == 'event_frame':
            converter = EventFrame((self.resolution[0],self.resolution[1]))
            event_data = converter.convert(event_batch)
        elif self.repre == 'mcts':
            converter = TsGenerator(settings = {"shape": self.resolution, "delta_t": [0.001, 0.003, 0.01, 0.03, 0.1]})
            event_data = converter.convert(event_batch)
            event_data = event_data.permute(2,0,1)
        elif self.repre == 'event_stack':
            event_voxel = Adaptive_interval((self.num_bin, self.resolution[0], self.resolution[1]), normalize=False, aug=0)
            data_event_pos = event_voxel.convert(get_event_polarity(event_batch, polarity=1))
            event_voxel_neg = Adaptive_interval((self.num_bin, self.resolution[0], self.resolution[1]), normalize=False, aug=0)
            data_event_neg = event_voxel_neg.convert(get_event_polarity(event_batch, polarity=-1))
            event_data = stack_pos_neg(data_event_pos, data_event_neg)
            event_data = event_norm(event_data)
        else:
            pass
    
        # depth[depth>1e3] = torch.inf
        return event_data, image, pose, depth, K, vis

    def __getitem__(self, idx):
        (idx0, idx1), _, _ = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = self.scene_info['image_paths'][idx0].split('/')[-1].split('.')[0]
        img_name1 = self.scene_info['image_paths'][idx1].split('/')[-1].split('.')[0]
    
        data0, image, T0, depth0, K0, vis0 = self.load_h5(img_name0)
        data1, image_paired, T1, depth1, K1, vis1 = self.load_h5(img_name1)

        train = False
        if self.mode == 'train':
            train = True
        
        # event = temporal_augmentation(event, train=train)
        # event_paired = temporal_augmentation(event_paired, train=train)

        T_0to1 = T1 @ torch.inverse(T0)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data0_resized, K0_new = self.undistort_resize(data0, K0, np.zeros(5))
        data1_resized, K1_new = self.undistort_resize(data1, K1, np.zeros(5))

        if vis0 is not None:
            vis0, _ = self.undistort_resize(vis0, K0, np.zeros(5))
            vis0 = vis0.squeeze(0)
        else:
            vis0 = data0_resized
        if vis1 is not None:
            vis1, _ = self.undistort_resize(vis1, K0, np.zeros(5))
            vis1 = vis1.squeeze(0)
        else:
            vis1 = data1_resized
            



        data = {
            'image': data0_resized,#.unsqueeze(0),  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'event': data1_resized,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K0_new,  # (3, 3)
            'K1': K1_new,
            'vis0':vis0,
            'vis1':vis1,
            # 'scale0': scale0,  # [scale_w, scale_h]
            # 'scale1': scale1,
            # 'dataset_name': 'MegaDepth',
            # 'scene_id': self.scene_id,
            # 'pair_id': idx,
            #
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }
        # if self.train_data == 'events':
        #     data['image'] = event_paired
        # if self.train_data == 'image':
        #     data['event'] = image_paired
        # for LoFTR training
        # if mask0 is not None:  # img_padding is True
        #     if self.coarse_scale:
        #         [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
        #                                                 scale_factor=self.coarse_scale,
        #                                                 mode='nearest',
        #                                                 recompute_scale_factor=False)[0].bool()
        #     data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})

        return data
    
   
class CustomDataset(Dataset):
    def __init__(self,root_dir, homo = True, repre = '', roi_start = (200,80),
                 config = None, ref_index = None):
        self.root_dir = root_dir
        self.event_dir = root_dir#opj(root_dir, 'event')
        self.homo = homo
        self.src_res = config['resolution']
        self.dst_res = config['val_res']
        # self.ransac_thr = config['ransac_thr']
        self.interval = config['interval_ms'] * 1e3
        self.num_bin = config['num_bin']
        self.roi = roi_start # TODO: Add ROI setup
        self.ref_index = ref_index
        self.calib_dir = opj(self.root_dir, 'calib.yaml')
        if not os.path.isfile(self.calib_dir):
            self.calibration = {'event':{
                    'K':np.eye(3),
                    'D':np.zeros(5),
                    'res':self.src_res
                         },
                'image':{
                    'K':np.eye(3),
                    'D':np.zeros(5),
                    'res':self.src_res
                },
                }
        else:
            self.calibration = read_calib(self.calib_dir)


        assert repre in config['repres']
        self.repre = repre
        self.event_h5 = opj(self.event_dir, 'event.h5')
        self.event_ref = opj(self.event_dir, 'ref.h5')
        ts_file = opj(self.event_dir, 'trigger.txt')
        

        with h5py.File(self.event_h5, 'r') as f:
            t = f['t'][:]

            if os.path.isfile(ts_file):
                self.ts = self.read_ts(ts_file)
            else:
                self.ts = np.arange(start=0, stop=t[-1], step=0.0663 * 1e6) # 30 hz comparison by default
           
            t_start = self.ts
            t_stop = t_start + self.interval
            t_stop_for_visualize = t_start + 30 * 1e3

            self.idx_start, self.idx_stop, self.idx_stop_vis = np.searchsorted(t, [t_start, t_stop, t_stop_for_visualize])

        

    def __len__(self):
        return len(self.ts)
    
    def read_ts(self, file):
        return np.genfromtxt(file)
    
    def read_im(self, dir):
        img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # Convert to float32 and normalize to 0-1
        H,W = img.shape
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img[None,...]).to(torch.float32)
    
    def undistort_resize(self, data, K, D, roi = None):
        h,w = data.shape[-2:]
        coeff = np.array(self.dst_res)/np.array([h,w])
        K_new = K.copy()
        K_new[0] *= coeff[1]
        K_new[1] *= coeff[0]

        data_np = np.transpose(data.numpy(),(1,2,0))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R=np.eye(3), newCameraMatrix=K,
                                         size=(w, h), m1type=cv2.CV_32FC1)
        undistorted = cv2.remap(data_np, map1, map2, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)

        if len(undistorted.shape) == 2:
            undistorted = undistorted[...,None]
        undistorted = np.transpose(undistorted,(2,0,1))

        # Cropping to fit the target ratio
        # h_new = h
        # w_new = w
        # if 1.0*h/w >= 1.0*self.dst_res[0]/self.dst_res[1]:
        #     h_new = int(w * self.dst_res[0]/self.dst_res[1])
        # else:
        #     w_new = int(h * self.dst_res[1]/self.dst_res[0])
        # undistorted = undistorted[:,:h_new-1,:w_new-1]

        data_torch = torch.from_numpy(undistorted)
        if roi is not None:
            data_torch = data_torch[:,roi[-1]:roi[-1]+self.dst_res[0], roi[0]:roi[0]+self.dst_res[-1]]

        data_torch = F.interpolate(data_torch.unsqueeze(0).float(),
                                    size=self.dst_res,
                                    mode='nearest'
                                    ).squeeze(0)

        return data_torch, K_new
    
    
    def load_h5(self, dir, index):
        with h5py.File(dir, 'r') as f:

            idx_start, idx_stop, idx_stop_vis = self.idx_start[index], self.idx_stop[index], self.idx_stop_vis[index]

            event = {
                'x':torch.from_numpy(f['x'][idx_start:idx_stop].copy().astype(float)).to(torch.float32),
                'y':torch.from_numpy(f['y'][idx_start:idx_stop].copy().astype(float)).to(torch.float32),
                't':torch.from_numpy(f['t'][idx_start:idx_stop].copy().astype(float)).to(torch.float32),
                'p':torch.from_numpy(f['p'][idx_start:idx_stop].copy().astype(float)).to(torch.float32) * 2 - 1,
            }
            event_vis = {
                'x':torch.from_numpy(f['x'][idx_start:idx_stop_vis].copy().astype(float)).to(torch.float32),
                'y':torch.from_numpy(f['y'][idx_start:idx_stop_vis].copy().astype(float)).to(torch.float32),
                't':torch.from_numpy(f['t'][idx_start:idx_stop_vis].copy().astype(float)).to(torch.float32),
                'p':torch.from_numpy(f['p'][idx_start:idx_stop_vis].copy().astype(float)).to(torch.float32) * 2 - 1,
            }


        import time 
        repre_start = time.perf_counter()
        vis = None
        
        converter = EventVis((3, self.src_res[0], self.src_res[1]))
        vis = converter.convert(event_vis)
        if self.repre == 'event_frame':
            converter = EventFrame((self.src_res[0],self.src_res[1]))
            event_data = converter.convert(event)
        elif self.repre == 'mcts':
            converter = TsGenerator(settings = {"shape": self.src_res, "delta_t": [0.001, 0.003, 0.01, 0.03, 0.1]})
            event_data = converter.convert(event)
            event_data = event_data.permute(2,0,1)
        elif self.repre == 'event_stack':
            event_voxel = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
            data_event_pos = event_voxel.convert(get_event_polarity(event, polarity=1))
            event_voxel_neg = Adaptive_interval((self.num_bin, self.src_res[0], self.src_res[1]), normalize=False, aug=0)
            data_event_neg = event_voxel_neg.convert(get_event_polarity(event, polarity=-1))
            event_data = stack_pos_neg(data_event_pos, data_event_neg)
            event_data = event_norm(event_data)
        else:
            pass

        repre_stop = time.perf_counter()

        print(f'build one rep {(repre_stop - repre_start)*1000:.1f} ms')
        if len(event_data.shape) == 2:
            event_data = event_data[None,...]

        return event_data, vis
    
    def __getitem__(self, index):
        vis0 = None
        if not self.homo:
            data0 = self.read_im(opj(self.root_dir, 'ref.jpg'))
        else:
            if self.ref_index is not None:
                data0, vis0 = self.load_h5(self.event_ref, self.ref_index)
            else:
                raise ValueError("Please provide ref index for event - event matching")
        data1, vis1 = self.load_h5(self.event_h5, index)
        K0 = self.calibration['image']['K']
        K1 = self.calibration['event']['K']
        D0 = self.calibration['image']['D']
        D1 = self.calibration['event']['D']
        data0_resized, K0_new = self.undistort_resize(data0, K0, np.zeros(5))
        data1_resized, K1_new = self.undistort_resize(data1, K1, D1, roi = self.roi)

        if vis0 is not None:
            vis0, _ = self.undistort_resize(vis0, K0, D0)
            vis0 = vis0.squeeze(0)
        else:
            vis0 = data0_resized
        if vis1 is not None:
            vis1, _ = self.undistort_resize(vis1, K0, D0, roi = self.roi)
            vis1 = vis1.squeeze(0)
        else:
            vis1 = data1_resized

    
        return {
            'image':data0_resized,
            # 'pose0':torch.from_numpy(pose0),
            'K0':torch.from_numpy(K0_new),
            'event':data1_resized,
            # 'pose1':torch.from_numpy(pose1),
            'K1':torch.from_numpy(K1_new),
            # 'T_0to1':torch.from_numpy(np.linalg.inv(pose1) @ pose0).to(torch.float32),
            'D':D0,
            'D1':D1,
            'vis0': vis0,
            'vis1': vis1
        }
    
if __name__=="__main__":
    edm = EDMDataset(root_dir='/media/rex/rex_4t/data/EDM/Franklin_corner', homo=True, repre='mcts')
    for i in range(20):
        data_dict = edm[i]
        filtered = {k: v for k, v in data_dict.items() if k.startswith('data')}
        fig, axs = plt.subplots(1, len(filtered), figsize=(10, 4))
        if len(filtered) == 1:
            axs = [axs]  # ensure iterable

        for ax, (name, data) in zip(axs, filtered.items()):
            # Remove channel dimension (1,H,W) → (H,W)
            #img = data.squeeze(0)
            img = ts2image(data.permute(1,2,0))
            # if len(img.shape) == 3:
            #     img = torch.mean(img, dim=0)
            im = ax.imshow(img, cmap='gray')
            ax.set_title(name)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()






