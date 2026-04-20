import torch
import h5py
import numpy as np
import yaml
import os
import sys
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R
sys.path.append('./dataset/EDM')
from evaluate import plot_matched_points

import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from representations import EventFrame, Adaptive_interval, TsGenerator, EventVis
from dataset.prepare_m3ed import get_event_polarity, stack_pos_neg, get_pose_t
from dataset.data_loader import event_norm
import torch.nn.functional as F
from os.path import join as opj

class ECDDataset(Dataset):
    def __init__(self,root_dir):
        self.interval_ms = 40
        self.root_dir = root_dir
        self.src_res = [180,240]
        self.dst_res = self.src_res
        self.image_dir = opj(self.root_dir, 'images')
        self.ts_dir = opj(self.root_dir, 'images.txt')
        self.event_dir = opj(self.root_dir, 'events.txt')
        self.image_names = [opj(self.image_dir, dir) for dir in sorted(os.listdir(self.image_dir))]
        self.ts = self.load_ts()
        self.events = self.load_events() # t,x,y,p
        self.start_index, self.stop_index = np.searchsorted(self.events[:,0], np.array([self.ts, self.ts+self.interval_ms * 1e-3]))

        assert len(self.image_names) == len(self.ts)

    def __len__(self):
        return len(self.image_names)
    
    def load_ts(self):
        image_info = np.genfromtxt(self.ts_dir)
        return image_info[:,0]
    
    def load_events(self):
        return np.genfromtxt(self.event_dir)
    
    def load_image(self, index):
        img = cv2.imread(self.image_names[index], cv2.IMREAD_GRAYSCALE)  # shape: (H, W)
        # Convert to float32 and normalize to 0-1
        H,W = img.shape
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img[None,...]).to(torch.float32)
    
    def __getitem__(self, index):
        event_start = self.start_index[index]
        event_stop = self.stop_index[index]
        event = {
            'x':torch.from_numpy(self.events[event_start:event_stop,1].copy().astype(float)).to(torch.float32),
            'y':torch.from_numpy(self.events[event_start:event_stop,2].copy().astype(float)).to(torch.float32),
            't':torch.from_numpy(self.events[event_start:event_stop,0].copy().astype(float)).to(torch.float32),
            'p':torch.from_numpy(self.events[event_start:event_stop,3].copy().astype(float)).to(torch.float32) * 2 - 1,
            }
        
        converter = EventVis((3, self.src_res[0], self.src_res[1]), classic=True)
        vis = converter.convert(event)

        image = self.load_image(index)

        

        return {'image':image, 'vis1': vis}
    

if __name__ == '__main__':
    # from edm_dataset import ECMDataset
    EDM_CONFIG = {
    'resolution': [720, 1280],
    # 'val_res': [360, 640], # superevent
    # 'val_res': [476, 630], # eds
    # 'val_res': [352, 640], # match anything
    # 'val_res': [336, 560], # EDMc
    'val_res': [350, 630], # vggt
    'interval_ms': 60,
    'num_bin':8,
    'repres': ['event_stack', 'mcts', 'event_frame', 'reconstruction', 'image'],
    'min_overlap_score': 0.0,
    'epipolar_threshold':1e-4
}
    # dataset = ECMDataset('/media/rex/rex_4t/data/EDM/Franklin_main', False, 'event_stack',EDM_CONFIG)

    dataset = ECDDataset('/home/rex/Downloads/shapes_6dof')
    for i, data in enumerate(dataset):
        fig_pred = plot_matched_points(data['image'].numpy(), data['vis1'].numpy(),None,None,None, path = './paperwriting/output%05d.png' % i)
        # cv2.imwrite("teaser.png", fig_pred)
        # cv2.imshow('win',fig_pred)
        # cv2.waitKey(0)
