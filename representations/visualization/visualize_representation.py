import h5py
import os
import glob
import numpy as np
import torch
import cv2
import argparse
import matplotlib.pyplot as plt

from adaptive_interval import Adaptive_interval

dataset_config = {
    'root_dir' : '/media/rex/rex_4t/data/m3ed',
    'save_dir': '/media/rex/rex_4t/data/hetero_m3ed',
    'test_seq' : 'car_urban_day_penno_small_loop',
    'shape': [720, 1280],
    'target_res': [360, 640],
    'num_bins':10,
    'event_interval_ms':256,
    'sample_interval': [0.5*1e6, 1.5*1e6],
    'seed': 42,
    'valid_img_margin':0.02*1e6,
    'event_min_number':1000000,
    'min_trans': 1.0,
    'rot_range': [5, 30]
}

def undistort_events(calib, eventx, eventy):
        pts = np.hstack([eventx.reshape(-1,1),eventy.reshape(-1,1)]).reshape(-1,1,2)
        dist = np.array(calib['distortion_coeffs'])
        intri = calib['intrinsics'][:].copy()
        K0_mat = np.array([
            [intri[0], 0, intri[2]],
            [0, intri[1], intri[3]],
            [0, 0, 1]
        ])
        T = calib['T_to_prophesee_left'][:].copy()
        undistorted = cv2.undistortPoints(pts, K0_mat, dist, P=K0_mat)
        return undistorted

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--idx', type=int, default=2000)
    parser.add_argument(
        '--seq', type=str, default='car_urban_day_city_hall_data.h5')
    parser.add_argument(
        '--intv_ms', type=int, default=128)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    file_path = os.path.join(dataset_config['root_dir'], args.seq)
    with h5py.File(file_path) as f:
        image_ts = f['/ovc/ts'][:].copy()
        ms2left = f['/prophesee/left/ms_map_idx'][:].copy()
        
    assert args.idx < len(image_ts)

    ref_ts = image_ts[args.idx]
    e_ref_start, e_ref_stop = int(ms2left[int(ref_ts/1e3 - args.intv_ms/2)]),int(ms2left[int(ref_ts/1e3 + args.intv_ms/2) % len(ms2left)])
    if e_ref_stop - e_ref_start < dataset_config['event_min_number']:
        print('Not enough event')
        exit()
    
    with h5py.File(file_path) as f:
        calib_e = f['/prophesee/left/calib']
        calib_i = f['/ovc/left/calib']
        event = {'x':torch.from_numpy(f['/prophesee/left/x'][e_ref_start:e_ref_stop-1].copy().astype(int)).to(torch.float32),
                    'y':torch.from_numpy(f['/prophesee/left/y'][e_ref_start:e_ref_stop-1].copy().astype(int)).to(torch.float32),
                    't':torch.from_numpy(f['/prophesee/left/t'][e_ref_start:e_ref_stop-1].copy().astype(int)).to(torch.float32),
                    'p':torch.from_numpy(f['/prophesee/left/p'][e_ref_start:e_ref_stop-1].copy().astype(int)).to(torch.float32)*2-1
                    }
        # rec = undistort_events(calib_e, event['x'].numpy(), event['y'].numpy())
        # event['x'] = torch.from_numpy(rec[...,0].flatten())
        # event['y'] = torch.from_numpy(rec[...,1].flatten())

    # with h5py.File(file_path) as f:
    #     event = {
    #          'x': torch.from_numpy(f['event/x'][:]).to(torch.float32),
    #          'y': torch.from_numpy(f['event/y'][:]).to(torch.float32),
    #          't': torch.from_numpy(f['event/t'][:]).to(torch.float32),
    #          'p': torch.from_numpy(f['event/p'][:]).to(torch.float32)
    #     }

    import time
    start = time.perf_counter()
    
    event_representation = Adaptive_interval((dataset_config['num_bins'], dataset_config['shape'][0], dataset_config['shape'][1]),normalize=False)
    data_event = event_representation.convert(event)

    end = time.perf_counter()
    print(f"Time taken for converting 4: {(end - start):.4f} seconds")

    fig, axs = plt.subplots(5, 3, figsize=(12, 20))  # 3 rows x 5 columns for 15 channels
    axs = axs.flatten()

    for i in range(dataset_config['num_bins']):
        slice_to_plot = data_event[i].numpy()
        # import pdb
        # pdb.set_trace()
        slice_to_plot = slice_to_plot/np.abs(slice_to_plot).max()/2.0 + 0.5
        axs[i].imshow(slice_to_plot, cmap='gray', vmin=0.0, vmax = 1.0)
        axs[i].set_title(f'Channel {i}')
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
