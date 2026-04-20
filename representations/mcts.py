# Multi-channel Timesurface modified from https://github.com/ethz-mrl/SuperEvent, setting follows the training setting of super event

import numpy as np
import torch
import cv2

model_delta_t = [0.001, 0.003, 0.01, 0.03, 0.1]

class TsGenerator:
    def __init__(self, camera_matrix=np.identity(3), distortion_coeffs=np.zeros(5), settings={}, device="cpu"):
        # Process settings
        default_settings = {"shape": [184, 240], "delta_t": [0.01], "undistort": False}
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.settings = settings
        self.device = device

        # Sanity checks
        if "shape" not in self.settings.keys() or not len(self.settings["shape"]) == 2:
            self.settings["shape"] = default_settings["shape"]
            print("TsGenerator: Using default shape setting:", default_settings["shape"])
        else:
            self.settings["shape"] = list(self.settings["shape"])  # make sure its a list

        if "delta_t" not in self.settings.keys() \
            or not np.all(np.array(self.settings["delta_t"]) > 0.):  # not all to also check empty list
            self.settings["delta_t"] = default_settings["delta_t"]
            print("TsGenerator: Using default delta_t setting:", default_settings["delta_t"])
        self.ts_dim = torch.tensor([self.settings["delta_t"]]).reshape([-1]).to(self.device)  # ensure exactly one dim

        if "undistort" not in self.settings.keys():
            self.settings["undistort"] = default_settings["undistort"]

        # Support for fisheye lens undistortion in mvsec dataset
        if self.settings["undistort"]:
            assert (self.device == "cpu"), "Undistortion is only supported on cpu!"
            if "fisheye_lens" in self.settings.keys() and settings["fisheye_lens"]:
                self.fisheye_lens_used = True
                self.new_camera_matrix = settings["new_camera_matrix"]
                self.valid_image_shape = settings["crop_to_idxs"]
                print("TsGenerator: Using fisheye lens camera model.")
            else:
                self.fisheye_lens_used = False

        # Initialize time stamp tracking
        self.time_stamps = torch.zeros(self.settings["shape"] + [2, 1], dtype=torch.float32).to(self.device)
    
    def update(self, t, x, y, p):
        # Assuming t is float32, x and y are int, and p is an int with value 0 or 1
        self.time_stamps[x, y, p] = t

    def batch_update(self, event_batch):
        # Only keep events with same x, y, p with lastest t
        sort_values = event_batch[:, 1] * 2 * torch.max(event_batch[:, 2] + 1) + event_batch[:, 2] * 2 + event_batch[:, 3]
        sort_values, sort_indeces = torch.sort(sort_values, dim=0, stable=True)
        event_batch = event_batch[sort_indeces]  # The tensor is now sorted in the order of row 1, then 2, then 3, then 0

        # Every x, y, p must be different from the next one, otherwise it is repeated and has not the most recent time stamp
        # The last element alwas has the most recent time stamp for its pixel
        keep_event_mask = sort_values[:-1] != sort_values[1:]
        keep_event_mask = torch.cat([keep_event_mask, torch.tensor([True], device=self.device)])
        event_batch = event_batch[keep_event_mask]

        # Assuming event_batch contains events with [t, x, y, p] with p being an int with value 0 or 1
        t = event_batch[:, 0].float()
        x = event_batch[:, 2].int()  # we use (row, column) instead of (x, y) coordinates
        y = event_batch[:, 1].int()
        p = event_batch[:, 3].int()
        self.time_stamps[x, y, p] = t[..., None]

        # Commented out for faster runtime
        #assert not any(self.time_stamps[x, y, p] < t[..., None])

    def get_ts(self):
        t_max = torch.max(self.time_stamps)
        ts = self.time_stamps - t_max
        
        ts = ts + self.ts_dim
        ts = torch.clamp(ts, min=0.)
        ts = ts / self.ts_dim
        ts = torch.reshape(ts, self.settings["shape"] + [2 * len(self.ts_dim)])

        # Undistort time surface (not supported on GPU)
        if self.settings["undistort"] and self.device == "cpu":
            if self.fisheye_lens_used:
                ts = cv2.fisheye.undistortImage(ts.numpy(), K=self.camera_matrix, D=self.distortion_coeffs[:4], Knew=self.new_camera_matrix)
                ts = ts[self.valid_image_shape[0]:self.valid_image_shape[1], self.valid_image_shape[2]:self.valid_image_shape[3]]  # crop invalid pixels
            else:
                ts = cv2.undistort(ts.numpy(), self.camera_matrix, self.distortion_coeffs)
            ts = torch.from_numpy(ts)

        return ts
    
    def convert(self, events):
        self.time_stamps = torch.zeros(self.settings["shape"] + [2, 1], dtype=torch.float32)
        if events['p'].min()<0:
            events['p'] = (events['p'] + 1)/ 2
        event_batch = torch.vstack([(events['t']) * 1e-6,
                                                           events['x'],
                                                           events['y'],
                                                           events['p']]).T
        self.batch_update(event_batch)
        return self.get_ts()

### USAGE: 
# settings = {"shape": ts_shape, "delta_t": model_delta_t}
# ts_gen = TsGenerator(settings=settings, device=device)
# event_batch = torch.from_numpy(np.vstack([(events_t[prev_event_idx:current_event_idx] - events_t[0]) * 1e-6,
#                                                            events['x'][prev_event_idx:current_event_idx],
#                                                            events['y'][prev_event_idx:current_event_idx],
#                                                            events['p'][prev_event_idx:current_event_idx]]).T).to(device)
# if len(event_batch) > 0:
#     ts_gen.batch_update(event_batch)
# ts = ts_gen.get_ts()