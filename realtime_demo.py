import torch.multiprocessing as mp
import threading
from collections import deque
import numpy as np
import time
import queue
import argparse

import os
import sys
sys.path.append("/usr/lib/python3/dist-packages/")

import cv2
from os import path
import metavision_hal
from metavision_core.event_io.raw_reader import RawReader
from metavision_core.event_io.py_reader import EventDatReader
from metavision_hal import I_LL_Biases
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_core.event_io import EventsIterator, DatWriter
from metavision_hal import DeviceDiscovery
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_base import EventCDBuffer
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, PolarityFilterAlgorithm, RoiFilterAlgorithm

import torch

from representations import EventFrame, Adaptive_interval, TsGenerator, EventVis
from dataset.data_loader import event_norm

from config.default import get_cfg_defaults
from lightning.lightning_model import PL_MatchAnyEvents
from utils.profiler import build_profiler

device = torch.device('cuda')

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

default_setting = {
    'stc_cut_trail' : True,  
    'roi_x0' : 0,
    'roi_y0' : 0,
    'roi_x1' : 640,
    'roi_y1' : 480,
    'output_path':r"./dataset",
    'output_file' : r"EventRaw.dat",
    'custom_bias': False,
    'bias': {
        "bias_diff":0,
        "bias_diff_off":0,
        "bias_diff_on":0,
        "bias_fo":0,
        "bias_hpf":0,
        "bias_refr":0
    },
    'input_filename' : None, 
    'serial' : "",
    'trigger_out': {
        'enable': False
    },
    'trigger_in': {
        'enable': True,
        'channel':0,
    }
}

stc_cut_trail = default_setting['stc_cut_trail']
nameoutglob = 1

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


def save_trigger(triggers,dir):
    with open(os.path.join(default_setting["output_path"],dir,"event","trigger.txt"), "w+") as f:
        for i in range(0,len(triggers)):
            f.write('{}\n'.format(triggers[i][1]))
        f.close()


def get_event_triggers(raw_path, polarity: int = -1, do_time_shifting=True):
    triggers = None
    with RawReader(str(raw_path), do_time_shifting=do_time_shifting) as ev_data:
        while not ev_data.is_done():
            a = ev_data.load_n_events(1000000)
        triggers = ev_data.get_ext_trigger_events()

    if polarity in (0, 1):
        triggers = triggers[triggers['p'] == polarity].copy()
    else:
        triggers = triggers.copy()
    return triggers


def to_tensor(e, num_bin=8, src_res = (480,640), device='cuda'):
    t_normalized = 1.0 * ( e['t'] - e['t'][0]) / 1e3
    event = {
        'x': torch.from_numpy(e['x'].astype(np.float32)).to(device=device, non_blocking=True),
        'y': torch.from_numpy(e['y'].astype(np.float32)).to(device=device, non_blocking=True),
        't': torch.from_numpy(t_normalized.astype(np.float32)).to(device=device,non_blocking=True),
        'p': torch.from_numpy(e['p'].astype(np.float32)).to(device=device,non_blocking=True) * 2.0 - 1.0,
    }
    event_voxel = Adaptive_interval((num_bin, src_res[0], src_res[1]), normalize=False, aug=0)
    data_event_pos = event_voxel.convert(get_event_polarity(event, polarity=1))
    event_voxel_neg = Adaptive_interval((num_bin, src_res[0], src_res[1]), normalize=False, aug=0)
    data_event_neg = event_voxel_neg.convert(get_event_polarity(event, polarity=-1))
    event_data = stack_pos_neg(data_event_pos, data_event_neg)
    event_data = event_norm(event_data)
    # print(event_data.device)
    return event_data

def init_edm(device):
    config = get_cfg_defaults()
    profiler = build_profiler(None)
    model = PL_MatchAnyEvents(config, pretrained_ckpt='./pretrained/pretrained_v1.pth', profiler=profiler).to(device=device) 

    return model

class VisualizationProcess(mp.Process):
    def __init__(self, result_queue, display_queue):
        super().__init__()
        self.result_queue = result_queue
        self.display_queue = display_queue
        self.running = mp.Event()
        self.running.set()

    def run(self):
        prev_time = time.perf_counter()
        fps = 0.0
        fps_smoothing = 0.1
        while self.running.is_set():
            try:
                # Wait for inference results
                pts0, pts1, vis1, vis2 = self.result_queue.get(timeout=0.1)
                
                if vis1 is None or vis2 is None:
                    continue

                vis1_crop = vis1[65:415, 5:635]
                vis2_crop = vis2[65:415, 5:635]

                h1, w1 = vis1_crop.shape[:2]
                combined = cv2.hconcat([vis1_crop, vis2_crop])

                # Only attempt to draw if we actually found matches
                if len(pts0) > 0:
                    x_norm = pts0[:, 0] / w1
                    y_norm = pts0[:, 1] / h1
                    
                    v = (x_norm + y_norm) / 2.0
                    v_255 = np.clip((v * 255), 0, 255).astype(np.uint8)
                    colors = cv2.applyColorMap(v_255, cv2.COLORMAP_HSV).reshape(-1, 3)

                    for p0, p1, color in zip(pts0, pts1, colors):
                        x0 = int(p0[0])
                        y0 = int(p0[1])
                        
                        x1 = int(p1[0]) + w1 
                        y1 = int(p1[1])
                        c = color.tolist()

                        cv2.circle(combined, (x0, y0), 3, c, -1)
                        cv2.circle(combined, (x1, y1), 3, c, -1)
                        cv2.line(combined, (x0, y0), (x1, y1), c, 1)
                    # --- FPS Calculation ---
                current_time = time.perf_counter()
                elapsed = current_time - prev_time
                prev_time = current_time

                if elapsed > 0:
                    current_fps = 1.0 / elapsed
                    # Apply Exponential Moving Average
                    fps = (fps_smoothing * current_fps) + ((1 - fps_smoothing) * fps)

                # Draw the FPS on the top-left corner of the combined image
                cv2.putText(
                    combined, 
                    f"Vis FPS: {fps:.1f}", 
                    (10, 30), # (x, y) coordinates
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0,      # Font scale
                    (0, 255, 0), # Color (B, G, R) - Green
                    2         # Thickness
                )
                try:
                    self.display_queue.put_nowait(combined)
                except queue.Full:
                    pass 

            except queue.Empty:
                continue

    def stop(self):
        self.running.clear()


class InferenceProcess(mp.Process):
    def __init__(self, input_queue1, input_queue2, output_queue, target_device):
        super().__init__()
        self.input_queue1 = input_queue1
        self.input_queue2 = input_queue2
        self.output_queue = output_queue
        self.target_device = target_device
        self.running = mp.Event()
        self.running.set()


    def run(self):
        print("Initializing Model inside Inference Process...")
        torch.backends.cudnn.benchmark = True
        model = init_edm(self.target_device)
        model.eval()

        # Create a background stream for the heavy to_tensor op
        prep_stream = torch.cuda.Stream(device=self.target_device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        try:
            events1, vis1 = self.input_queue1.get(timeout=5.0)
            events2, vis2 = self.input_queue2.get(timeout=5.0)
            

            with torch.cuda.stream(prep_stream):
                t_e1 = to_tensor(events1, device=self.target_device)[:, 65:415, 5:635]
                t_e2 = to_tensor(events2, device=self.target_device)[:, 65:415, 5:635]
        except queue.Empty:
            print("Failed to get initial frame.")
            return


        while self.running.is_set():
            t0 = time.perf_counter()

            # Wait for the background prep stream to finish processing the CURRENT frame
            torch.cuda.current_stream().wait_stream(prep_stream)

            data = {
                'image': t_e1.unsqueeze(0),
                'event': t_e2.unsqueeze(0),
            }

            t1 = time.perf_counter()
            start_event.record()

            with torch.inference_mode(), torch.autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
                model.matcher(data)
            end_event.record()

            try:
                next_events1, next_vis1 = self.input_queue1.get_nowait()
                next_events2, next_vis2 = self.input_queue2.get_nowait()
                
                with torch.cuda.stream(prep_stream):
                    next_t_e1 = to_tensor(next_events1, device=self.target_device)[:, 65:415, 5:635]
                    next_t_e2 = to_tensor(next_events2, device=self.target_device)[:, 65:415, 5:635]
                
                has_next = True
            except queue.Empty:
                has_next = False

            combined_pts = torch.stack([data['mkpts0_f'], data['mkpts1_f']])
            combined_cpu = combined_pts.cpu().numpy()

            pts0 = combined_cpu[0]
            pts1 = combined_cpu[1]

            inference_time_ms = start_event.elapsed_time(end_event)
                
            print(f"Model Fwd Time: {inference_time_ms:.2f} ms")

            try:
                self.output_queue.put_nowait([pts0, pts1, vis1, vis2])
            except queue.Full:
                pass 


            if has_next:
                t_e1, t_e2 = next_t_e1, next_t_e2
                vis1, vis2 = next_vis1, next_vis2
            else:
                # If queue was empty, wait synchronously for next data
                try:
                    events1, vis1 = self.input_queue1.get_nowait()
                    events2, vis2 = self.input_queue2.get_nowait()
                    
                    # events1 = events1[:, 65:415, 5:635]
                    # events2 = events2[:, 65:415, 5:635]

                    t_e1 = to_tensor(events1, device=self.target_device)[:, 65:415, 5:635]
                    t_e2 = to_tensor(events2, device=self.target_device)[:, 65:415, 5:635]
                except queue.Empty:
                    continue
    def stop(self):
        self.running.clear()

class CameraProcess(mp.Process):
    def __init__(self, serial_number, output_queue, buffer_time_ms=64):
        super().__init__()
        self.serial = serial_number
        self.output_queue = output_queue # mp.Queue to send data back to main
        self.buffer_time_us = buffer_time_ms * 1000 
        self.stop_event = mp.Event()
        self.running = mp.Event()
        self.running.set()
        

    def run(self):
        init = False
        
        while not init:
            print(f"Initializing device [Cam {self.serial}]")
            try:
                device = initiate_device(path=self.serial)
                init = True
            except:
                pass
        if not device:
            print(f"Could not open camera with serial: {self.serial}")
            return
        print(f"[Cam {self.serial}] ✅ Camera opened successfully!")

        mv_iterator = EventsIterator.from_device(device=device)
        height, width = mv_iterator.get_size()
        
        event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=width, sensor_height=height, fps=15, palette=ColorPalette.Dark
        )

        # Local state (No locks needed in a separate process!)
        event_buffer = deque()
        
        def on_cd_frame_cb(ts, cd_frame):
            if not event_buffer:
                recent_events = np.empty((0,), dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i2'), ('t', '<i8')])
            else:
                combined_evs = np.concatenate(list(event_buffer))
                time_limit = ts - self.buffer_time_us
                start_idx = np.searchsorted(combined_evs['t'], time_limit)
                recent_events = combined_evs[start_idx:]

            data = (recent_events, cd_frame.copy())

            try:
                self.output_queue.put_nowait(data)
                
            except queue.Full:
                    
                pass

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Main Camera Loop
        while self.running.is_set():
            for evs in mv_iterator:
                if self.stop_event.is_set():
                    break
                    
                if evs.size > 0:
                    # print(f"[Cam {self.serial}] 🟢 First events received! Iterator is working.")
                    latest_time = evs['t'][-1] 
                    event_buffer.append(evs)
                    
                    # Prune old events
                    while event_buffer:
                        oldest_chunk = event_buffer[0]
                        if latest_time - oldest_chunk['t'][-1] > self.buffer_time_us:
                            event_buffer.popleft()
                        else:
                            break
                            
                    event_frame_gen.process_events(evs)

    def stop(self):
        self.stop_event.set()


def main(serial_cam_1, serial_cam_2):

    # Use multiprocessing Queues
    inference_queue1 = mp.Queue(maxsize=3)
    inference_queue2 = mp.Queue(maxsize=3)
    result_queue = mp.Queue(maxsize=2)
    display_queue = mp.Queue(maxsize=2) 

    print("Starting Camera Threads...")

    # Initialize processes
    cam1 = CameraProcess(serial_number=serial_cam_1, output_queue=inference_queue1)
    cam2 = CameraProcess(serial_number=serial_cam_2, output_queue=inference_queue2)

    
    cameras = [cam1, cam2]

    print("Starting Inference & Visualization Processes...")
    model_process = InferenceProcess(input_queue1=inference_queue1, 
                                     input_queue2=inference_queue2, 
                                     output_queue=result_queue,
                                     target_device=device)
                                   
    vis_process = VisualizationProcess(result_queue=result_queue, display_queue=display_queue)
    
    cam1.start()
    cam2.start()
    model_process.start()
    vis_process.start()

    print("\n>>> System running. Press 'ESC' on the visualizer window or 'Ctrl+C' in terminal to stop. <<<\n")

    loop_counter = 0

    try:
        # --- MAIN LOOP ---
        while True:
            start_time = time.perf_counter()
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  
                print("ESC pressed. Initiating graceful shutdown...")
                break
            
            loop_counter += 1
            
            # Draw live streams to CV2 windows
            # for cam in cameras:
            #     frame = cam.get_latest_frame()
            #     if frame is not None:
            #         cv2.imshow(f"Live Event Stream - Cam {cam.serial}", frame)
            #         frames_drawn = True
                    
            try:
                matches_frame = display_queue.get_nowait()
                cv2.imshow("MatchAnyEvents", matches_frame)
                frames_drawn = True
            except queue.Empty:
                pass

            
            # GUI refresh & Escape condition
            # if loop_counter % 100 == 0:
            #     print(f"cv wait ({(sensor_time-start_time)*1000:.1f} ms)| draw ({(draw_time-sensor_time)*1000:.1f} ms)")
            
            #     print(f"Queue Sizes - In1: {inference_queue1.qsize()} | In2: {inference_queue2.qsize()} | Res: {result_queue.qsize()} | Disp: {display_queue.qsize()}")
            
    except KeyboardInterrupt:
        print("\nCtrl+C Detected. Initiating graceful shutdown...")

    finally:
        # --- TEARDOWN ---
        print("Stopping cameras...")
        for cam in cameras:
            cam.stop()
        
        print("Stopping processes...")
        model_process.stop()
        vis_process.stop() 
        
        for cam in cameras:
            cam.join()
        model_process.join()
        vis_process.join() 
        
        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Input two serial numbers.")
    parser.add_argument('--s1', type=str, required=True, help="The first camera serial number")
    parser.add_argument('--s2', type=str, required=True, help="The second camera serial number")
    
    args = parser.parse_args()
    
    serial_1 = args.s1
    serial_2 = args.s2
    print(f"Serial Number 1: {serial_1}")
    print(f"Serial Number 2: {serial_2}")
    
    # This is STRICTLY REQUIRED for PyTorch CUDA multiprocessing to work correctly.
    mp.set_start_method('spawn', force=True)
    main(serial_1, serial_2)