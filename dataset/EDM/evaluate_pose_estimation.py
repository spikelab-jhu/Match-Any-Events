import argparse
import cv2
from glob import glob
import h5py
import hdf5plugin
import math
import numpy as np
import os
# from pandas import read_csv
from scipy.spatial.transform import Rotation as R
import time
import torch
from tqdm import tqdm
import yaml
import sys
sys.path.append('./')

from representations import EventFrame, Adaptive_interval, TsGenerator, EventVis
from dataset.prepare_m3ed import get_event_polarity, stack_pos_neg, get_pose_t
from dataset.data_loader import event_norm

from data.util import create_tencode_from_ts
from data_preparation.util import helpers
from models.super_event import SuperEvent, SuperEventFullRes
from models.util import fast_nms
from ts_generation.generate_ts import TsGenerator
from util import visualization
from util.eval_utils import fix_seed
import json

edm_config = {
    'bin': 8,
    'res': [476, 630]
}
def init_edm(device):
    sys.path.append('./third_party/eloftr/')
    from src.config.default import get_cfg_defaults
    from src.lightning.lightning_loftr import PL_LoFTR
    from src.utils.profiler import build_profiler
    config = get_cfg_defaults()

    config.LOFTR.TRAIN_DATA = 'event'
    profiler = build_profiler(None)
    #model = PL_LoFTR(config, pretrained_ckpt='./pretrained/epoch=8-auc@5=0.656-auc@10=0.758-auc@20=0.827.ckpt', profiler=profiler).to(device=device)
    # model = PL_LoFTR(config, pretrained_ckpt='./pretrained/epoch=7-auc@5=0.245-auc@10=0.360-auc@20=0.470.ckpt', profiler=profiler).to(device=device)
    # model = PL_LoFTR(config, pretrained_ckpt='./pretrained/epoch=7-auc@5=0.246-auc@10=0.362-auc@20=0.473.ckpt', profiler=profiler).to(device=device)
    # model = PL_LoFTR(config, pretrained_ckpt='./pretrained/epoch=2-auc@5=0.111-auc@10=0.186-auc@20=0.271.ckpt', profiler=profiler).to(device=device)
    # model = PL_LoFTR(config, pretrained_ckpt='./pretrained/epoch=6-auc@5=0.196-auc@10=0.301-auc@20=0.406.ckpt', profiler=profiler).to(device=device)
    model = PL_LoFTR(config, pretrained_ckpt='./pretrained/epoch=6-auc@5=0.210-auc@10=0.319-auc@20=0.426.ckpt', profiler=profiler).to(device=device)
    return {'model':model}
    

def infer_edm(pack, data, device):
    model = pack['model']
    # model.matcher = model.matcher.eval().half()
    model.eval()
    # data.update({'image':data['data0'].to(device),  'event':data['data1'].to(device)})
    with torch.autocast(enabled=True, dtype=torch.float16, device_type='cuda'):
        model.matcher(data)
    b_mask = data['m_bids'] == 0
    pts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    pts1 = data['mkpts1_f'][b_mask].cpu().numpy()

    return pts0, pts1


def convert_events(event):
    event_voxel = Adaptive_interval((edm_config['bin'], edm_config['res'][0], edm_config['res'][1]), normalize=False, aug=0)
    data_event_pos = event_voxel.convert(get_event_polarity(event, polarity=1))
    event_voxel_neg = Adaptive_interval((edm_config['bin'], edm_config['res'][0], edm_config['res'][1]), normalize=False, aug=0)
    data_event_neg = event_voxel_neg.convert(get_event_polarity(event, polarity=-1))
    event_data = stack_pos_neg(data_event_pos, data_event_neg)
    event_data = event_norm(event_data)

    return event_data

def predict_keypoints(args, config, ts_shape, model, events, poses, start_time, end_time, cropped_shape=None, compiled=False):
    if cropped_shape == None:
        cropped_shape = ts_shape
    settings = {"shape": ts_shape, "delta_t": args.model_delta_t}
    ts_gen = TsGenerator(settings=settings, device=device)

    if args.dataset_name == "ecd":
        events = torch.from_numpy(events).to(device)
    elif args.dataset_name == "eds":
        # Load timestamps into numpy array to speed-up loop
        events_t = np.array(events['t'], dtype=np.int64)

    print("Starting event loop.")
    current_event_idx = 0
    pred_list = []
    ts_vis_list = []

    # Time measurement
    model_inference_time = 0.
    model_inference_iterations = 0
    nms_time = 0.
    nms_iterations = 0

    with torch.inference_mode():
        # print("Warming up network...")
        # for _ in range(100):
        #     example = torch.rand([1, config["input_channels"]] + cropped_shape).to(device)
        #     if compiled:
        #         example = example.to(torch.float16)
        #     model(example)
        # print("Done. Starting evaluation with profiling.")

        for i, pose in enumerate(tqdm(poses)):
            # prev_event_idx = current_event_idx

            # Feed events in batches
            # if args.dataset_name == "ecd":
                
            #     current_event_idx = torch.argmax(torch.clamp(events[:, 0], max=pose[0]))
            #     event_batch = events[prev_event_idx:current_event_idx]
            # elif args.dataset_name == "eds": 

               

            
            # # if len(event_batch) > 0:
            # #     ts_gen.batch_update(event_batch)

            # # Skip if not in experiment time range
            # if pose[0] < start_time:
            #     continue
            # if pose[0] > end_time:
            #     break

            # Experiment
            # ----------
            # Get time surface
            # ts = ts_gen.get_ts()

            # For ablation study: convert to other representations
            # if config["input_representation"] == "ts":
            #     ts = torch.max(ts[..., [2, 7]], dim=-1)[0][..., None]
            # elif config["input_representation"] == "mcts_1":
            #     ts = ts[..., [2, 7]]
            # elif config["input_representation"] == "tencode":
            #     ts = ts.cpu().numpy()
            #     ts = create_tencode_from_ts(ts[..., 4], ts[..., 9], 0.01)  # EventPoint paper reports delta_t=10ms in the experiment section
            #     ts = torch.from_numpy(ts).float().to(device)
            #     ts = ts.permute(1, 2, 0)  # channels last to keep code cleaner (tencode only used for ablation)

            # ts = ts.permute(2, 0, 1).unsqueeze(0)  # channels first
            # ts = ts[..., crop_mask].reshape(list(ts.shape[:-2]) + cropped_shape)
            # if compiled:
            #     ts = ts.to(torch.float16)
            # start_t_measurement = time.time()
            # pred = model(ts)
            # torch.cuda.synchronize()
            # model_inference_time += (time.time() - start_t_measurement)
            # model_inference_iterations += 1

            # # Non-maximum-surpression
            # if compiled:
            #     pred = {"prob": pred[0], "descriptors": pred[1].to(torch.float32)}
            # top_k = np.max(ts.shape) // 2
            # start_t_measurement = time.time()
            # kpts, _ = fast_nms(pred["prob"], config, top_k=top_k)
            # torch.cuda.synchronize()
            # nms_time += (time.time() - start_t_measurement)
            # nms_iterations += 1

            # # Extract descriptors
            # desc = pred["descriptors"][0, :, kpts[0][:, 0], kpts[0][:, 1]].permute(1, 0).cpu().detach().numpy()
            # kpts = kpts[0].cpu().detach().numpy()

            # # Move kpts to original positions
            # kpts[:, 0] += math.ceil(crop[0] / 2)
            # kpts[:, 1] += math.ceil(crop[1] / 2)

            # Save keypoints and descriptors in list
            pred_list.append({"t": pose[0], "index":i,# "kpts": kpts, "desc": desc, 
                              "gt_t": pose[1:4], "gt_rot": pose[4:8]})
            # if args.visualize:
            #     # Save for visualization
            #     ts_vis_list.append(cv2.cvtColor(visualization.ts2image(ts[0].detach().cpu().numpy().transpose(1,2,0)), cv2.COLOR_BGR2GRAY))

    return pred_list, ts_vis_list, model_inference_time, model_inference_iterations, nms_time, nms_iterations

def process_tracked_keypoints(kpts, num_keypoints_per_step, poses, start_time, end_time, max_eval_delta_t):
    pred_list = []
    for pose in tqdm(poses):
        # Skip if not in experiment time range
        if pose[0] < start_time:
            continue
        if pose[0] > end_time:
            break

        # Find relevant keypoints (that were last tracked)
        t = pose[0]
        kpts_step = kpts[kpts[:, 1] <= t][::-1]
        if kpts_step[0, 1] < t - max_eval_delta_t:  # Tracking failure (newest track is too long ago)
            pred_list.append({"t": pose[0], "kpts": np.array([[], []]).T,
                              "ids": [], "gt_t": pose[1:4], "gt_rot": pose[4:8]})
            continue
        _, idx = np.unique(kpts_step[:, 0], return_index=True)
        kpts_step = kpts_step[np.sort(idx)[:num_keypoints_per_step]]

        # Save keypoints and descriptors in list
        pred_list.append({"t": pose[0], "kpts": np.stack([kpts_step[:, 3], kpts_step[:, 2]], axis=1),  # Keypoints are transposed
                          "ids": kpts_step[:, 0], "gt_t": pose[1:4], "gt_rot": pose[4:8]})
        
    return pred_list

def calculate_pose_estimation_error(pred_list, deg_intervals, max_delta_t, ransac_threshold=3.0, events = None):
    pack = init_edm(device)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    results = {"r_error": [], "inlier_ratio": [], "gt_rot": [], "dt": []}
    skip_until_step = 0
    evalutation_steps = []
    for step in tqdm(range(len(pred_list))):
        # Skip steps that were already used for evaluation
        if skip_until_step > step:
            continue

        # Make sure maximal required rotation change is achieved in the maximal number of steps
        gt_rot_list = []
        abs_gt_rot_deg_list = []
        evaluate_at_these_steps = False
        for matching_step in range(step + 1, len(pred_list)):
            if pred_list[matching_step]["t"] - pred_list[step]["t"] > max_delta_t:
                # Too much time elapsed
                break

            # Calculate groundtruth rotation change
            gt_rot = R.from_quat(pred_list[matching_step]["gt_rot"]).inv() * R.from_quat(pred_list[step]["gt_rot"])
            abs_gt_rot_deg = np.linalg.norm(gt_rot.as_rotvec(degrees=True))
            gt_rot_list.append(gt_rot)
            abs_gt_rot_deg_list.append(abs_gt_rot_deg)

            if abs_gt_rot_deg > np.max(deg_intervals):
                # Enough rotation change found
                evaluate_at_these_steps = True
                break

        if not evaluate_at_these_steps:
            continue

        last_step = step + len(gt_rot_list)
        interval_idx = 0
        events_t = events['t']
        skip_until_step = last_step + 1
        for matching_step in range(step + 1, last_step + 1):
            gt_rot_idx = matching_step-step-1

            if abs_gt_rot_deg_list[gt_rot_idx] >= deg_intervals[interval_idx]:
                interval_idx += 1
                evalutation_steps.append([pred_list[step]['index'], pred_list[matching_step]['index']])
                # Match
                time_src = pred_list[step]['t']
                time_dst = pred_list[matching_step]['t']

                src_prev_event_idx = np.argmax(events_t > time_src * 1e6)
                src_current_event_idx = np.argmax(events_t > time_src * 1e6 + 128*1e3)

                tgt_prev_event_idx = np.argmax(events_t > time_src * 1e6)
                tgt_current_event_idx = np.argmax(events_t > time_src * 1e6 + 128*1e3)
                

                # event_batch = torch.from_numpy(np.vstack([(events_t[prev_event_idx:current_event_idx] - events_t[0]) * 1e-6,
                #                                            events['x'][prev_event_idx:current_event_idx],
                #                                            events['y'][prev_event_idx:current_event_idx],
                #                                            events['p'][prev_event_idx:current_event_idx]]).T).to(device)
                event_src = {'x':torch.from_numpy(events['x'][src_prev_event_idx:src_current_event_idx-1]).to(torch.float32),
                'y': torch.from_numpy(events['y'][src_prev_event_idx:src_current_event_idx-1]).to(torch.float32),
                't': torch.from_numpy(events['t'][src_prev_event_idx:src_current_event_idx-1]).to(torch.float32),
                'p': torch.from_numpy(events['p'][src_prev_event_idx:src_current_event_idx-1]*2-1).to(torch.float32)
                }

                event_dst = {'x':torch.from_numpy(events['x'][tgt_prev_event_idx:tgt_current_event_idx-1]).to(torch.float32),
                'y': torch.from_numpy(events['y'][tgt_prev_event_idx:tgt_current_event_idx-1]).to(torch.float32),
                't': torch.from_numpy(events['t'][tgt_prev_event_idx:tgt_current_event_idx-1]).to(torch.float32),
                'p': torch.from_numpy(events['p'][tgt_prev_event_idx:tgt_current_event_idx-1]*2-1).to(torch.float32)
                }
                data = {'image':convert_events(event_src),
                        'event':convert_events(event_dst)
                }
                pts0, pts1 = infer_edm(pack,data,device)



                # if "desc" in pred_list[0].keys():
                #     matches = bf.match(pred_list[step]["desc"], pred_list[matching_step]["desc"])
                #     matched_kpts = np.array([[pred_list[step]["kpts"][m.queryIdx], pred_list[matching_step]["kpts"][m.trainIdx]] for m in matches]).astype(np.float32)
                # elif "ids" in pred_list[0].keys():
                #     _, idxs_step, idxs_step_dist = np.intersect1d(pred_list[step]["ids"], pred_list[matching_step]["ids"], return_indices=True)
                #     matched_kpts = np.array([pred_list[step]["kpts"][idxs_step], pred_list[matching_step]["kpts"][idxs_step_dist]]).transpose(1, 0, 2)

                #     if args.visualize:
                #         matches = [cv2.DMatch(idxs_step[i], idxs_step_dist[i], 0, 0) for i in range(len(matched_kpts))]
                # else:
                #     raise ValueError("Either descriptors or track id must be specified in predictions.")

                # Estimate pose
                # matched_kpts = np.roll(matched_kpts, 1, axis=-1)  # Prediction is (row, column) instead of (x, y)

                try:
                    num_inliers, E, est_R, est_t, ransac_mask = cv2.recoverPose(points1=pts0.astype(np.float64), points2=pts1.astype(np.float64), 
                                                                                cameraMatrix1=camera_matrix, cameraMatrix2=camera_matrix,
                                                                                distCoeffs1=distortion_coeffs, distCoeffs2=distortion_coeffs,
                                                                                method=cv2.RANSAC, threshold=ransac_threshold)
                except cv2.error:
                    num_inliers = 0

                # Calculate error
                if num_inliers > 0:
                    est_rot = R.from_matrix(est_R)
                    rot_error = est_rot * gt_rot_list[gt_rot_idx].inv()
                    error_angle_deg = np.linalg.norm(rot_error.as_rotvec(degrees=True))

                else:  # Recovering pose failed
                    error_angle_deg = 180  # max error

                results["r_error"].append(error_angle_deg)
                results["inlier_ratio"].append(num_inliers / max(len(matched_kpts), 1))  # prevent devision by zero
                results["gt_rot"].append(abs_gt_rot_deg_list[gt_rot_idx])
                results["dt"].append(pred_list[matching_step]["t"] - pred_list[step]["t"])

                if args.visualize:
                    print("Groundtruth rotvec:", gt_rot_list[gt_rot_idx].as_rotvec(degrees=True))
                    if num_inliers > 0:
                        print("Estimated rotvec:", est_rot.as_rotvec(degrees=True))
                        print("Inlier ratio:", num_inliers / max(len(matched_kpts), 1))
                        print("Error in degrees:", error_angle_deg)
                    else:
                        print("Pose estimation failed!")

                    kpts_cv2_0 = [cv2.KeyPoint(float(pred_list[step]["kpts"][i, 1]), float(pred_list[step]["kpts"][i, 0]), 1) for i in range(len(pred_list[step]["kpts"]))]
                    kpts_cv2_1 = [cv2.KeyPoint(float(pred_list[matching_step]["kpts"][i, 1]), float(pred_list[matching_step]["kpts"][i, 0]), 1) for i in range(len(pred_list[matching_step]["kpts"]))]

                    matched_img = cv2.drawMatches(ts_vis_list[step], kpts_cv2_0, ts_vis_list[matching_step], kpts_cv2_1, matches,
                                                None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255))

                    cv2.imshow(f"SuperEvent matches", matched_img)
                    cv2.waitKey(0)

    return results, evalutation_steps

def calculate_auc(results, auc_threshold_list):
    # Create precision-recall curve
    sorted_idxs = np.argsort(results["r_error"])
    results["r_error"] = [0] + np.array(results["r_error"])[sorted_idxs].tolist()
    results["inlier_ratio"] = [0] + np.array(results["inlier_ratio"])[sorted_idxs].tolist()
    results["gt_rot"] = [0] + np.array(results["gt_rot"])[sorted_idxs].tolist()

    # Taken from LoFTR:
    # https://github.com/zju3dv/LoFTR/blob/df7ca80f917334b94cfbe32cc2901e09a80e70a8/src/utils/metrics.py#L148C2-L154C42
    aucs = []
    recall = list(np.linspace(0, 1, len(results["r_error"])))
    for thr in auc_threshold_list:
        last_index = np.searchsorted(results["r_error"], thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = results["r_error"][:last_index] + [thr]
        aucs.append(np.trapezoid(y, x) / thr)
    print(f"AUCs for {auc_threshold_list}: {aucs}.")

    return aucs


# Fix seed for reproducibility
fix_seed()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="Root directory of dataset")
parser.add_argument("--dataset_name", default="", help="Dataset used for evaluation. Supported datasets are Event Camera Dataset ('ecd') and Event-aided Direct Sparse Odometry ('eds').")
parser.add_argument("--sequence_names", nargs="*", default=None, help="Names of evaluation sequences")
parser.add_argument("--config", default="config/super_event.yaml", help="Parameter configuration.")
parser.add_argument("--model", default="", help="Model weights to be evaluated. If not specified, the most recent weights in saved_models/ are used.")
parser.add_argument("--model_delta_t", nargs="*", default=[0.001, 0.003, 0.01, 0.03, 0.1], type=float, help="Time delta of time surfaces")
parser.add_argument("--max_eval_delta_t", nargs="*", default=2.0, type=float, help="List of durations between testing the repeatability")
parser.add_argument("--auc_thresholds",  nargs="*", default=[5.0, 10.0, 20.0], help="Thresholds for AUC evaluation.")
parser.add_argument('--visualize', default=False, action=argparse.BooleanOptionalAction, help="Visualize matches, only for debugging")
parser.add_argument("--kpts_dir", default="", help="Directory where keypoints are loaded from. If not specified (default), SuperEvent will predict the keypoints.")
parser.add_argument("--num_kpts_per_step", default=-1, help="Number of keypoints tracked in parallel. Only used when 'kpts_dir' is specified. -1 to infer (default).")
parser.add_argument('--compile', default=False, action=argparse.BooleanOptionalAction, help="torch.compile model before inference.")
args = parser.parse_args()

if not args.dataset_name:
    args.dataset_name = os.path.basename(os.path.normpath(args.dataset_path))
    print(f"Evaluating on dataset {args.dataset_name}.")

supported_datasets = ["ecd", "eds"]
assert args.dataset_name in supported_datasets, f"Datset {args.dataset_name} not supported. Please use one of {supported_datasets}."

if not args.sequence_names:  # Use default sequences
    if args.dataset_name == "ecd":
        args.sequence_names = ["boxes_6dof", "boxes_rotation", "poster_6dof", "poster_rotation", "shapes_6dof", "shapes_rotation"]
    elif args.dataset_name == "eds":
        args.sequence_names = ["00_peanuts_dark", "01_peanuts_light", "02_rocket_earth_light", "03_rocket_earth_dark", "06_ziggy_and_fuzz", "07_ziggy_and_fuzz_hdr", "08_peanuts_running", "11_all_characters"]

if args.dataset_name == "ecd":
    ts_shape = [180, 240]
    pose_file_name = "groundtruth.txt"
    events_file_name = "events.txt"
    image_timestampes_file_name = "images.txt"
    ransac_threshold = 1.0
elif args.dataset_name == "eds":
    ts_shape = [480, 640]
    pose_file_name = "stamped_groundtruth.txt"
    events_file_name = "events.h5"
    image_timestampes_file_name = "images_timestamps.txt"
    ransac_threshold = 3.0

# Time measurement
total_model_inference_time = 0.
total_model_inference_iterations = 0
total_nms_time = 0.
total_nms_iterations = 0

if args.kpts_dir:
    experiment_name = "pose_estimation_" + args.dataset_name + "_" + os.path.basename(os.path.normpath(args.kpts_dir)) + "_at_time_" + time.strftime("%Y%m%d-%H%M%S")
else:
    experiment_name = "pose_estimation_" + args.dataset_name + "_" + os.path.splitext(os.path.basename(args.model))[0] + "_at_time_" + time.strftime("%Y%m%d-%H%M%S")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        print("Loaded config from", args.config)
    if "backbone" in config:
        # Load backbone config and add
        backbone_config_path = os.path.join(os.path.dirname(args.config), "backbones", config["backbone"] + ".yaml")
        if os.path.exists(backbone_config_path):
            with open(backbone_config_path, "r") as f:
                backbone_config = yaml.safe_load(f)
                print("Loaded backbone config from", backbone_config_path)
            config = config | backbone_config
            config["backbone_config"]["input_channels"] = config["input_channels"]
        else:
            print("No additional config file found for backbone", config["backbone"])

    # Calculate required shape for model inference
    max_factor_required = config["grid_size"]
    if "backbone_config" in config:
        max_factor_required = 2 ** (len(config["backbone_config"]["num_blocks"]) - 1) * \
                                config["backbone_config"]["stem"]["patch_size"] * \
                                np.max(config["backbone_config"]["stage"]["attention"]["partition_size"])
    crop = np.array(ts_shape) % max_factor_required

    crop_mask = torch.ones(ts_shape, dtype=bool)
    crop_mask[:math.ceil(crop[0] / 2)] = False
    crop_mask[:, :math.ceil(crop[1] / 2)] = False
    if crop[0] > 1:
        crop_mask[-math.floor(crop[0] / 2):] = False
    if crop[1] > 1:
        crop_mask[:, -math.floor(crop[1] / 2):] = False
    cropped_shape = [ts_shape[0] - crop[0], ts_shape[1]- crop[1]]

    # Load model
    if args.model == "":
        # Use most recent model in saved_models
        list_of_files = glob("saved_models/*.pth")
        args.model = max(list_of_files, key=os.path.getctime)

    if config["pixel_wise_predictions"]:
        model = SuperEventFullRes(config, tracing=args.compile)
    else:
        model = SuperEvent(config, tracing=args.compile)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.to(device)
    model.eval()
    print("Loaded model weights from", args.model)

    if args.compile:
        model = torch.compile(model.to(torch.float16), backend="inductor", mode="max-autotune")
        print("Compiled model")

calib_path = os.path.join(args.dataset_path, "calib.txt")
calib = np.genfromtxt(calib_path)
print(f"Loaded calibration from {calib_path}.")
camera_matrix, distortion_coeffs = helpers.get_camera_matrix_and_distortion_coeffs(calib)

results = {"r_error": [], "inlier_ratio": [], "gt_rot": [], "dt": []}
deg_intervals = np.array(range(0, 45, 1)) + 1
evaluation_indexs = []
for sequence in args.sequence_names:
    # Load sequence data
    sequence_path = os.path.join(args.dataset_path, sequence)
    poses = np.genfromtxt(os.path.join(sequence_path, pose_file_name))
    print(f"Loaded poses for sequence {sequence}.")

    if args.kpts_dir:
        if args.dataset_name == "eds":
            event_t = h5py.File(os.path.join(sequence_path, events_file_name), 'r')["t"]
            first_event_time = event_t[0] * 1e-6
            last_event_time = event_t[-1] * 1e-6 - first_event_time
            poses[:, 0] -= first_event_time
        else: 
            # Avoid loading .txt, none of the approaches failed for this dataset
            last_event_time = np.inf
        
        baseline_name = os.path.basename(os.path.normpath(args.kpts_dir))
        if baseline_name == "RATE":
            distortion_coeffs = np.zeros(4)  # Haste (tracker in RATE) performs undistortion
            keypoints_path = os.path.join(args.kpts_dir, "tracks_" + sequence + ".txt")
            kpts = np.genfromtxt(keypoints_path)
        elif baseline_name == "LLAK":
            keypoints_path = os.path.join(args.kpts_dir, "tracks_" + sequence + ".csv")
            kpts = np.genfromtxt(keypoints_path, delimiter=',', skip_header=1)
            
            # Match format of RATE (id, t, x, y)
            kpts = np.vstack([kpts[:, 3], kpts[:, 2], kpts[:, 0], kpts[:, 1]]).T
            kpts[:, 1] *= 1e-6
        print(f"Loaded tracks of {np.max(kpts[:, 0])} keypoints from {keypoints_path}.")
        
        start_time = max(poses[0, 0], kpts[np.argmax(kpts[:, 0] >= args.num_kpts_per_step), 1])
        end_time = min(poses[-1, 0], last_event_time)
        if args.num_kpts_per_step < 0:
            # Infer predicted keypoints per time step
            pose_frequency = (poses[-1, 0] - poses[0, 0]) / len(poses)
            num_kpts_per_step = max(math.ceil(len(kpts) / (end_time - start_time) * pose_frequency), ts_shape[0] // 10)
            print(f"Assuming the latest {num_kpts_per_step} keypoint tracks to be active.")

        pred_list = process_tracked_keypoints(kpts, num_kpts_per_step, poses, start_time, end_time, max_eval_delta_t=args.max_eval_delta_t)

        if args.visualize:
            ts_vis_list = len(pred_list) * [np.zeros(ts_shape, dtype=np.uint8)]
    else:
        # Get keypoints with 
        if args.dataset_name == "ecd":
            events = read_csv(os.path.join(sequence_path, events_file_name), header='infer', delimiter=" ", usecols=range(4)).to_numpy()
            start_time = poses[np.argmax(poses[:, 0] > events[0, 0].item() + np.max(args.model_delta_t)), 0]
            end_time = min(events[-1, 0].item(), poses[-1, 0])
        elif args.dataset_name == "eds":
            events = h5py.File(os.path.join(sequence_path, events_file_name), 'r')
            os.path.join(os.path.join(sequence_path, events_file_name))
            start_time = poses[np.argmax(poses[:, 0] > events['t'][0] * 1e-6 + np.max(args.model_delta_t)), 0]
            end_time = min(events['t'][-1] * 1e-6, poses[-1, 0])
        print(f"Loaded events for sequence {sequence}.")
        pred_list, ts_vis_list, model_inference_time, model_inference_iterations, nms_time, nms_iterations = \
            predict_keypoints(args, config, ts_shape, model, events, poses, start_time, end_time, 
                              cropped_shape=cropped_shape, compiled=args.compile)
        
        total_model_inference_time += model_inference_time
        total_model_inference_iterations += model_inference_iterations
        total_nms_time += nms_time
        total_nms_iterations += nms_iterations

    # Evaluate predictions
    sequence_results, matching_steps = calculate_pose_estimation_error(pred_list, deg_intervals, args.max_eval_delta_t, ransac_threshold=ransac_threshold)

    evaluation_indexs.append({'name':sequence,'indexs':matching_steps})
    for key in results.keys():
        results[key].extend(sequence_results[key])
with open("./edc_indices.json", "w") as f:
    json.dump(evaluation_indexs, f, indent=4)
# Calculate auc
aucs = calculate_auc(results, args.auc_thresholds)

improved_ratio = np.count_nonzero(np.array(results["r_error"]) < np.array(results["gt_rot"])) / len(results["gt_rot"])
print(f"Ratio of improved poses: {improved_ratio}")
# print(f"Average inlier ratio: {np.mean(results["inlier_ratio"])}")
print(f"Average inference time of model: {total_model_inference_time / total_model_inference_iterations}")
print(f"Average runtime time of NMS: {total_nms_time / total_nms_iterations}")