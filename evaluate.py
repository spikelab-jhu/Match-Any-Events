# Evalutation code for MatchAnyEvent, error_auc, estimate_pose, relative_pose_error are modified from eloftr: https://github.com/zju3dv/EfficientLoFTR
import argparse
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import sys
import yaml
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataset.EDM.edm_dataset import EDMDataset, M3EDDataset, EDSDataset, EMegaDataset, CustomDataset

from kornia.geometry.epipolar import numeric
from kornia.geometry.conversions import convert_points_to_homogeneous
from utils.plotting import dynamic_alpha, error_colormap, make_matching_figure

from PIL import Image
from torch.utils.data import Subset
from tqdm import tqdm

EDM_CONFIG = {
    'resolution': [720, 1280],
    'val_res': [350, 630],
    'interval_ms': 128,
    'num_bin':8,
    'repres': ['event_stack', 'mcts', 'event_frame', 'reconstruction', 'image'],
    'min_overlap_score': 0.0,
    'epipolar_threshold':1e-4
}

def _make_evaluation_figure(img0, img1, kpts0, kpts1, epi_errs, r_error, t_error, alpha='dynamic'):
    conf_thr = EDM_CONFIG['epipolar_threshold']
    
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    # n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu())
    # recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        # f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'R error: {r_error:.1f}, t error: { t_error:.1f}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text, path='./visualization/100.png')
    return figure

def _make_evaluation_figure_green(img0, img1, kpts0, kpts1, epi_errs, r_error, t_error, alpha='dynamic', index = None):
    conf_thr = EDM_CONFIG['epipolar_threshold']
    alpha = 1.0

    color = np.stack([np.zeros(len(kpts0)), np.ones(len(kpts0)), np.zeros(len(kpts0)), np.ones(len(kpts0))*alpha], -1)
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=[], path='./paperwriting/results%050d.png'%index)
    return figure

def plot_matched_points(image1, image2, kpts0, kpts1, inliers):

    bs = image2.shape[0]
    num_points = len(kpts0)  # Number of matched pairs

    

    # Generate colors using the Jet colormap
    colormap = plt.cm.jet(np.linspace(0, 1, num_points))
    jet = plt.cm.jet

    dark_jet = mcolors.LinearSegmentedColormap.from_list(
    'dark_jet', jet(np.linspace(0, 1, num_points)) * [0.7, 0.7, 0.7, 1]
)(np.linspace(0, 1, num_points))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(16, 16))
    axes = axes.ravel()

    for i in range(1):
        # Plot Image 1 with matched points
        axes[2*i].imshow(image1.transpose(1,2,0))
        axes[2*i].set_title("Image 1")
        axes[2*i].axis("off")
        # for i, (pt, color) in enumerate(zip(points1, colormap)):
        #     axes[2*i].scatter(pt[0], pt[1], color=color, s=80)

        # Plot Image 2 with matched points
        axes[2*i+1].imshow(image2.transpose(1,2,0))
        axes[2*i+1].set_title("Image 2")
        axes[2*i+1].axis("off")

    # if inliers is not None:
    #     for i, (pt1, pt2, color) in enumerate(zip(kpts0, kpts1, colormap)):

    #         if inliers[i] == False:

    #             # axes[0].scatter(pt1[0], pt1[1], color='white', s=35,alpha=0.8)
    #             # axes[1].scatter(pt2[0], pt2[1], color='white', s=35,alpha=0.8)
    #             pass
    #         else:
    #             axes[0].scatter(pt1[0], pt1[1], color=color, s=35,alpha=0.95)
    #             axes[1].scatter(pt2[0], pt2[1], color=color, s=35,alpha=0.95)
    # else:
    #     for i, (pt1, pt2, color) in enumerate(zip(kpts0, kpts1, colormap)):

    #         axes[0].scatter(pt1[0], pt1[1], color=color, s=35,alpha=0.95)
    #         axes[1].scatter(pt2[0], pt2[1], color=color, s=35,alpha=0.95)

    plt.savefig("output.png", bbox_inches='tight', pad_inches=0)
    
    # Convert the figure to a NumPy array
    fig.canvas.draw()
    
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    

    plt.close(fig)  # Close the figure to free memory
    return image_array[...,1:]  # Return the NumPy array of the plotted image

def error_auc(errors, thresholds = [5, 10, 20]):
    """
    Args:
        errors (list): [N,]
        thresholds (list)
    """
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def symmetric_epipolar_distance(pts0, pts1, E, K0, K1):
    """Squared symmetric epipolar distance.
    This can be seen as a biased estimation of the reprojection error.
    Args:
        pts0 (torch.Tensor): [N, 2]
        E (torch.Tensor): [3, 3]
    """
    pts0 = (pts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    pts1 = (pts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    pts0 = convert_points_to_homogeneous(pts0)
    pts1 = convert_points_to_homogeneous(pts1)

    Ep0 = pts0 @ E.T  # [N, 3]
    p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
    Etp1 = pts1 @ E  # [N, 3]
    # import pdb
    # pdb.set_trace()

    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))  # N
    return d


def compute_symmetrical_epipolar_errors(pts0, pts1, data):
    """ 
    Update:
        data (dict):{"epi_errs": [M]}
    """
    Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
    E_mat = (Tx @ data['T_0to1'][:, :3, :3]).to(torch.float32)

    # m_bids = data['m_bids']
    # pts0 = data['mkpts0_f']
    # pts1 = data['mkpts1_f']


    epi_errs = []
    for bs in range(Tx.size(0)):
        # mask = m_bids == bs

        epi_errs.append(
            symmetric_epipolar_distance(pts0, pts1, E_mat[bs], data['K0'][bs], data['K1'][bs]))
    epi_errs = torch.cat(epi_errs, dim=0)

    data.update({'epi_errs': epi_errs})

def epidist_prec(errors, thresholds, ret_dict=False):
    precs = []
    for thr in thresholds:
        prec_ = []
        for errs in errors:
            correct_mask = errs < thr
            prec_.append(np.mean(correct_mask) if len(correct_mask) > 0 else 0)
        precs.append(np.mean(prec_) if len(prec_) > 0 else 0)
    if ret_dict:
        return {f'prec@{t:.0e}': prec for t, prec in zip(thresholds, precs)}
    else:
        return precs

def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    # normalize keypoints
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    # normalize ransac threshold
    ransac_thr = thresh / np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])

    # compute pose with cv2
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=ransac_thr, prob=conf, method=cv2.RANSAC)
    if E is None:
        print("\nE is None while trying to recover pose.\n")
        return None

    # recover pose from E
    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            ret = (R, t[:, 0], mask.ravel() > 0)
            best_num_inliers = n
    return ret

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--exp', type=str, default='exp')
    parser.add_argument(
        '--edm_dir', type=str, default='')
    parser.add_argument(
        '--m3ed_dir', type=str, default='')
    parser.add_argument(
        '--eds_dir', type=str, default='')
    parser.add_argument(
        '--mega_dir', type=str, default='')
    parser.add_argument(
        '--test_hetero', type=bool, default=False)
    parser.add_argument(
        '--custom_dir', type=str, default='')
    parser.add_argument(
        '--ref', type=int, default=1)
    parser.add_argument(
        '--repre', type=str, default='')
    parser.add_argument(
        '--ckpt_path', type=str, default=None)
    parser.add_argument(
        '--num_workers', type=int, default=16)
    parser.add_argument(
        '--ransac_thr', type=float, default=0.5)
    parser.add_argument(
        '--min_overlap', type=float, default=0.1)
    parser.add_argument(
        '--model', type=str, default='ours')
    parser.add_argument(
        '--test_set', type=str, default='ecm')
    parser.add_argument(
        '--vis', action='store_true')
    parser.add_argument(
        '--eval', action='store_true')
    parser.add_argument(
        '--no_metrics', action='store_true')
    
    
    parser.add_argument(
        '--eval_num', type=int, default=0)

    return parser.parse_args()

def init_superevent(device):
    sys.path.append('./third_party/SuperEvent')

    from models.super_event import SuperEvent, SuperEventFullRes
    
    with open('./third_party/SuperEvent/config/super_event.yaml', "r") as f:
        config = yaml.safe_load(f)

    if "backbone" in config:
        # Load backbone config and add
        backbone_config_path = os.path.join('./third_party/SuperEvent/config/', "backbones", config["backbone"] + ".yaml")
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
    print(max_factor_required)
    list_of_files = glob.glob("./third_party/SuperEvent/saved_models/*.pth")
    model_dir = max(list_of_files, key=os.path.getctime)

    model = SuperEvent(config, tracing=False)
    model.load_state_dict(torch.load(model_dir, weights_only=True))
    model.to(device)
    model.eval()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    return {'model':model,
            'bf':bf,
            'config':config
            }

def infer_superevent(pack, data, device):

    from models.util import fast_nms
    model = pack['model']
    bf = pack['bf']
    config = pack['config']
    pred0 = model(data['image'])
    pred1 = model(data['event'])
    top_k = np.max(data['image'].shape) // 2
    kpts0, _ = fast_nms(pred0["prob"], config, top_k=top_k)
    kpts1, _ = fast_nms(pred1["prob"], config, top_k=top_k)

    desc_0 = pred0["descriptors"][0, :, kpts0[0][:, 0], kpts0[0][:, 1]].permute(1, 0).cpu().detach().numpy()
    kpts_0 = kpts0[0].cpu().detach().numpy()
    desc_1 = pred1["descriptors"][0, :, kpts1[0][:, 0], kpts1[0][:, 1]].permute(1, 0).cpu().detach().numpy()
    kpts_1 = kpts1[0].cpu().detach().numpy()

    matches = bf.match(desc_0, desc_1)
    matched_kpts = np.array([[kpts_0[m.queryIdx], kpts_1[m.trainIdx]] for m in matches]).astype(np.float32)
    matched_kpts = np.roll(matched_kpts, 1, axis=-1)
    pts0 = matched_kpts[:,0]
    pts1 = matched_kpts[:,1]

    return pts0, pts1


def init_edm(device):
    from config.default import get_cfg_defaults
    from lightning.lightning_model import PL_MatchAnyEvents
    from utils.profiler import build_profiler
    config = get_cfg_defaults()

    config.MAE.TRAIN_DATA = 'image'
    profiler = build_profiler(None)
    model = PL_MatchAnyEvents(config, pretrained_ckpt='./pretrained/pretrained_v1.pth', profiler=profiler).to(device=device) # event dino larger set

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

def init_matchanything(device):
    import sys
    sys.path.append('./third_party/MatchAnything/')
    
    from src.lightning.lightning_loftr import PL_LoFTR
    from src.config.default import get_cfg_defaults
    from src.utils.dataset import dict_to_cuda
    from src.utils.metrics import estimate_homo, estimate_pose, relative_pose_error
    from src.utils.homography_utils import warp_points
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    from transformers.image_utils import load_image

    config = get_cfg_defaults()
    config.merge_from_file('/home/rex/Desktop/github/heterogeneous_matching/third_party/MatchAnything/configs/models/eloftr_model.py')
    config.LOFTR.COARSE.NPE = [832, 832, 640, 640]
    config.METHOD = "matchanything_eloftr"
    matcher = PL_LoFTR(config, pretrained_ckpt='/home/rex/Desktop/github/matchanything/weights/matchanything_eloftr.ckpt', test_mode=True).matcher
    matcher.eval().cuda()

    return {'matcher':matcher}

def infer_matchanything(pack, data, device):
    # Load the processor and model from the Hugging Face Hub
    image1 = data['image'].squeeze(0).squeeze(0)
    image2 = data['event'].squeeze(0).squeeze(0)

    img1 = (image1 * 255).byte().cpu().numpy()
    img2 = (image2 * 255).byte().cpu().numpy()

    # Convert grayscale to RGB
    img_rgb1 = Image.fromarray(img1, mode='L').convert('RGB')
    img_rgb2 = Image.fromarray(img2, mode='L').convert('RGB')
    images = [img_rgb1, img_rgb2]
    matcher = pack['matcher']
    # processor = pack['processor']
    # Process images and get model outputs
    batch = {'image0':data['image'].to(device),  'image1':data['event'].to(device)}

    with torch.no_grad():
        outputs = matcher(batch)

    b_mask = batch['m_bids'] == 0
    pts0 = batch['mkpts0_f'][b_mask].cpu().numpy()
    pts1 = batch['mkpts1_f'][b_mask].cpu().numpy()
    return pts0, pts1

def init_vggt(device):
    from vggt.models.vggt import VGGT

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    return {'model':model}

def infer_vggt(pack, data, device):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    model = pack['model']
    # Load and preprocess example images (replace with your own image paths)
    data0 = data['image'].repeat(1,3,1,1)
    data1 = data['event'].repeat(1,3,1,1)
    input = torch.cat([data0, data1], dim=0)
    
   

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = input[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Tracks
        w = EDM_CONFIG['val_res'][1]
        h = EDM_CONFIG['val_res'][0]
        # choose your own points to track, with shape (N, 2) for one scene
        y = torch.arange(0, h, 14)
        x = torch.arange(0, w, 14)

        # Create meshgrid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        query_points = torch.stack([xx.flatten(),yy.flatten()],dim=-1).to(device).to(torch.float)
 
        
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
        points = track_list[0].squeeze(0)
        conf = conf_score[0].squeeze(0)[-1]

        vggthreshold = 0.02 # M3ed
        # vggthreshold = 0.001 # ecm
        # vggthreshold = 0.004 # ecm0
        # vggthreshold = 0.01 #eds
        conf_mask = conf>vggthreshold

        mask = ((points[-1,:,-1] < h) & ( points[-1,:,-1] > 0)) & ((points[-1,:,-2] < w) & (points[-1,:,-2]>0))

        points = points[:,mask & conf_mask]

        pts0 = points[0]
        pts1 = points[1]
    return pts0.cpu().numpy(), pts1.cpu().numpy()

    # with torch.no_grad():
    #     with torch.cuda.amp.autocast(dtype=dtype):
    #         # Predict attributes including cameras, depth maps, and point maps.
    #         predictions = model(input)
    #         import pdb
    #         pdb.set_trace()
    #         extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], input.shape[-2:])
    #         predictions["extrinsic"] = extrinsic
    #         predictions["intrinsic"] = intrinsic

    # return extrinsic[0,-1]
           
def main():
    args = parse_args()
    assert args.model in ['edm','matchanything','vggt','superevent']
    test_seq_ecm = ['Franklin_corner', 'Franklin_main']
    EDM_CONFIG['min_overlap_score'] = args.min_overlap
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_dataset = []
    if args.test_set == 'ecm':
        root_dirs = [os.path.join(args.edm_dir, name) for name in test_seq_ecm]
    
        for dir in root_dirs:
            test_dataset.append(EDMDataset(dir, not args.test_hetero, args.repre, config=EDM_CONFIG))

    elif args.test_set == 'm3ed':
        for test_seq in ['falcon_outdoor_day_fast_flight_1','spot_outdoor_day_srt_under_bridge_1','car_urban_day_penno_small_loop', 'car_urban_night_penno_small_loop_darker']:
            test_dataset.append(M3EDDataset(os.path.join(args.m3ed_dir, test_seq), not args.test_hetero, args.repre, config=EDM_CONFIG))
        
    elif args.test_set == 'eds':
        # seqs =[f for f in os.listdir(args.eds_dir) if os.path.isdir(os.path.join(args.eds_dir, f))]
        seqs = ['11_all_characters']
        for seq in seqs:
            test_dataset.append(EDSDataset(os.path.join(args.eds_dir, seq), not args.test_hetero, args.repre, config=EDM_CONFIG))

    elif args.test_set == 'emega':
        mega_root = os.path.join( args.mega_dir,'event_megadepth/e_mega')
        # npz_root = os.path.join( args.mega_dir,'train-data/megadepth_indices/scene_info_val_1500')
        npz_root = os.path.join( args.mega_dir,'train-data/megadepth_indices/scene_info_0.1_0.7')
        test_list = os.path.join(args.mega_dir, "train-data/megadepth_indices/trainvaltest_list/train_list.txt")
        # test_list = os.path.join(args.mega_dir, "train-data/megadepth_indices/trainvaltest_list/val_list.txt")

        with open(test_list, "r") as f:
            test_npz_names = [f"{name.split()[0]}.npz" for name in f.readlines()]
        for npz_name in tqdm(test_npz_names, desc="Building MegaDepth test set"):
            npz_path = os.path.join(npz_root, npz_name)
            test_dataset.append(
                        EMegaDataset(
                            mega_root,
                            npz_path,
                            mode='test',
                            train_data='events',
                            min_overlap_score=0,
                            normalize_event=True,
                            ignore_list=[],
                            coarse_scale=0.125,
                            train_res=EDM_CONFIG['val_res']
                        )
                    )
    else:
        test_dataset.append(CustomDataset(os.path.join(args.custom_dir), not args.test_hetero, args.repre, roi_start=None, config=EDM_CONFIG, ref_index=args.ref))
    
    test_dataset = ConcatDataset(test_dataset)

    subset = Subset(test_dataset, range(args.eval_num, len(test_dataset)))
    test_loader = DataLoader(subset, batch_size=1, shuffle=False)

    # Initialize Model
    if args.model == 'superevent':
        pack = init_superevent(device)

    elif args.model == 'edm':
        pack = init_edm(device)
    
    elif args.model == 'matchanything':
        pack = init_matchanything(device)

    elif args.model == 'vggt':
        pack = init_vggt(device)

    with torch.no_grad():

        # print("Warming up network...")
        # for _ in range(100):
        #     example = torch.rand([1, config["input_channels"]] + cropped_shape).to(device)
        #     if compiled:
        #         example = example.to(torch.float16)
        #     model(example)
        # print("Done. Starting evaluation with profiling.")
        res = {}
        res.update({'R_errs': [], 't_errs': [], 'inliers': [], 'epi_errs': []})
        eval_len = len(test_loader)
        for iter, data in enumerate(test_loader):


            # if iter > 10: break
            vis0 = data['vis0']
            vis1 = data['vis1']
            K = data['K0'].numpy()[0]

            # R = data['T_0to1'][0,:3,:3].numpy()  # example rotation
            # t = data['T_0to1'][0,:3,-1].numpy().flatten()  # example translation

            # # --- Step 1: compute essential and fundamental matrices ---
            # def skew(t):
            #     return np.array([[0, -t[2], t[1]],
            #                     [t[2], 0, -t[0]],
            #                     [-t[1], t[0], 0]])

            # E = skew(t) @ R
            # F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

            # # --- Step 2: pick a point in image1 (in pixels) ---
            # # x1 = np.array([200, 200, 1])  # example point
            # x1 = np.array([148, 430, 1])  # (u, v, 1)

            # # # epipolar line in image 2
            # # l2 = F @ x1  # [a, b, c]
            # # a, b, c = l2 / np.linalg.norm(l2[:2])

            # # --- Step 3: draw the line on image 2 size (e.g. 1280x720) ---
            # img1 = (vis0[0,0].numpy()*255).astype(np.uint8)
            # img2 = (vis1[0,0].numpy()*255).astype(np.uint8)
            # w = vis0.shape[-1]
            # # --- Step 4: pick a query point in image1 ---
            
            # cv2.circle(img1, (int(x1[0]), int(x1[1])), 8, (255, 0, 0), -1)

            # # --- Step 5: compute epipolar line in image2 ---
            # l2 = F @ x1  # [a, b, c]
            # a, b, c = l2
            # # compute intersection with left and right borders
            # x0, x1b = 0, w
            # y0 = int((-c - a*x0) / b)
            # y1 = int((-c - a*x1b) / b)
            # cv2.line(img2, (x0, y0), (x1b, y1), (0, 255, 0), 2)

            # # --- Step 6: visualize side by side ---
            # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            # axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            # axes[0].set_title("Image 1 (query point)")
            # axes[0].axis('off')

            # axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            # axes[1].set_title("Image 2 (epipolar line)")
            # axes[1].axis('off')

            # plt.tight_layout()
            # plt.show()


            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            # Model inference
            import time
            start_time = time.perf_counter()
            if args.model == 'superevent':
                pts0, pts1 = infer_superevent(pack, data, device)
            elif args.model == 'edm':
                pts0, pts1 = infer_edm(pack, data, device)
            elif args.model == 'matchanything':
                pts0, pts1 = infer_matchanything(pack, data, device)
            elif args.model == 'vggt':
                pts0, pts1 = infer_vggt(pack, data, device)



            # import pdb
            # pdb.set_trace()

            stop_time = time.perf_counter()

            print(f'Infer speed {(stop_time - start_time)*1000:.1f} ms')
            
            # if args.model == 'vggt':
            #     # No need to estimate pose
            #     R = extrinsic[:3,:3]
            #     t = extrinsic[:3,-1]
            #     T_0to1 = data['T_0to1'].cpu().numpy()
            #     t_err, R_err = relative_pose_error(T_0to1[0], R.cpu(), t.cpu(), ignore_gt_t_thr=0.0)
            #     res['R_errs'].append([R_err])
            #     res['t_errs'].append([t_err])
            #     print(iter,'/',eval_len, 'R error',[R_err])
            # else:
            if not args.no_metrics:
                compute_symmetrical_epipolar_errors(torch.from_numpy(pts0).to(device),torch.from_numpy(pts1).to(device),data)
                res['epi_errs'].append(data['epi_errs'].cpu().numpy())
                K0 = data['K0'].cpu().numpy()
                K1 = data['K1'].cpu().numpy()
                T_0to1 = data['T_0to1'].cpu().numpy()
            # print(T_0to1)
            # print(K0,K1)

            # for bs in range(K0.shape[0]):
            bs = 0

            if not args.no_metrics:
                bpts0, bpts1 = pts0, pts1

                R_list, T_list, inliers_list = [], [], []
                for _ in range(5):
                    shuffling = np.random.permutation(np.arange(len(bpts0)))
                    bpts0 = bpts0[shuffling]
                    bpts1 = bpts1[shuffling]

                    # if args.test_set == 'eds':
                    #     try:
                    #         num_inliers, E, est_R, est_t, ransac_mask = cv2.recoverPose(points1=bpts0, points2=bpts1, 
                    #                                                                     cameraMatrix1=K0, cameraMatrix2=K1,
                    #                                                                     distCoeffs1=data['D'], distCoeffs2=data['D'],
                    #                                                                     method=cv2.RANSAC, threshold=3.0)
                    #     except cv2.error:
                    #         num_inliers = 0
                    #     if num_inliers == 0:
                    #         R_list.append(np.inf)
                    #         T_list.append(np.inf)
                    #         inliers_list.append(np.array([]).astype(bool))
                    #         inliers = None
                    #     else:
                    #         R_list.append(float(est_R))
                    #         T_list.append(float(est_t))
                    #         inliers = ransac_mask
                    #         inliers_list.append(ransac_mask)
                    # else:
                    
                    ret= estimate_pose(bpts0, bpts1, K0[bs], K1[bs], args.ransac_thr, conf=0.99999)
                    if ret is None:
                        inliers = None
                        R_list.append(np.inf)
                        T_list.append(np.inf)
                        inliers_list.append(np.array([]).astype(bool))
                    else:
                        R, t, inliers = ret
                        t_err, R_err = relative_pose_error(T_0to1[bs], R, t, ignore_gt_t_thr=0.0)
                        R_list.append(float(R_err))
                        T_list.append(float(t_err))
                        inliers_list.append(inliers)


                res['R_errs'].append(R_list)
                res['t_errs'].append(T_list)
                res['inliers'].append(inliers_list[0])
                print(iter,'/',eval_len, 'R error',R_list)
            
            if args.eval:
                vis0 = data['vis0']
                vis1 = data['vis1']
                event_plot_1 = torch.mean(data['image'][0:1,:].cpu(), dim=1)+0.5#plot_event(data_event.cpu())
                event_plot_2 = torch.mean(data['event'][0:1,:].cpu(), dim=1)+0.5#plot_event(data_event.cpu())
                if len(vis0) <1:
                    vis0 = event_plot_1
                if len(vis1) <1:
                    vis1 = event_plot_2


                fig_pred = _make_evaluation_figure(vis0.squeeze(0).permute(1,2,0).cpu().numpy(), vis1.squeeze(0).permute(1,2,0).cpu().numpy(), pts0, pts1, data['epi_errs'].cpu().numpy(),np.mean(R_list), np.mean(T_list))
                plt.show()

            if args.vis:
                event_plot_1 = torch.mean(data['image'][0:1,:].cpu(), dim=1)+0.5#plot_event(data_event.cpu())
                event_plot_2 = torch.mean(data['event'][0:1,:].cpu(), dim=1)+0.5#plot_event(data_event.cpu())
                vis0 = data['vis0'].squeeze(0).cpu().numpy()
                vis1 = data['vis1'].squeeze(0).cpu().numpy()

                
                fig_pred = plot_matched_points(vis0, vis1, pts0, pts1, inliers)
                cv2.imwrite("teaser.png", fig_pred)
                cv2.imshow('win',fig_pred)
                cv2.waitKey(0)


        
        if args.test_set == 'eds':
            # For eds dataset only test on rotation error
            pose_errors = np.array(res['R_errs']).reshape(-1)
            pass
        else:
            pose_errors = np.max(np.stack([res['R_errs'], res['t_errs']]), axis=0).reshape(-1)#[:,0]

        all_error = np.min(np.stack([np.min(res['R_errs'],axis=-1), np.min(res['t_errs'],axis=-1)]),axis=0)

        print(np.where(all_error > 10))
        aucs = error_auc(pose_errors)
        precs = epidist_prec(np.array(res['epi_errs'], dtype=object), [EDM_CONFIG['epipolar_threshold']], True)
        print(precs)
        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            print(f'auc@{thr}', torch.tensor(np.mean(aucs[f'auc@{thr}'])))

if __name__ == '__main__':
    main()