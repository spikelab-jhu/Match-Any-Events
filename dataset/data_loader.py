import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import os 
import glob
import h5py
import numpy as np
# from event_utils import gen_discretized_event_volume
from dataset.utils.event_voxelgrid import VoxelGrid
# from image_utils import sobel_gradient_map
import torchvision.transforms as T
from utils.event_augumentation import temporal_augmentation


def prepare_mask(neg_event_volume, pos_event_volume, threshold = 0.05, erode_k = 3):
    neg_flat = torch.sum(neg_event_volume, dim=0)
    pos_flat = torch.sum(pos_event_volume, dim=0)
    valid_mask = torch.zeros_like(neg_flat)
    neg_max = torch.abs(neg_flat).max()
    pos_max = torch.abs(pos_flat).max()
    neg_flat = torch.abs(neg_flat/(neg_max+1e-6))
    pos_flat = torch.abs(pos_flat/(pos_max+1e-6))
    
    valid_mask[pos_flat>threshold] = 1
    valid_mask[neg_flat>threshold] = 1

    if erode_k>0:
        # Erosion using min pooling (invert values, apply max pooling, then invert back)
        valid_mask = -torch.nn.functional.max_pool2d(-valid_mask.unsqueeze(0), kernel_size=erode_k, stride=1, padding=(erode_k-1)//2).squeeze(0)
    if erode_k<0:
        valid_mask = torch.nn.functional.max_pool2d(valid_mask.unsqueeze(0), kernel_size=-erode_k, stride=1, padding=(-erode_k-1)//2).squeeze(0)
    return valid_mask

def event_norm(event_volume):
    '''
    Normalize event by each bin
    '''
    N,H,W = event_volume.shape
    max = torch.amax(torch.abs(event_volume),dim=(-2,-1),keepdim=True)
    event_volume = event_volume/(max+1e-6)
    # print(event_volume.max(), event_volume.min())
    return event_volume

class Heteromatch_dataloader:
    def __init__(self, dst_res, root_dir, mvsec = True, train_data = '', test_set = []):
        super().__init__()
        assert train_data in ['events','image','image_grad','event_duo_bin','default','all']
        seq_names = os.listdir(root_dir)
        # self.loftr_config = loftr_config

        assert len(seq_names) > 0
        
        self.train_seqs = [name for name in seq_names if name not in test_set]
        self.test_seqs = [name for name in seq_names if name in test_set]
        train_sequences = []
        test_sequences = []
        t_step = 1

        for name in self.train_seqs:
            train_sequences.append(Sequence(os.path.join(root_dir, name), 
                                            train_res=dst_res,
                                            use_mvsec=mvsec, 
                                            train_data=train_data,
                                            t_step=t_step,
                                            # loftr_config=self.loftr_config,
                                            train = True
                                            ))
        self.train_dataset = ConcatDataset(train_sequences)
        for name in self.test_seqs:
            test_sequences.append(Sequence(os.path.join(root_dir, name),
                                           train_res=dst_res,
                                           use_mvsec=mvsec, 
                                           train_data=train_data,
                                           t_step=t_step,
                                        #    loftr_config=self.loftr_config,
                                           train=False
                                           ))
        self.test_dataset = ConcatDataset(test_sequences)
        

        print('Dataset Summary: Test seqs ',test_set)
        print('Train length:',len(self.train_dataset), ' Test length:',len(self.test_dataset))

    
    def __len__(self):
        return len(self.train_dataset)+len(self.test_dataset)
    
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset
    

class Sequence(Dataset):
    
    def __init__(self, seq_dir, 
                 use_mvsec = False, 
                 train_res = (256, 512), 
                 train_data = '', 
                 normalize_event = False, 
                 coarse_scale = (1./8, 1./8),
                 dtype = torch.float32, 
                 t_step = 1,
                train = True
                ):
        super().__init__()

        self.seq_dir = seq_dir
        self.files = sorted(glob.glob(os.path.join(seq_dir,'*.h5')))
        self.coarse_scale = coarse_scale
        self.dtype = dtype
        self.train_data = train_data
        self.event_norm = normalize_event
        self.train = train
        if use_mvsec:
            self.resolution = (256, 336)
        else:
            self.resolution = (360, 640)
        self.train_res = train_res
        self.t_step = t_step
        self.transform = T.ColorJitter(
                brightness=0.2,
                contrast=0.2)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):

        filename = self.files[index]
        try:
            with h5py.File(filename) as file:

            # Read events and the corresponding image
                try:
                    name = str(file['dataset'][:])
                    id = int(file['index'][:])
                except:
                    print(filename, ' read problem')
                image = file['image'][:].astype(float)
                image_paired = file['image_paired'][:].astype(float)
                # if self.train_data == 'image_grad':
                #     image_paired = sobel_gradient_map(image_paired)
                seq_name = file['dataset'][:][0]
                e_idx = int(file['idx_e'][:])
                i_idx = int(file['idx_i'][:])

                K0 = torch.from_numpy(file['K0'][:].astype(float))
                K1 = torch.from_numpy(file['K1'][:].astype(float))
                data_image = torch.from_numpy(image/255.0).unsqueeze(0)
                data_image = self.transform(data_image)
                data_event = torch.from_numpy(file['event'][:].astype(float)[::self.t_step])
                data_event_paired = torch.from_numpy(file['event_paired'][:].astype(float)[::self.t_step])

                data_image_paired = torch.from_numpy(image_paired/255.0).unsqueeze(0)
                # event = {'x':torch.from_numpy(file['x'][:].astype(int)).to(self.dtype),
                #         'y':torch.from_numpy(file['y'][:].astype(int)).to(self.dtype),
                #         't':torch.from_numpy(file['t'][:]).to(self.dtype),
                #         'p':torch.from_numpy(file['p'][:]).to(self.dtype)
                #         }

                event_pose = torch.from_numpy(file['pose_e'][:]).to(self.dtype)
                image_pose = torch.from_numpy(file['pose_i'][:]).to(self.dtype)
                pose_paired = torch.from_numpy(file['pose_pi'][:].astype(float)).to(self.dtype)
                pose_event_paired = torch.from_numpy(file['pose_pe'][:].astype(float)).to(self.dtype)

                T_0to1 =  torch.linalg.inv(event_pose) @ (image_pose) 
                T_1to0 = torch.linalg.inv(T_0to1)
                depth_event = torch.from_numpy((file['depth_e'][:])).to(self.dtype)
                depth_image = torch.from_numpy((file['depth_i'][:])).to(self.dtype)
                depth_paired = torch.from_numpy((file['depth_pi'][:])).to(self.dtype)
                depth_event_paired = torch.from_numpy((file['depth_pe'][:])).to(self.dtype)
                depth_event[torch.isinf(depth_event)] = 0.
                depth_image[torch.isinf(depth_image)] = 0.
                depth_paired[torch.isinf(depth_paired)] = 0.
                depth_event_paired[torch.isinf(depth_event_paired)] = 0.
                
        except:
            print('bad file', filename)

        data_event = temporal_augmentation(data_event, train=self.train)
        data_event_paired = temporal_augmentation(data_event_paired, train=self.train)
        data_event = data_event[:self.num_bins*2]
        data_event_paired = data_event_paired[:self.num_bins*2]

        # if self.train_data == 'events' or self.train_data == 'all':

        #     # Mask region with valid events
        #     mask1 = prepare_mask(pos_event_volume=data_event[::2], neg_event_volume=data_event[1::2], threshold=0.03, erode_k=-21)
        #     mask0 = prepare_mask(pos_event_volume=data_event_paired[::2], neg_event_volume=data_event_paired[1::2], threshold = 0.03, erode_k=-21)
            
        #     mask_data = F.interpolate(torch.stack([mask0, mask1],dim=0).unsqueeze(1).float(),
        #                                                 scale_factor=self.coarse_scale,
        #                                                 mode='bilinear',
        #                                                 recompute_scale_factor=False).squeeze(1).bool()
        #     mask0_data = mask_data[0]
        #     mask1_data = mask_data[1]

        # Reshape data
        data_stack = torch.cat([
            data_image, data_image_paired, depth_event.unsqueeze(0), depth_image.unsqueeze(0), depth_paired.unsqueeze(0),depth_event_paired.unsqueeze(0), data_event, data_event_paired
        ],dim=0)
        data_resized = F.interpolate(data_stack.unsqueeze(0).float(),
                                size=self.train_res,
                                mode='nearest'
                                ).squeeze(0)
        data_event = data_resized[6:6+self.num_bins*2]
        data_event_paired = data_resized[6+self.num_bins*2:]
        data_image = data_resized[0].unsqueeze(0)
        data_image_paired = data_resized[1].unsqueeze(0)
        depth_event = data_resized[2]
        depth_image = data_resized[3]
        depth_paired = data_resized[4]
        depth_event_paired = data_resized[5]

            
        K0 = torch.tensor([
            [1.0 * self.train_res[1]/self.resolution[1]],
            [1.0 * self.train_res[0]/self.resolution[0]],
            [1.0]
        ])*K0
        K1 = torch.tensor([
            [1.0 * self.train_res[1]/self.resolution[1]],
            [1.0 * self.train_res[0]/self.resolution[0]],
            [1.0]
        ])*K1
        K0[-1,-1]=1
        K1[-1,-1]=1
        

        if self.event_norm:
            data_event = event_norm(data_event)
        if self.train_data == 'event_duo_bin':
            data_event = data_event[:2]
        
        data = {
            'event':data_event.to(self.dtype),
            'image':data_image.to(self.dtype),
            # 'mask1':mask1,
            # 'mask0':mask0,
            'K0':K0.to(self.dtype),
            'K1':K1.to(self.dtype),
            # 'image_paired': data_image_paired.to(self.dtype),
            'T_0to1': T_0to1,
            'T_1to0': T_1to0,
            'depth0': depth_image,
            'depth1': depth_event,
            'pair_names': (f'{seq_name, e_idx}', f'{seq_name,i_idx}'),
            # 'contrast1' : contrast_1.to(self.dtype),
            # 'e_idx':e_idx,
            # 'i_idx':i_idx
            # 'D0':torch.tensor(D0, dtype=self.dtype),
            # 'D1':torch.tensor(D1, dtype=self.dtype),
        }
        if self.train_data == 'image' or self.train_data == 'image_grad':
            T_0to1 = torch.linalg.inv(pose_paired) @ image_pose 
            data.update({
                'event':data_image_paired.to(self.dtype),
                'depth1':depth_paired,
                'T_0to1':T_0to1,
                'T_1to0':torch.linalg.inv(T_0to1),
                'K1':K0.to(self.dtype)
            })
        if self.train_data == 'events':
            T_0to1 = torch.linalg.inv(event_pose) @ pose_event_paired 
            data.update({
                'image':data_event_paired.to(self.dtype),
                'depth0':depth_event_paired,
                # 'mask1':mask1_data,
                # 'mask0':mask0_data,
                'T_0to1':T_0to1,
                'T_1to0':torch.linalg.inv(T_0to1),
                'K0':K1.to(self.dtype),
                # 'contrast0' : contrast_0.to(self.dtype),
            })
        if self.train_data == 'all':
            T_0to1 = torch.linalg.inv(event_pose) @ pose_event_paired 
            data.update({
                'event_paired':data_event_paired.to(self.dtype),
                'depth0_paired':depth_event_paired,
                'mask1_paired':mask1_data,
                'mask0_paired':mask0_data,
                'T_0to1_paired':T_0to1,
                'T_1to0_paired':torch.linalg.inv(T_0to1),
                'K0_paired':K1.to(self.dtype)
            })
            
        return data
    
def depth_interpolate(sparse_depth):


    import cv2
    nearest_depth = cv2.erode(sparse_depth, np.ones((9,9), np.float32))
    nearest_depth[nearest_depth>1e3] = np.inf
    return nearest_depth


import os.path as osp

class MegaDepthDataset(Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 mode='train',
                 normalize_event = False,
                 train_data = '',
                 ignore_list = [''],
                 min_overlap_score=0.3,
                 max_overlap_score=1.0,
                 train_res = (336, 560),
                 num_bins = 8,
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
        self.train_res = train_res
        self.num_bins = num_bins

        # prepare scene_info and pair_info
        # if mode == 'test' and min_overlap_score != 0:
        #     print("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
        #     min_overlap_score = 0
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
    
    def load_h5(self, name):
        with h5py.File(osp.join(self.root_dir, self.scene_id, 'images',  name+'.h5'), 'r') as f:
            event = torch.from_numpy(f['event'][:]).to(torch.float32)
            image = torch.from_numpy(f['image'][:].astype(int)).to(torch.float32)/255.0
            pose = torch.from_numpy(f['pose'][:]).to(torch.float32)
            depth = torch.from_numpy(f['depth'][:]).to(torch.float32)
            K = torch.from_numpy(f['K'][:]).to(torch.float32)
        if self.event_norm:
            event = event_norm(event)
        # depth[depth>1e3] = torch.inf
        return event, image, pose, depth, K

    def __getitem__(self, idx):
        (idx0, idx1), _, _ = self.pair_infos[idx]

        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = self.scene_info['image_paths'][idx0].split('/')[-1].split('.')[0]
        img_name1 = self.scene_info['image_paths'][idx1].split('/')[-1].split('.')[0]
    
        event_paired, image, T0, depth0, K_0 = self.load_h5(img_name0)
        event, image_paired, T1, depth1, K_1 = self.load_h5(img_name1)

        train = False
        if self.mode == 'train':
            train = True
        
        event = temporal_augmentation(event, train=train, T= self.num_bins)
        event_paired = temporal_augmentation(event_paired, train=train, T= self.num_bins)

        T_0to1 = T1 @ torch.inverse(T0)[:4, :4]  # (4, 4)
        T_1to0 = T_0to1.inverse()

        data_stack = torch.cat([
            image.unsqueeze(0), image_paired.unsqueeze(0), depth1.unsqueeze(0), depth0.unsqueeze(0), event, event_paired
        ],dim=0)
        data_resized = F.interpolate(data_stack.unsqueeze(0).float(),
                                size=self.train_res,
                                mode='nearest'
                                ).squeeze(0)
        event = data_resized[4:4+self.num_bins*2]
        event_paired = data_resized[4+self.num_bins*2:]
        image = data_resized[0].unsqueeze(0)
        image_paired = data_resized[1].unsqueeze(0)
        depth1 = data_resized[2]
        depth0 = data_resized[3]
            
        K_0 = torch.tensor([
            [1.0 * self.train_res[1]/self.resolution[1]],
            [1.0 * self.train_res[0]/self.resolution[0]],
            [1.0]
        ])*K_0
        K_1 = torch.tensor([
            [1.0 * self.train_res[1]/self.resolution[1]],
            [1.0 * self.train_res[0]/self.resolution[0]],
            [1.0]
        ])*K_1
        K_0[-1,-1]=1
        K_0[-1,-1]=1


        data = {
            'image': image,#.unsqueeze(0),  # (1, h, w)
            'depth0': depth0,  # (h, w)
            'event': event,
            'depth1': depth1,
            'T_0to1': T_0to1,  # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            # 'scale0': scale0,  # [scale_w, scale_h]
            # 'scale1': scale1,
            # 'dataset_name': 'MegaDepth',
            # 'scene_id': self.scene_id,
            # 'pair_id': idx,
            #
            'pair_names': (self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]),
        }
        if self.train_data == 'events':
            data['image'] = event_paired
        if self.train_data == 'image':
            data['event'] = image_paired


        return data