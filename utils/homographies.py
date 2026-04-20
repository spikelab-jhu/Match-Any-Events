import torch
import numpy as np
import cv2
import kornia
import kornia.geometry as KG
import torch.nn.functional as F
import time

config = {
    'translation': True,
    'rotation': True,
    'scaling': False,
    'perspective': True,
    'scaling_amplitude': 0.1,
    'perspective_amplitude_x': 0.1,
    'perspective_amplitude_y': 0.1,
    'patch_ratio': 0.8,
    'max_angle': 0.25,
    'allow_artifacts': False,
}

def batch_h_augumentation(data, train = True, scale = 8):
    images = data['image']
    assert len(images.shape) == 4
    bs, channel, h, w = images.shape

    warped_images = []
    batched_H = []
    batched_H_inv = []

    try: mask0 = data['mask0']
    except: mask0 = (torch.ones((bs, h//scale, w//scale)) > 0).to(images.device)

    try: mask1 = data['mask1']
    except: mask1 = (torch.ones((bs, h//scale, w//scale)) > 0).to(images.device)
    
    # Sample homographies
    for i in range(bs):
        
        # new_image, H, H_inv, mask0 = homography_augumentation(images[i], images[i].shape[-2:], mask0)
        H = sample_homography(images[i].shape[-2:], images.device, **config)
        H_inv = invert_homography(H).to(images.device)
        # warped_images.append(new_image)
        batched_H.append(H)
        batched_H_inv.append(H_inv)
        # batched_mask0.append(mask0)
        # batched_mask1.append(mask1)
    data.update({
        'H':torch.stack(batched_H,dim=0),
        'H_inv':torch.stack(batched_H_inv,dim=0),
    })
    # Homography transform

    warped_images, mask0 = tranform_images(data['H'], images, mask0)
    mask0 = mask0.bool()
    if 'contrast0' in data:
        warped_images, contrast0 = tranform_images(data['H'], images, data['contrast0'])
        if train:
            data.update({
                'contrast0':contrast0
            })
    if train:
        data.update({
            'image':warped_images,
            'mask0':mask0,
            'mask1':mask1
        })  

# def homography_augumentation(image, shape, mask,  homography_config = config):
#     # homo_start = time.perf_counter()
#     H = sample_homography(shape, image.device, **homography_config)
#     H_inv = invert_homography(H).to(image.device)
#     # homo_sample = time.perf_counter()
#     warped_image, mask = tranform_images(H, image, mask)
#     # homo_trans = time.perf_counter()
#     # print(f"[Iter {iter}] "
#     #         f"sample: {(homo_sample - homo_start)*1000:.1f} ms | "
#     #         f"transform: {(homo_trans - homo_sample)*1000:.1f} ms ")

#     return warped_image, H, H_inv, mask

def sample_homography(
        shape, device, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """Sample a random valid homography in PyTorch."""
    
    # Corners of the output image
    margin = (1 - patch_ratio) / 2
    pts1 = margin + torch.tensor([[0, 0], [0, patch_ratio],
                                  [patch_ratio, patch_ratio], [patch_ratio, 0]], dtype=torch.float32)
    # Corners of the input patch
    pts2 = pts1.clone()

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = torch.empty(1)
        torch.nn.init.trunc_normal_(perspective_displacement, mean=0.0, std=perspective_amplitude_y/2, a=-perspective_amplitude_y, b=perspective_amplitude_y)
        h_displacement_left = torch.empty(1)
        torch.nn.init.trunc_normal_(h_displacement_left, mean=0.0, std=perspective_amplitude_x/2, a=-perspective_amplitude_x, b=perspective_amplitude_x)
        h_displacement_right = torch.empty(1)
        torch.nn.init.trunc_normal_(h_displacement_right, mean=0.0, std=perspective_amplitude_x/2, a=-perspective_amplitude_x, b=perspective_amplitude_x)
        
        pts2 += torch.stack([torch.cat([h_displacement_left, perspective_displacement], 0),
                          torch.cat([h_displacement_left, -perspective_displacement], 0),
                          torch.cat([h_displacement_right, perspective_displacement], 0),
                          torch.cat([h_displacement_right, -perspective_displacement],
                                    0)])
        
    # Random scaling
    if scaling:
        n_scales = torch.empty(n_scales)
        torch.nn.init.trunc_normal_(n_scales, mean=1., std=scaling_amplitude/2, a = 1. - scaling_amplitude, b = 1. - scaling_amplitude)
        scales = torch.cat(
                [torch.tensor([1.]), n_scales], 0)
        center = torch.mean(pts2, dim=0, keepdim=True)
        scaled = (pts2 - center).unsqueeze(0) * scales.unsqueeze(1).unsqueeze(1) + center
        if allow_artifacts:
            valid = torch.arange(1, n_scales[0] + 1)
        else:
            valid = torch.nonzero((scaled >= 0.) & (scaled <= 1.), as_tuple=False)[:, 0]
        idx = valid[torch.randint(len(valid), (1,))]  # Random index
        pts2 = scaled[idx.to(int)].squeeze(0)

    # Random translation
    if translation:
        t_min, t_max = pts2.min(dim=0)[0], (1 - pts2).min(dim=0)[0]
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += torch.cat([torch.empty(1).uniform_(-t_min[0], t_max[0]),
                                         torch.empty(1).uniform_(-t_min[1], t_max[1])],0).unsqueeze(0)
    

    # Random rotation
    if rotation:
        angles = torch.linspace(-max_angle, max_angle, n_angles)
        angles = torch.cat([torch.tensor([0.]), angles], dim=0)  # Include 0-degree rotation
        center = pts2.mean(dim=0, keepdim=True)
        rot_mat = torch.stack([
            torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)
        ], dim=1).view(-1, 2, 2)
        rotated = torch.matmul(pts2 - center, rot_mat) + center
        if allow_artifacts:
            valid = torch.arange(1, n_angles + 1)
        else:
            valid = torch.nonzero((rotated >= 0.) & (rotated <= 1.), as_tuple=False)[:, 0]
        idx = valid[torch.randint(len(valid), (1,))]  # Random index
        pts2 = rotated[idx.to(int)].squeeze(0)

    # Rescale to actual size
    shape = torch.tensor(shape[::-1], dtype=torch.float32)  # reverse order [height, width]
    pts1 *= shape
    pts2 *= shape

    H = KG.find_homography_dlt(pts1.unsqueeze(0).to(device), pts2.unsqueeze(0).to(device))
    return H.squeeze(0)

    # def ax(p, q): return [-p[0], -p[1], -1, 0, 0, 0, p[0] * q[0], p[1] * q[0], q[0]]
    # def ay(p, q): return [0, 0, 0, -p[0], -p[1], -1, p[0] * q[1], p[1] * q[1], q[1]]
    # a_mat = torch.tensor([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)])

    # _,_,VT = torch.linalg.svd(a_mat)
    # H = VT[-1].reshape(3, 3)
    # H = H / H[2, 2]
    return H

def invert_homography(H):
    """
    Computes the inverse transformation for a flattened homography transformation.
    """
    return torch.linalg.inv(H)

def tranform_images(H, image, mask):
    '''
    Warp image and its mask with given homography
    Args:
        image in shape (Bs, C, h, w)
        mask in shape (Bs, h_c, w_c)
        H in shape (Bs, 3, 3)
    Returns:
        image in shape (Bs, C, h, w)
        mask in shape (Bs, h_c, w_c)
    '''
    def scale_homography(M, H, W, h, w):
        """Rescale homography from (H,W) space to (h,w) space."""
        sx, sy = w / W, h / H
        S = torch.tensor([[sx, 0, 0],
                        [0, sy, 0],
                        [0,  0, 1]], dtype=M.dtype, device=M.device).unsqueeze(0)
        S_inv = torch.inverse(S)
        return S @ M @ S_inv

    w = image.shape[-1]
    h = image.shape[-2]

    w_c = mask.shape[-1]
    h_c = mask.shape[-2]

    if len(mask.shape) == 3:
        mask = mask.unsqueeze(1)

    H_mask = scale_homography(H, h, w, h_c, w_c)
    # 2. Apply the homography to the mask
    image = kornia.geometry.transform.warp_perspective(image, H.to(image.device), dsize=(h, w))
    warped_mask = kornia.geometry.transform.warp_perspective(mask.float(), H_mask.to(image.device), dsize=(h_c, w_c))

    warped_mask = warped_mask.squeeze(1)
    # 3. Threshold it to create a binary mask
    
    return image, warped_mask

def tranform_points(H, points, shape_limit):
    # Input points in [n,2]
    assert points.shape[1] == 2 and len(shape_limit)==2
    points = torch.from_numpy(points)
    points = torch.cat([points, torch.ones((points.shape[0], 1))], dim=1)
    H_mat = torch.cat([H.flatten(), torch.tensor([1.0])],dim=0).reshape(3,3)
    points = points @ H_mat.T
    transformed_points = torch.round(points/points[:,-1].reshape(-1,1))
    index = torch.argwhere((transformed_points[:,0] >= 0.) & (transformed_points[:,0] < shape_limit[0]) & 
                        (transformed_points[:,1] >= 0.) & (transformed_points[:,1] < shape_limit[1]))
    return transformed_points[index.to(int),:-1], index.to(int)

