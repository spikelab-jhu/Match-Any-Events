import numpy as np
import cv2


def sample_random_transform(img_shape, max_angle=np.pi/36, max_shift_factor=0.05):
    h, w = img_shape[:2]
    max_shift = int(max(h, w) * max_shift_factor)
    angle = np.random.uniform(-max_angle, max_angle)
    cx = np.random.uniform(0, w)
    cy = np.random.uniform(0, h)
    tx = np.random.uniform(-max_shift, max_shift)
    ty = np.random.uniform(-max_shift, max_shift)
    return angle, (cx, cy), (tx, ty)


def interpolate_transforms(start, end, duration, fps=500):
    n_frames = int(duration * fps)
    angles = np.linspace(start[0], end[0], n_frames)
    centers = np.linspace(start[1], end[1], n_frames)
    translations = np.linspace(start[2], end[2], n_frames)
    return list(zip(angles, centers, translations))


def apply_transform(img, angle, center, translation):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)
    M[:, 2] += translation
    transformed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return transformed, M


def compute_largest_overlap_crop(img_shape, last_transform):
    """
    Compute the largest crop that overlaps between the original image and the last transformed image,
    preserving the original image aspect ratio.
    """
    h, w = img_shape[:2]
    dst_corners = np.array([
        [0, 0, 1],
        [w-1, 0, 1],
        [0, h-1, 1],
        [w-1, h-1, 1]
    ]).T

    mask, M = apply_transform(np.ones(img_shape, dtype=np.uint8), *last_transform)
    ymin, xmin, ymax, xmax = largest_rectangle_vectorized(mask)

    return xmin, ymin, xmax, ymax

def center_crop_aspect(img, target_h=360, target_w=640):
    H, W = img.shape[:2]
    target_ratio = target_h / target_w
    img_ratio = H / W
    
    if img_ratio > target_ratio:
        # Too tall → limit by width
        new_w = W
        new_h = int(W * target_ratio)
    else:
        # Too wide → limit by height
        new_h = H
        new_w = int(H / target_ratio)
    
    # Center crop
    y1 = (H - new_h) // 2
    x1 = (W - new_w) // 2
    y2 = y1 + new_h
    x2 = x1 + new_w
    
    return img[y1:y2+1, x1:x2+1], [x1,y1, x2,y2]

def largest_rectangle_vectorized(mask):
    h, w = mask.shape
    aspect_ratio = 1.0 * w/h
    integral = mask.astype(np.uint32).cumsum(axis=0).cumsum(axis=1)
    
    max_area = 0
    best_rect = None
    
    # Try candidate heights (or widths)
    for height in range(1, h+1):
        width = int(height * aspect_ratio)
        if width > w:
            break
        
        # sliding windows using vectorized sum
        y1 = np.arange(0, h-height+1)[:, None]
        y2 = y1 + height - 1
        x1 = np.arange(0, w-width+1)
        x2 = x1 + width - 1
        
        # compute sum for all windows
        s = integral[y2[:, None], x2] \
            - np.where(y1[:, None]-1 >= 0, integral[y1[:, None]-1, x2], 0) \
            - np.where(x1-1 >= 0, integral[y2[:, None], x1-1], 0) \
            + np.where((y1[:, None]-1 >= 0) & (x1-1 >= 0), integral[y1[:, None]-1, x1-1], 0)
        
        # check which windows are fully valid
        valid_windows = np.argwhere(s == height*width)
        if valid_windows.size > 0:
            max_area = height * width
            best_window = valid_windows[0]  # shape (2,)
            i = best_window[0]
            j = best_window[1]
            best_rect = (y1[i,0], x1[j], y2[i,0], x2[j])
    
    return best_rect



def generate_transformed_images(img, wt, ht, duration=1.0, fps=500, max_shift_factor = 0.025, max_angle = np.pi/48):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img, crop1 = center_crop_aspect(img, ht, wt)
    h, w = img.shape[0], img.shape[1]
    start = 0,0,0,0,0 #sample_random_transform(img.shape)
    end = sample_random_transform(img.shape, max_angle=max_angle, max_shift_factor = max_shift_factor)
    bound_w = int((np.tan(max_angle)+max_shift_factor) * max(h, w))
    bound_h = int((np.tan(max_angle)+max_shift_factor) * max(h, w) * h / w)

    transforms = interpolate_transforms(start, end, duration, fps)

    x1, y1, x2, y2 = 0 + bound_w, 0 + bound_h, w - bound_w, h - bound_h

    frames = []
    for angle, center, trans in transforms:
        frame, _ = apply_transform(img, angle, tuple(center), tuple(trans))
        frame_cropped = frame[y1:y2+1, x1:x2+1]
        scale_y, scale_x = 1.0 * ht/(y2+1-y1), 1.0 * wt/(x2+1-x1)
        frame_resized = cv2.resize(frame_cropped,(wt, ht), interpolation=cv2.INTER_LINEAR)
        frames.append(frame_resized)
    return frames, crop1, [x1, y1, x2, y2], scale_y, scale_x 

import os
def write_timestamp(dir, duration = 1.0, fps = 500):
    ts = np.linspace(0, duration, int(duration * fps))
    np.savetxt(os.path.join(dir, 'timestamps.txt'), ts)


# Example usage
if __name__ == "__main__":
    fps = 530
    duration = 0.256
    img = cv2.imread('visualization/000276.png')  # replace with your image
    frames = generate_transformed_images(img, wt=640, ht=360, duration=duration, fps=fps)
    print(f"Generated {len(frames)} cropped grayscale frames at 500Hz")
    
    # show a few sample frames
    for i in range(0, len(frames), 1):
        cv2.imwrite("./dataset/megadepth/test_frame/frame%05i.png"%i, frames[i])
    #     cv2.waitKey(100)
    # cv2.destroyAllWindows()
