import os
import torch
import math
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import webdataset as wds
from torchvision import transforms
import torchvision
from einops import rearrange
from PIL import Image
import cv2

from ldm.util import instantiate_from_config
from ldm.modules.evaluate.evaluate import compute_evaluation_metrics
from ldm.data.simple import ObjaverseDataModuleFromConfig, ObjaverseData
from ldm.modules.evaluate.consistency import get_4x4_RT_matrix, get_R_and_t, get_relative_RT, get_essential_matrix, get_fundamental_matrix
from txining_consistency import draw_cross_keypoints, drawlines, distance_to_epipolar_line

# 1. Load config and instantiate model (adapt path as needed)
config_path = "/txining/zero123/zero123/configs/crossattn-vertical.yaml"  # <-- set this
ckpt_path = "/txining/zero123/zero123/logs/2025-07-04T08-17-57_sd-objaverse-finetune-c_concat-256/checkpoints/last.ckpt"  # <-- set this

config = OmegaConf.load(config_path)
model = instantiate_from_config(config.model)
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(state_dict, strict=False)
model.cuda()
model.eval()

# 2. Instantiate the datamodule
image_transforms = [torchvision.transforms.Resize(256)]
image_transforms.extend([transforms.ToTensor(),
                        transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
image_transforms = torchvision.transforms.Compose(image_transforms)

def load_im(path, color):
    '''
    replace background pixel with random color in rendering
    '''
    print("LOADING IMAGE FROM", path)
    img = plt.imread(path)
    img[img[:, :, -1] == 0.] = color
    img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
    return img

def process_im(im):
    im = im.convert("RGB")
    return image_transforms(im)

def cartesian_to_spherical(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    z = np.sqrt(xy + xyz[:,2]**2)
    theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    return np.array([theta, azimuth, z])


def get_T(target_RT, cond_RT):
    R, T = target_RT[:3, :3], target_RT[:, -1]
    T_target = -R.T @ T

    R, T = cond_RT[:3, :3], cond_RT[:, -1]
    T_cond = -R.T @ T

    theta_cond, azimuth_cond, z_cond = cartesian_to_spherical(T_cond[None, :])
    theta_target, azimuth_target, z_target = cartesian_to_spherical(T_target[None, :])
    
    d_theta = theta_target - theta_cond
    d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
    d_z = z_target - z_cond
    
    d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
    return d_T

# Convert from (1, 3, 256, 256) torch tensor to grayscale numpy array for cv2
def to_cv2_gray(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.shape[0] == 1:
        img = img[0]
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

root_dir = '/txining/zero123/objaverse-rendering/heuristic'
object_id = '437025b923c34f8085cdae03194b9c24'
filename1 = os.path.join(root_dir, object_id)
filename2 = os.path.join(root_dir, object_id)
index_target = 0
index_cond = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

color = [1., 1., 1., 1.]
target_im = process_im(load_im(os.path.join(filename1, '%03d.png' % index_target), color))
cond_im = process_im(load_im(os.path.join(filename2, '%03d.png' % index_cond), color))
cond_im_cv2 = cv2.imread(os.path.join(filename2, '%03d.png' % index_cond), cv2.IMREAD_GRAYSCALE)

target_RT = np.load(os.path.join(filename1, '%03d.npy' % index_target))
cond_RT = np.load(os.path.join(filename2, '%03d.npy' % index_cond))
target_K = np.load(os.path.join(filename1, '%03d_K.npy' % index_target))
cond_K = np.load(os.path.join(filename2, '%03d_K.npy' % index_cond))

def format_single_object_batch(target_im, cond_im, target_RT, cond_RT):
    T = get_T(target_RT, cond_RT)

    target_R, target_t = get_R_and_t(target_RT)
    target_RT4 = get_4x4_RT_matrix(target_R, target_t)
    cond_R, cond_t = get_R_and_t(cond_RT)
    cond_RT4 = get_4x4_RT_matrix(cond_R, cond_t)
    relative_RT4 = get_relative_RT(target_RT4, cond_RT4)

    batch = {
        "image_target": torch.tensor(target_im).unsqueeze(0),         # (1, C, H, W)
        "image_cond": torch.tensor(cond_im).unsqueeze(0),           # (1, num_cond_views, C, H, W)
        "T": torch.tensor(T).unsqueeze(0),                             # (1, num_cond_views, 4)
        "relative_RT4": torch.tensor(relative_RT4).unsqueeze(0),       # (1, num_cond_views, 4, 4)
    }
    return batch

# 3. Evaluate with tqdm
MAX_FEATURES = 50
sift = cv2.SIFT_create(MAX_FEATURES)

with torch.no_grad():
    device = next(model.parameters()).device
    batch = format_single_object_batch(target_im, cond_im, target_RT, cond_RT)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    _, loss_dict_no_ema = model.shared_step(batch)
    z, c, x_gt, xrec, xc = model.get_input(
        batch, model.first_stage_key, return_first_stage_outputs=True, force_c_encode=True, return_original_cond=True
    )
    
    relative_RT = batch["relative_RT4"][0].cpu().numpy()
    relative_R, relative_t = get_R_and_t(relative_RT)
    
    target_im = to_cv2_gray(xrec[0])
    cond_im = cond_im_cv2

    E = get_essential_matrix(relative_R, relative_t)
    F = get_fundamental_matrix(target_K, E, cond_K)

    print("E", E)
    print("F", F)

    # Compute SIFT keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(target_im, None)
    kp2, des2 = sift.detectAndCompute(cond_im, None)

    draw_cross_keypoints(target_im, kp1, f'{object_id}_target_kps.jpg')
    draw_cross_keypoints(cond_im, kp2, f'{object_id}_cond_kps.jpg')

    pts1 = np.array([kp.pt for kp in kp1])  # Shape: (N, 2)
    pts2 = np.array([kp.pt for kp in kp2])  # Shape: (N, 2)
    print("pts1", pts1)

    if len(pts1) == 0 or len(pts2) == 0:
        print("len(pts1) == 0 or len(pts2) == 0")

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # Note: the epilines are already normalised
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1,3)
    print("lines1", lines1)
    drawlines(target_im, cond_im, lines1, pts1, pts2, f'{object_id}_target_lines.jpg')

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1,3)
    drawlines(cond_im, target_im, lines, pts2, pts1, f'{object_id}_cond_lines.jpg')

    # Compute distances1 for every pair of line and point (all pairs)
    distances1 = np.zeros((lines1.shape[0], pts1.shape[0]))
    for i, line in enumerate(lines1):
        for j, pt in enumerate(pts1):
            distances1[i, j] = distance_to_epipolar_line(line, pt)

    distances2 = np.zeros((lines.shape[0], pts2.shape[0]))
    for i, line in enumerate(lines):
        for j, pt in enumerate(pts2):
            distances2[i, j] = distance_to_epipolar_line(line, pt)

    # Find the closest point in coordinates for each epipolar line (from lines1)
    min_indices_1 = np.argmin(distances1, axis=1)  # For each line, the closest point index in coordinates
    min_distances_1 = np.min(distances1, axis=1)
    min_indices_2 = np.argmin(distances2, axis=1)
    min_distances_2 = np.min(distances2, axis=1)

    # Prepare matched pairs: for each line (from pts2), get the closest point in coordinates
    matched_pts1 = pts1[min_indices_1]  # shape (N, 2)
    matched_pts2 = pts2[min_indices_2]  # shape (N, 2)

    color1 = np.random.randint(0,255,(lines1.shape[0],3))
    color2 = np.random.randint(0,255,(lines.shape[0],3))

    print("Avg min_dist:", np.mean(min_distances_1), np.mean(min_distances_2))

    drawlines(target_im, cond_im, lines1, pts1, pts2, f'{object_id}_target_matched.jpg', color1)
    drawlines(cond_im, target_im, lines, pts2, pts1, f'{object_id}_cond_matched.jpg', color2)



