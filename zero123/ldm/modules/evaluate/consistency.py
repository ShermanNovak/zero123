import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
import numpy as np
import cv2
from tqdm import tqdm
import math
from PIL import Image

def skew(t):
    """Convert a vector to a skew-symmetric matrix"""
    if T.shape[1] != 3: print(T, T.shape)
    assert T.shape[1] == 3
    return np.array([
        [ 0,     -t[2],  t[1]],
        [ t[2],   0,    -t[0]],
        [-t[1],  t[0],   0  ]
    ])

def get_4x4_RT_matrix(R, t):
    RT4 = np.eye(4)
    RT4[:3, :3] = R
    RT4[:3, 3] = t
    return RT4

def get_R_and_t(RT_matrix):
    return RT_matrix[:3, :3], RT_matrix[:, -1]

def get_essential_matrix(R, t):
    return skew(t) @ R

def get_fundamental_matrix(K1, E, K2):
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

def get_relative_RT(target_RT4, cond_RT4):
    return cond_RT4 @ np.linalg.inv(target_RT4)

def distance_to_epipolar_line(line, point):
    # line: [a, b, c] coefficients of the line ax + by + c = 0
    # point: (x, y) coordinates of the point
    a, b, c = line
    x, y = point
    distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    return distance

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

K = np.array([
    [560.0,   0.0, 256.0],
    [  0.0, 560.0, 256.0],
    [  0.0,   0.0,   1.0]
])

def compute_geometric_consistency_with_K(img_rec_batch, img_gt_batch, relative_RT_batch):
    """
    Compute geometric consistency for batches of images and camera matrices.
    img_rec_batch: list or array of reconstructed images
    img_gt_batch: list or array of ground truth images
    relative_RT_batch: list or array of relative RT matrices (shape: [B, 4, 4])

    Returns: list of mean distances for each batch element
    """

    # print(img_rec_batch.shape)
    # print(img_gt_batch.shape)
    # print(relative_RT_batch.shape)

    batch_size = len(img_rec_batch)
    mean_distances = []
    MAX_FEATURES = 50
    sift = cv2.SIFT_create(MAX_FEATURES)

    for i in tqdm(range(batch_size)):
        img_rec = img_rec_batch[i]
        img_gt = img_gt_batch[i]
        relative_RT = relative_RT_batch[i].detach().cpu().numpy()

        if np.all(relative_RT == 0) and np.all(img_gt == 0): # padded 
            continue

        relative_R, relative_t = get_R_and_t(relative_RT)

        E = get_essential_matrix(relative_R, relative_t)
        F = get_fundamental_matrix(K, E, K)

        img_rec_cv = to_cv2_gray(img_rec)
        img_gt_cv = to_cv2_gray(img_gt)

        # Compute SIFT keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img_rec_cv, None)
        kp2, des2 = sift.detectAndCompute(img_gt_cv, None)

        if kp1 is None or kp2 is None or len(kp1) == 0 or len(kp2) == 0:
            mean_distances.append(np.nan)
            continue

        pts1 = np.array([kp.pt for kp in kp1])  # Shape: (N, 2)
        pts2 = np.array([kp.pt for kp in kp2])  # Shape: (M, 2)

        # Find epipolar lines corresponding to points in img_gt and drawing its lines on img_rec
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1,3)

        # Find epipolar lines corresponding to points in img_rec and drawing its lines on img_gt
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        
        # Compute distances for every pair of line and point (all pairs)
        distances1 = np.zeros((lines1.shape[0], pts1.shape[0]))
        for i, line in enumerate(lines1):
            for j, pt in enumerate(pts1):
                distances1[i, j] = distance_to_epipolar_line(line, pt)

        distances2 = np.zeros((lines2.shape[0], pts2.shape[0]))
        for m, line in enumerate(lines2):
            for n, pt in enumerate(pts2):
                distances2[m, n] = distance_to_epipolar_line(line, pt)

        # Find the closest point in pts2 for each epipolar line (from lines2)
        min_distances_1 = np.min(distances1, axis=1)
        min_distances_2 = np.min(distances2, axis=1)

        mean_distance_1 = np.mean(min_distances_1)
        mean_distance_2 = np.mean(min_distances_2)

        mean_distances.append(np.mean([mean_distance_1, mean_distance_2]))

    return np.nanmean(mean_distances)

def compute_geometric_consistency_approx_F(img_rec_batch, img_gt_batch):
    """
    Compute geometric consistency for batches of images and camera matrices.
    img_rec_batch: list or array of reconstructed images
    img_gt_batch: list or array of ground truth images

    Returns: list of mean distances for each batch element
    """
    batch_size = len(img_rec_batch)
    mean_distances = []
    MAX_FEATURES = 50
    sift = cv2.SIFT_create(MAX_FEATURES)

    for i in tqdm(range(batch_size)):
        img_rec = img_rec_batch[i]
        img_gt = img_gt_batch[i]

        if np.all(img_gt == 0): # padded 
            continue

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

        img_rec_cv = to_cv2_gray(img_rec)
        img_gt_cv = to_cv2_gray(img_gt)

        # Compute SIFT keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(img_rec_cv, None)
        kp2, des2 = sift.detectAndCompute(img_gt_cv, None)

        if kp1 is None or kp2 is None or len(kp1) == 0 or len(kp2) == 0:
            mean_distances.append(np.nan)
            continue

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            mean_distances.append(np.nan)
            continue
            
        matches = flann.knnMatch(des1,des2,k=2)
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        if len(pts1) == 0 or len(pts2) == 0 or len(pts1) < 8 or len(pts2) < 8:
            mean_distances.append(np.nan)
            continue

        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        # print(F)

        if F is None:
            mean_distances.append(np.nan)
            continue

        if F.shape != (3, 3): # OpenCV can return multiple 3×3 matrices stacked vertically if the method returns multiple candidates.
            F = F[:3, :] # Ensure F is 3x3

        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        # Find epilines corresponding to points in img_gt and drawing its lines on img_rec
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1,3)

        # Find epilines corresponding to points in img_rec and drawing its lines on img_gt
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)

        # Compute distances1 for every pair of line and point (all pairs)
        distances1 = np.zeros((lines1.shape[0], pts1.shape[0]))
        for i, line in enumerate(lines1):
            for j, pt in enumerate(pts1):
                distances1[i, j] = distance_to_epipolar_line(line, pt)

        # Compute distances for every pair of line and point (all pairs)
        distances2 = np.zeros((lines2.shape[0], pts2.shape[0]))
        for m, line in enumerate(lines2):
            for n, pt in enumerate(pts2):
                distances2[m, n] = distance_to_epipolar_line(line, pt)

        # Find the closest point in pts2 for each epipolar line (from lines2)
        min_distances_1 = np.min(distances1, axis=1)
        min_distances_2 = np.min(distances2, axis=1)

        mean_distance_1 = np.nanmean(min_distances_1)
        mean_distance_2 = np.nanmean(min_distances_2)

        mean_distances.append(np.nanmean([mean_distance_1, mean_distance_2]))

    return np.nanmean(mean_distances)

def compute_geometric_consistency_with_latent(
    img_rec_batch, img_gt_batch, relative_RT_batch, THRESHOLD=1e-2
):
    batch_size = len(img_rec_batch)
    mean_distances = []

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Remove the final classification layer to get feature embeddings
    # This will give you a model that outputs features of shape [B, 512]
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-5])
    # Set to evaluation mode
    feature_extractor.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    processed_img_rec_batch = torch.stack([preprocess(img) for img in img_rec_batch])
    processed_img_gt_batch = torch.stack([preprocess(img) for img in img_gt_batch])

    target_f = feature_extractor(processed_img_rec_batch).to(device)
    cond_f = feature_extractor(processed_img_gt_batch).to(device)

    # Create a matrix of coordinates for the pixels in (x, y) order
    N = img_rec_batch.shape[-1]
    d = target_f.shape[2]
    scale = (d-1)/(N-1)

    coordinates = np.array([(x, y) for y in range(N) for x in range(N)])
    # Convert to homogeneous coordinates (Nx3)
    coordinates_3x3 = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])  # Nx3

    for b in tqdm(range(batch_size)):
        relative_RT = relative_RT_batch[b].detach().cpu().numpy()

        if np.all(relative_RT == 0): # padded 
            continue

        relative_R, relative_t = get_R_and_t(relative_RT)

        E = get_essential_matrix(relative_R, relative_t)
        F = get_fundamental_matrix(K, E, K)

        with torch.no_grad():
            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            # Note: the epilines are already normalised
            lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
            # lines = coordinates_3x3 @ F
            lines = lines.reshape(-1, 3)

            dot = lines @ coordinates_3x3.T  # shape: (NxN,NxN)
            i, j = np.where(np.isclose(dot, 0, atol=THRESHOLD))

            # Get corresponding coordinates
            coords1 = coordinates[i]  # shape (M, 2) → (x1, y1)
            coords2 = coordinates[j]  # shape (M, 2) → (x2, y2)

            # Scale normalized coordinates to pixel indices
            fx1 = np.round(coords1[:, 0] * scale).astype(int)
            fy1 = np.round(coords1[:, 1] * scale).astype(int)
            fx2 = np.round(coords2[:, 0] * scale).astype(int)
            fy2 = np.round(coords2[:, 1] * scale).astype(int)

            # Use advanced indexing to get feature vectors: result shape (M, C)
            target_feat = target_f[b, :, fy1, fx1].T  # → (M, C)
            cond_feat = cond_f[b, :, fy2, fx2].T      # → (M, C)

            sims = cosine_similarity(target_feat, cond_feat, dim=1)
            mean_distances.append(torch.mean(sims).item())

    return np.mean(mean_distances)

def compute_geometric_consistency_with_translation(
    img_rec_batch, img_gt_batch, relative_RT_batch, THRESHOLD=1
):
    batch_size = len(img_rec_batch)
    N = img_rec_batch.shape[-1]
    d = 112

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Remove the final classification layer to get feature embeddings
    # This will give you a model that outputs features of shape [B, 512]
    feature_extractor = torch.nn.Sequential(*list(model.children())[:2])
    # Set to evaluation mode
    feature_extractor.eval().to(device)

    # image coordinates
    coordinates = np.array([(x, y) for y in range(N) for x in range(N)])
    # convert to homogeneous coordinates (Nx3)
    coordinates_3x3 = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])  # Nx3

    preprocess = transforms.Compose([
        transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for b in tqdm(range(batch_size)):
        img_rec = img_rec_batch[b].cpu().numpy()
        img_gt = img_gt_batch[b].cpu().numpy()

        if np.all(img_gt == 0): # padded 
            continue

        target_feature_map = np.zeros((64, N, N), dtype=np.float32)
        cond_feature_map = np.zeros((64, N, N), dtype=np.float32)
        scale = int(math.floor((N-1) / (d-1)))

        with torch.no_grad():
            for tx in range(scale): 
                for ty in range(scale):
                    print(img_rec.shape)

                    # translate image
                    target_im_translated = np.zeros_like(img_rec)
                    target_im_translated[max(tx,0):N+min(tx,0), max(ty,0):N+min(ty,0)] = img_rec[-min(tx,0):N-max(tx,0), -min(ty,0):N-max(ty,0)] 

                    cond_im_translated = np.zeros_like(img_gt)
                    cond_im_translated[max(tx,0):N+min(tx,0), max(ty,0):N+min(ty,0)] = img_gt[-min(tx,0):N-max(tx,0), -min(ty,0):N-max(ty,0)] 

                    # convert image to tensor
                    target_im_pil = Image.fromarray(target_im_translated)
                    cond_im_pil = Image.fromarray(cond_im_translated)
                    target_im_processed = preprocess(target_im_pil).unsqueeze(0).to(device)
                    cond_im_processed = preprocess(cond_im_pil).unsqueeze(0).to(device)

                    # get features
                    target_f = feature_extractor(target_im_processed).cpu().numpy()
                    cond_f = feature_extractor(cond_im_processed).cpu().numpy()

                    # slot features into feature map
                    for r in range(target_f.shape[-1]):
                        for c in range(target_f.shape[-1]):
                            # print(r*scale+tx, c*scale+ty, r, c)
                            target_feature_map[:, r*scale+tx, c*scale+ty] = target_f[0, :, r, c]
                            cond_feature_map[:, r*scale+tx, c*scale+ty] = cond_f[0, :, r, c]

        relative_RT = relative_RT_batch[b].detach().cpu().numpy()

        if np.all(relative_RT == 0): # padded 
            continue

        relative_R, relative_t = get_R_and_t(relative_RT)

        E = get_essential_matrix(relative_R, relative_t)
        F = get_fundamental_matrix(target_K, E, cond_K)

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        # Note: the epilines are already normalised
        lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
        lines = lines.reshape(-1, 3)

        max_sims = []

        for i in tqdm(range(10)):
        # for i in tqdm(range(len(lines))):
            dot = lines[i] @ coordinates_3x3.T  # shape: (NxN,)
            zero_indices = np.where(np.isclose(dot, 0, atol=THRESHOLD))[0]  # indices where dot is close to 0
            sims = []

            for j in zero_indices:
                # Get (x, y) from coordinates[i] and coordinates[coord_idx]
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]

                # Extract feature vectors
                target_feat = target_feature_map[:, x1, y1]
                cond_feat = cond_feature_map[:, x2, y2]

                # Compute cosine similarity using numpy
                sim = np.dot(target_feat, cond_feat) / (np.linalg.norm(target_feat) * np.linalg.norm(cond_feat) + 1e-8)
                sims.append(sim)

            if sims:
                max_sim = np.max(sims)
                # print("max_sim", max_sim)
                max_sims.append(max_sim)
            
    return np.mean(max_sims)