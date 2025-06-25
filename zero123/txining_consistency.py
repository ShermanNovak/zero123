import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def skew(t):
    """Convert a vector to a skew-symmetric matrix"""
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

def draw_cross_keypoints(img, keypoints, output_file, color=(120,157,187)):
    """ Draw keypoints as crosses, and return the new image with the crosses. """
    img_kp = img.copy()  # Create a copy of img

    # Iterate over all keypoints and draw a cross on evey point.
    for kp in keypoints:
        x, y = kp.pt  # Each keypoint as an x, y tuple  https://stackoverflow.com/questions/35884409/how-to-extract-x-y-coordinates-from-opencv-cv2-keypoint-object

        x = int(round(x))  # Round an cast to int
        y = int(round(y))

        # Draw a cross with (x, y) center
        cv2.drawMarker(img_kp, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_8)

    cv2.imwrite(output_file, img_kp)

def drawlines(img1, img2, lines, coordinates, pts2, output_file, colors=None):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if colors is None:
        colors = [(120, 157, 187)] * len(lines)
    for color, r_line, pt1, pt2 in zip(colors, lines, coordinates, pts2):
        pt1 = int(round(pt1[0])), int(round(pt1[1]))
        pt2 = int(round(pt2[0])), int(round(pt2[1]))
        x0, y0 = map(int, [0, -r_line[2] / r_line[1]])
        x1, y1 = map(int, [c, -(r_line[2] + r_line[0] * c) / r_line[1]])
        color = tuple(map(int, color))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 3, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 3, color, -1)
    cv2.imwrite(output_file, img1)

def compute_geometric_consistency_approx_F(target_im, cond_im, object_id1, object_id2, index_target, index_cond):

    # Compute SIFT keypoints and descriptors
    MAX_FEATURES = 50
    sift = cv2.SIFT_create(MAX_FEATURES)
    kp1, des1 = sift.detectAndCompute(target_im, None)
    kp2, des2 = sift.detectAndCompute(cond_im, None)

    draw_cross_keypoints(target_im, kp1, f'{object_id1}_{index_target}_kps.jpg')
    draw_cross_keypoints(cond_im, kp2, f'{object_id2}_{index_cond}_kps.jpg')

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    coordinates = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            coordinates.append(kp1[m.queryIdx].pt)
    coordinates = np.int32(coordinates)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(coordinates,pts2,cv2.FM_LMEDS)

    print(f"{coordinates.shape[0]} matches found")
    if F is None:
        print("No fundamental matrix found.")
        return

    # We select only inlier points
    coordinates = coordinates[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1,3)
    drawlines(target_im, cond_im, lines1, coordinates, pts2, f'{object_id1}_{index_target}_lines.jpg')

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1,3)
    drawlines(cond_im, target_im, lines, pts2, coordinates, f'{object_id2}_{index_cond}_lines.jpg')

    # Compute distances1 for every pair of line and point (all pairs)
    distances1 = np.zeros((lines1.shape[0], coordinates.shape[0]))
    for i, line in enumerate(lines1):
        for j, pt in enumerate(coordinates):
            distances1[i, j] = distance_to_epipolar_line(line, pt)
    print(distances1)

    distances2 = np.zeros((lines.shape[0], pts2.shape[0]))
    for i, line in enumerate(lines):
        for j, pt in enumerate(pts2):
            distances2[i, j] = distance_to_epipolar_line(line, pt)

    # Find the closest point in coordinates for each epipolar line (from lines1)
    min_indices_1 = np.argmin(distances1, axis=1)  # For each line, the closest point index in coordinates
    min_distances_1 = np.min(distances1, axis=1)
    min_indices_2 = np.argmin(distances2, axis=1)
    min_distances_2 = np.min(distances2, axis=1)
    print(min_indices_1)

    # Prepare matched pairs: for each line (from pts2), get the closest point in coordinates
    matched_coordinates = coordinates[min_indices_1]  # shape (N, 2)
    matched_pts2 = pts2[min_indices_2]  # shape (N, 2)

    print(matched_coordinates)

    color1 = np.random.randint(0,255,(lines1.shape[0],3))
    color2 = np.random.randint(0,255,(lines.shape[0],3))

    print("Avg min_dist:", np.mean(min_distances_1), np.mean(min_distances_2))

    drawlines(target_im, cond_im, lines1, coordinates, pts2, color1, f'{object_id1}_{index_target}_matched.jpg')
    drawlines(cond_im, target_im, lines, pts2, coordinates, color2, f'{object_id2}_{index_cond}_matched.jpg')


def compute_geometric_consistency_with_K(
        target_im, cond_im, target_RT, cond_RT, target_K, cond_K, 
        object_id1, object_id2, index_target, index_cond
    ):
    target_R, target_t = get_R_and_t(target_RT)
    target_RT4 = get_4x4_RT_matrix(target_R, target_t)

    cond_R, cond_t = get_R_and_t(cond_RT)
    cond_RT4 = get_4x4_RT_matrix(cond_R, cond_t)

    relative_RT4 = get_relative_RT(target_RT4, cond_RT4)
    relative_R, relative_t = get_R_and_t(relative_RT4)

    E = get_essential_matrix(relative_R, relative_t)
    F = get_fundamental_matrix(target_K, E, cond_K)

    print("E", E)
    print("F", F)
    print("K", target_K)

    # Compute SIFT keypoints and descriptors
    MAX_FEATURES = 50
    sift = cv2.SIFT_create(MAX_FEATURES)
    kp1, des1 = sift.detectAndCompute(target_im, None)
    kp2, des2 = sift.detectAndCompute(cond_im, None)

    draw_cross_keypoints(target_im, kp1, f'{object_id1}_{index_target}_kps.jpg')
    draw_cross_keypoints(cond_im, kp2, f'{object_id2}_{index_cond}_kps.jpg')

    pts1 = np.array([kp.pt for kp in kp1])  # Shape: (N, 2)
    pts2 = np.array([kp.pt for kp in kp2])  # Shape: (N, 2)
    print("pts1", pts1)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # Note: the epilines are already normalised
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1,3)
    print("lines1", lines1)
    drawlines(target_im, cond_im, lines1, pts1, pts2, f'{object_id1}_{index_target}_lines.jpg')

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1,3)
    drawlines(cond_im, target_im, lines, pts2, pts1, f'{object_id2}_{index_cond}_lines.jpg')

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

    drawlines(target_im, cond_im, lines1, pts1, pts2, f'{object_id1}_{index_target}_matched.jpg', color1)
    drawlines(cond_im, target_im, lines, pts2, pts1, f'{object_id2}_{index_cond}_matched.jpg', color2)

def compute_geometric_consistency_with_translation(
    target_im, cond_im, target_RT, cond_RT, target_K, cond_K, 
    object_id1, object_id2, index_target, index_cond, 
    THRESHOLD=1
):
    target_im_np = np.array(target_im)
    cond_im_np = np.array(cond_im)
    N = target_im_np.shape[0]

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Remove the final classification layer to get feature embeddings
    # This will give you a model that outputs features of shape [B, 512]
    feature_extractor = torch.nn.Sequential(*list(model.children())[:2])
    # Set to evaluation mode
    feature_extractor.eval().to(device)

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    print(preprocess)

    target_feature_map = np.zeros((64, N, N), dtype=np.float32)
    cond_feature_map = np.zeros((64, N, N), dtype=np.float32)

    scale = int(math.floor((N-1) / (112-1)))
    print("scale", scale)

    with torch.no_grad():
        for tx in range(scale): 
            for ty in range(scale):
                N, M, _ = target_im_np.shape

                # translate image
                target_im_translated = np.zeros_like(target_im_np)
                target_im_translated[max(tx,0):M+min(tx,0), max(ty,0):N+min(ty,0)] = target_im_np[-min(tx,0):M-max(tx,0), -min(ty,0):N-max(ty,0)] 

                cond_im_translated = np.zeros_like(cond_im_np)
                cond_im_translated[max(tx,0):M+min(tx,0), max(ty,0):N+min(ty,0)] = cond_im_np[-min(tx,0):M-max(tx,0), -min(ty,0):N-max(ty,0)] 

                # convert image to tensor
                target_im_processed = preprocess(target_im).unsqueeze(0).to(device)
                cond_im_processed = preprocess(cond_im).unsqueeze(0).to(device)

                # get features
                target_f = feature_extractor(target_im_processed).cpu().numpy()
                cond_f = feature_extractor(cond_im_processed).cpu().numpy()

                # slot features into feature map
                for r in range(target_f.shape[-1]):
                    for c in range(target_f.shape[-1]):
                        print(r*scale+tx, c*scale+ty, r, c)
                        target_feature_map[:, r*scale+tx, c*scale+ty] = target_f[0, :, r, c]
                        cond_feature_map[:, r*scale+tx, c*scale+ty] = cond_f[0, :, r, c]

    target_R, target_t = get_R_and_t(target_RT)
    target_RT4 = get_4x4_RT_matrix(target_R, target_t)

    cond_R, cond_t = get_R_and_t(cond_RT)
    cond_RT4 = get_4x4_RT_matrix(cond_R, cond_t)

    relative_RT4 = get_relative_RT(target_RT4, cond_RT4)
    relative_R, relative_t = get_R_and_t(relative_RT4)

    E = get_essential_matrix(relative_R, relative_t)
    F = get_fundamental_matrix(target_K, E, cond_K)

    # image coordinates
    coordinates = np.array([(x, y) for y in range(N) for x in range(N)])

    # convert to homogeneous coordinates (Nx3)
    coordinates_3x3 = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])  # Nx3

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    # Note: the epilines are already normalised
    lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
    lines = lines.reshape(-1, 3)

    max_sims = []

    # for i in tqdm(range(len(lines))):
    for i in tqdm(range(10)):
        grid = np.zeros((N, N), dtype=np.uint8)
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

            grid[y2, x2] = 255

        if sims:
            max_sim = np.max(sims)
            print("max_sim", max_sim)
            max_sims.append(max_sim)

        if i < 10:
            plt.imshow(grid, cmap='gray')
            plt.title(f'Coordinates where dot product is 0 (line {i})')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'./feature_grid_translation/{object_id1}_{index_target}_dot_zero_grid_{i}.png')
            plt.close()
        
    return    

def compute_geometric_consistency_with_resnet(
        target_im, cond_im, target_RT, cond_RT, target_K, cond_K, 
        object_id1, object_id2, index_target, index_cond,
        THRESHOLD=1e-2
    ):

    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Remove the final classification layer to get feature embeddings
    # This will give you a model that outputs features of shape [B, 512]
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-5])
    # Set to evaluation mode
    feature_extractor.eval().to(device)

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    # convert image to tensor
    target_im_processed = preprocess(target_im).unsqueeze(0).to(device)
    cond_im_processed = preprocess(cond_im).unsqueeze(0).to(device)

    print(target_im_processed.shape)

    # what if there is background
    with torch.no_grad():
        target_f = feature_extractor(target_im_processed).to(device)
        cond_f = feature_extractor(cond_im_processed).to(device)
        print(target_f.shape) # [1, 2048, 7, 7]

        target_R, target_t = get_R_and_t(target_RT)
        target_RT4 = get_4x4_RT_matrix(target_R, target_t)

        cond_R, cond_t = get_R_and_t(cond_RT)
        cond_RT4 = get_4x4_RT_matrix(cond_R, cond_t)

        relative_RT4 = get_relative_RT(target_RT4, cond_RT4)
        relative_R, relative_t = get_R_and_t(relative_RT4)

        E = get_essential_matrix(relative_R, relative_t)
        F = get_fundamental_matrix(target_K, E, cond_K)

        print("E", E)
        print("F", F)

        # Create a matrix of coordinates for the pixels in (x, y) order
        N = 512
        d = target_f.shape[2]
        scale = (d-1)/(N-1)

        coordinates = np.array([(x, y) for y in range(N) for x in range(N)])
        coord_to_feat = np.round(coordinates * scale).astype(int)

        # Convert to homogeneous coordinates (Nx3)
        coordinates_3x3 = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])  # Nx3

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        # Note: the epilines are already normalised
        lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
        # lines = coordinates_3x3 @ F
        lines = lines.reshape(-1, 3)

        dot = lines @ coordinates_3x3.T  # shape: (NxN,NxN)
        i, j = np.where(np.isclose(dot, 0, atol=1e-2))

        # Get corresponding coordinates
        coords1 = coordinates[i_idx]  # shape (M, 2) → (x1, y1)
        coords2 = coordinates[j_idx]  # shape (M, 2) → (x2, y2)

        # Scale normalized coordinates to pixel indices
        fx1 = np.round(coords1[:, 0] * scale).astype(int)
        fy1 = np.round(coords1[:, 1] * scale).astype(int)
        fx2 = np.round(coords2[:, 0] * scale).astype(int)
        fy2 = np.round(coords2[:, 1] * scale).astype(int)

        # Use advanced indexing to get feature vectors: result shape (M, C)
        target_feat = target_f[0, :, fy1, fx1].T  # → (M, C)
        cond_feat = cond_f[0, :, fy2, fx2].T      # → (M, C)

        sims = cosine_similarity(target_feat, cond_feat, dim=1)
        print(np.mean(sims))

        return

        # normalise F
        # F = F * (1.0 / F[2, 2])

        # normalising coordinates
        # coordinates = np.array([(x * scale, y * scale) for y in range(d) for x in range(d)])
        # coordinates_3x3 = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])
        # lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
        # lines = lines.reshape(-1, 3)

        # for i in range(len(lines)):
        #     grid = np.zeros((d, d), dtype=np.uint8)
        #     dot = lines[i] @ coordinates_3x3.T  # shape: (NxN,)
        #     print("dot", dot)
        #     zero_indices = np.where(np.isclose(dot, 0, atol=1e-2))[0]  # indices where dot is close to 0

        #     for j in zero_indices:
        #         x1, y1 = coordinates[i]
        #         x2, y2 = coordinates[j]

        #         print(x1, y1, x2, y2)

        #         target_feat = target_f[0, :, x1, y1]
        #         cond_feat = cond_f[0, :, x2, y2]

        #         # Compute cosine similarity
        #         sim = cosine_similarity(target_feat.unsqueeze(0), cond_feat.unsqueeze(0)).item()
        #         print("sim", sim)
        #         sims.append(sim)

        #         grid[fy2, fx2] = 255
            
        #     if i < 10:
        #         plt.imshow(grid, cmap='gray')
        #         plt.title(f'Coordinates where dot product is 0 (line {i})')
        #         plt.xlabel('x')
        #         plt.ylabel('y')
        #         plt.savefig(f'./feature_grid_normalised/{object_id1}_{index_target}_dot_zero_grid_{i}.png')
        #         plt.close()

        # return

        coordinates = np.array([(x, y) for y in range(N) for x in range(N)])
        coord_to_feat = np.round(coordinates * scale).astype(int)

        # Convert to homogeneous coordinates (Nx3)
        coordinates_3x3 = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])  # Nx3

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        # Note: the epilines are already normalised
        lines = cv2.computeCorrespondEpilines(coordinates.reshape(-1, 1, 2), 1, F)
        # lines = coordinates_3x3 @ F
        lines = lines.reshape(-1, 3)
        print("lines", lines)

        max_sims = []

        for i in tqdm(range(len(lines))):
            grid = np.zeros((d, d), dtype=np.uint8)
            dot = lines[i] @ coordinates_3x3.T  # shape: (NxN,)
            zero_indices = np.where(np.isclose(dot, 0, atol=THRESHOLD))[0]  # indices where dot is close to 0
            sims = []

            for j in zero_indices:
                # Get (x, y) from coordinates[i] and coordinates[coord_idx]
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                # (rest of your code using x1, y1, x2, y2)

                # Map (x, y) to feature map indices
                fx1, fy1 = int(round(x1 * scale)), int(round(y1 * scale))
                fx2, fy2 = int(round(x2 * scale)), int(round(y2 * scale))

                # Extract feature vectors
                target_feat = target_f[0, :, fx1, fy1]
                cond_feat = cond_f[0, :, fx2, fy2]

                # Compute cosine similarity
                sim = cosine_similarity(target_feat.unsqueeze(0), cond_feat.unsqueeze(0)).item()
                sims.append(sim)

                grid[fy2, fx2] = 255

            if sims:
                max_sim = np.max(sims)
                print("max_sim", max_sim)
                max_sims.append(max_sim)

            if i < 10:
                plt.imshow(grid, cmap='gray')
                plt.title(f'Coordinates where dot product is 0 (line {i})')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.savefig(f'./feature_grid/{object_id1}_{index_target}_dot_zero_grid_{i}.png')
                plt.close()
        
    print("RESULTS", np.mean(max_sims), np.std(max_sims))

    return np.mean(max_sims)


        

        
    


root_dir = '/txining/zero123/objaverse-rendering/heuristic'
object_id1 = '660515f4ef554bb79da4d3bdf369a3ce'
object_id2 = '660515f4ef554bb79da4d3bdf369a3ce'
filename1 = os.path.join(root_dir, object_id1)
filename2 = os.path.join(root_dir, object_id2)
index_target = 5
index_cond = 6

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

target_im = cv2.imread(os.path.join(filename1, '%03d.png' % index_target), cv2.IMREAD_GRAYSCALE)
cond_im = cv2.imread(os.path.join(filename2, '%03d.png' % index_cond), cv2.IMREAD_GRAYSCALE)

target_RT = np.load(os.path.join(filename1, '%03d.npy' % index_target))
cond_RT = np.load(os.path.join(filename2, '%03d.npy' % index_cond))
target_K = np.load(os.path.join(filename1, '%03d_K.npy' % index_target))
cond_K = np.load(os.path.join(filename2, '%03d_K.npy' % index_cond))

# compute_geometric_consistency_with_K(
#     target_im, cond_im, target_RT, cond_RT, target_K, cond_K,
#     object_id1, object_id2, index_target, index_cond
# )

# compute_geometric_consistency_approx_F(
#     target_im, cond_im, object_id1, object_id2, index_target, index_cond
# )

target_im = Image.open(os.path.join(filename1, '%03d.png' % index_target)).convert('RGB')
cond_im = Image.open(os.path.join(filename2, '%03d.png' % index_cond)).convert('RGB')

# compute_geometric_consistency_with_resnet(
#     target_im, cond_im, target_RT, cond_RT, target_K, cond_K,
#     object_id1, object_id2, index_target, index_cond
# )

compute_geometric_consistency_with_translation(
    target_im, cond_im, target_RT, cond_RT, target_K, cond_K,
    object_id1, object_id2, index_target, index_cond
)