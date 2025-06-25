"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2 \
        --camera_type fixed

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import bpy
import bpy_extras

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default=".objaverse/hf-objaverse-v1/views_whole_sphere")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--total_images", type=int, default=24)
parser.add_argument("--camera_type", type=str, default='fixed', choices=['random', 'fixed', 'heuristic'])
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA" # or "OPENCL"

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camera):
    if camera.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camera.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camera.sensor_fit, camera.sensor_width, camera.sensor_height)
    sensor_fit = get_sensor_fit(
        camera.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camera.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camera.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def randomize_camera():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def compute_normal_entropy(obj, camera):
    mesh = obj.to_mesh()
    normals = []
    for v in mesh.vertices:
        co_world = obj.matrix_world @ v.co
        normal_world = obj.matrix_world.to_3x3() @ v.normal
        # Project to camera view
        co_ndc = bpy_extras.object_utils.world_to_camera_view(bpy.context.scene, camera, co_world)
        print("co_ndc:", co_ndc)
        if 0 <= co_ndc.x <= 1 and 0 <= co_ndc.y <= 1 and co_ndc.z >= 0:
            normals.append(normal_world.normalized())
    print(normals)
    normals = np.array([list(n) for n in normals], dtype=np.float32)
    # Compute entropy of normal directions (e.g., using histogram on spherical coordinates)
    theta = np.arccos(normals[:, 2])
    phi = np.arctan2(normals[:, 1], normals[:, 0])
    hist, _ = np.histogram2d(theta, phi, bins=16)
    p = hist / hist.sum()
    entropy = -np.nansum(p * np.log(p + 1e-8))
    return entropy

def get_top_k_by_normal_entropy(k=6):
    azimuths = (np.arange(args.total_images) / args.total_images * np.pi * 2).astype(np.float32)  # 0 to 2pi
    elevations = ((np.arange(args.total_images) / args.total_images - 0.5) * np.pi).astype(np.float32)  # -pi/2 to pi/2

    entropies = []
    mesh_objs = [obj for obj in bpy.context.scene.objects if isinstance(obj.data, bpy.types.Mesh)]
    if not mesh_objs:
        return [], []

    main_obj = mesh_objs[0]
    for i in range(args.total_images):
        # Set camera for each azimuth/elevation
        camera = set_camera_location(elevations[i], azimuths[i], args.camera_dist)
        entropy = compute_normal_entropy(main_obj, camera)
        entropies.append((entropy, azimuths[i], elevations[i]))

    # Sort by entropy descending and get top k
    entropies.sort(reverse=True, key=lambda x: x[0])
    print("(entropy, azimuth, elevation):", entropies)
    top_azimuths = [v[1] for v in entropies[:k]]
    top_elevations = [v[2] for v in entropies[:k]]
    return top_azimuths, top_elevations

def compute_surface_visibility(obj, camera, samples=1000):
    depsgraph = bpy.context.evaluated_depsgraph_get() # get dependency graph
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    visible_faces = set() # keep track of the indices of faces that are found to be visible

    for _ in range(samples): # randomly sample points
        face = random.choice(mesh.polygons) # randomly select a face
        # if len(face.vertices) < 3:
        #     continue  # skip non-triangular faces

        # Generate random barycentric coordinates
        bary = np.random.dirichlet([1, 1, 1])
        # v_indices = face.vertices[:3]
        # v0, v1, v2 = [mesh.vertices[i].co for i in v_indices]

        verts = face.vertices
        if len(verts) < 3:
            continue

        # Random triangle from face (fan method)
        i = random.randint(1, len(verts) - 2)
        v0 = mesh.vertices[verts[0]].co
        v1 = mesh.vertices[verts[i]].co
        v2 = mesh.vertices[verts[i + 1]].co

        # Convert to Vector for Blender math
        pt = (bary[0] * Vector(v0) +
              bary[1] * Vector(v1) +
              bary[2] * Vector(v2))
        pt = Vector(pt)

        direction = (pt - camera.location).normalized() # calculate the direction from the camera to the point
        print("direction", direction)

        # Raycast from camera to point
        result, loc, norm, idx = obj_eval.ray_cast(camera.location, direction)
        if result and idx == face.index:
            visible_faces.add(face.index)

    coverage = len(visible_faces) / len(mesh.polygons) if mesh.polygons else 0
    obj_eval.to_mesh_clear()
    return coverage 

def get_top_k_by_surface_visibility(k=6):
    azimuths = (np.arange(args.total_images) / args.total_images * np.pi * 2).astype(np.float32)  # 0 to 2pi
    elevations = ((np.arange(args.total_images) / args.total_images - 0.5) * np.pi).astype(np.float32)  # -pi/2 to pi/2

    visibilities = []
    mesh_objs = [obj for obj in bpy.context.scene.objects if isinstance(obj.data, bpy.types.Mesh)]
    if not mesh_objs:
        return [], []

    main_obj = mesh_objs[0]
    for i in range(args.total_images):
        # Set camera for each azimuth/elevation
        camera = set_camera_location(elevations[i], azimuths[i], args.camera_dist)
        visibility = compute_surface_visibility(main_obj, camera, samples=500)
        visibilities.append((visibility, azimuths[i], elevations[i]))

    # Sort by visibility descending and get top k
    visibilities.sort(reverse=True, key=lambda x: x[0])
    print("(visibility, azimuth, elevation):", visibilities)
    top_azimuths = [v[1] for v in visibilities[:k]]
    top_elevations = [v[2] for v in visibilities[:k]]
    return top_azimuths, top_elevations

# def get_visible_faces(obj, camera, samples=1000):
#     """Return a set of visible face indices from a given camera."""
#     depsgraph = bpy.context.evaluated_depsgraph_get()
#     obj_eval = obj.evaluated_get(depsgraph)
#     mesh = obj_eval.to_mesh()
#     visible_faces = set()
#     total_faces = len(mesh.polygons)
#     for _ in range(samples):
#         face = random.choice(mesh.polygons)
#         verts = face.vertices
#         if len(verts) < 3:
#             continue
#         i = random.randint(1, len(verts) - 2)
#         bary = np.random.dirichlet([1, 1, 1])
#         v0 = mesh.vertices[verts[0]].co
#         v1 = mesh.vertices[verts[i]].co
#         v2 = mesh.vertices[verts[i + 1]].co
#         pt = (bary[0] * Vector(v0) +
#               bary[1] * Vector(v1) +
#               bary[2] * Vector(v2))
#         pt = Vector(pt)
#         direction = (pt - camera.location).normalized()
#         result, loc, norm, idx = obj_eval.ray_cast(camera.location, direction)
#         if result and idx == face.index:
#             visible_faces.add(face.index)
#     obj_eval.to_mesh_clear()
#     return visible_faces, total_faces

def get_visible_faces(obj, camera, samples=1000):
    """Return a set of visible face indices from a given camera.

    This version samples the face centroid instead of random barycentric points,
    which works for faces with any number of vertices (n-gons).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()
    visible_faces = set()
    total_faces = len(mesh.polygons)
    for _ in range(samples):
        face = random.choice(mesh.polygons)
        verts = face.vertices
        if len(verts) < 3:
            continue
        # Compute centroid of the face (works for any n-gon)
        centroid = Vector((0, 0, 0))
        for idx in verts:
            centroid += mesh.vertices[idx].co
        centroid /= len(verts)
        pt = centroid
        direction = (pt - camera.location).normalized()
        result, loc, norm, idx = obj_eval.ray_cast(camera.location, direction)
        if result and idx == face.index:
            visible_faces.add(face.index)
    obj_eval.to_mesh_clear()
    return visible_faces, total_faces

def compare_visible_faces(obj, camera1, camera2, samples=1000):
    """Compare visible faces between two cameras."""
    faces1, total1 = get_visible_faces(obj, camera1, samples)
    faces2, total2 = get_visible_faces(obj, camera2, samples)
    overlap = faces1 & faces2
    only1 = faces1 - faces2
    only2 = faces2 - faces1
    print(f"Faces visible from camera 1: {len(faces1)} / {total1} = {len(faces1) / total1}")
    print(f"Faces visible from camera 2: {len(faces2)} / {total2} = {len(faces2) / total2}")
    print(f"Overlapping faces: {len(overlap)}")
    print(f"Unique to camera 1: {len(only1)}")
    print(f"Unique to camera 2: {len(only2)}")
    return {
        "camera1": faces1,
        "camera2": faces2,
        "overlap": overlap,
        "only_camera1": only1,
        "only_camera2": only2,
    }

def greedy_top_k_heuristic_viewpoints(obj, k=12, num_candidates=24, samples=1000, target_overlap=0.5):
    # Sample candidate camera positions (azimuth, elevation)
    azimuths = np.linspace(0, 2 * np.pi, num_candidates, endpoint=False)
    elevations = np.linspace(-np.pi/4, np.pi/4, num_candidates, endpoint=True)
    candidates = []
    for az in azimuths:
        for el in elevations:
            camera = set_camera_location(el, az, args.camera_dist)
            faces, total = get_visible_faces(obj, camera, samples)
            candidates.append((len(candidates), faces, (el, az, args.camera_dist)))
    assert k < len(candidates)

    top_k = []
    used_indices = set()
    used_faces = set()
    max_score = float('inf')

    # find 1 and 2
    v1, v2 = None, None
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            faces1 = candidates[i][1]
            faces2 = candidates[j][1]
            non_overlap = faces1 ^ faces2  # symmetric difference
            # print(non_overlap)
            overlap = len(faces1 & faces2)
            union = len(faces1 | faces2)
            if union == 0:
                continue
            overlap_ratio = overlap / union
            score = abs(overlap_ratio - target_overlap)
            if score < max_score:
                v1, v2 = i, j
                max_score = score
    top_k.append((max_score, candidates[v1][2]))
    top_k.append((max_score, candidates[v2][2]))
    used_indices.add(v1)
    used_indices.add(v2)
    used_faces = used_faces.union(candidates[v1][1], candidates[v2][1])

    while len(top_k) < k:
        v = None
        max_score = float('inf')
        for j in range(len(candidates)):
            if j in used_indices: continue
            faces2 = candidates[j][1]
            non_overlap = faces1 ^ faces2  # symmetric difference
            # print(non_overlap)
            overlap = len(used_faces & faces2)
            union = len(used_faces | faces2)
            overlap_ratio = overlap / union
            core = abs(overlap_ratio - target_overlap)
            if score < max_score:
                v = j
                max_score = score

        top_k.append((max_score, candidates[v][2]))
        used_indices.add(v)
        used_faces = used_faces.union(candidates[v][1])

    return top_k

def find_top_k_50_overlap_viewpoints(obj, k=12, num_candidates=24, samples=1000, target_overlap=0.5):
    """
    Find the top k pairs of camera viewpoints with overlap in visible faces closest to a target overlap.

    This function samples candidate camera positions around an object, computes the set of visible faces from each viewpoint,
    and then finds the top k pairs of viewpoints whose overlap in visible faces is closest to the specified target_overlap.

    Parameters:
        obj: The 3D object to render and analyze for visible faces.
        k (int): Number of top pairs to return. Default is 2.
        num_candidates (int): Number of candidate positions to sample for both azimuth and elevation. Default is 24.
        samples (int): Number of samples to use when determining visible faces. Default is 1000.
        target_overlap (float): The target overlap ratio (between 0 and 1) for visible faces between viewpoint pairs. Default is 0.5.

    Returns:
        list: A list of k tuples, each containing:
            - (el, az, dist): Elevation, azimuth, and distance for each camera.
    """
    # Sample candidate camera positions (azimuth, elevation)
    azimuths = np.linspace(0, 2 * np.pi, num_candidates, endpoint=False)
    elevations = np.linspace(-np.pi/4, np.pi/4, num_candidates, endpoint=True)
    candidates = []
    for az in azimuths:
        for el in elevations:
            camera = set_camera_location(el, az, args.camera_dist)
            faces, total = get_visible_faces(obj, camera, samples)
            candidates.append((len(candidates), faces, el, az, args.camera_dist))

    # Compare all pairs and find the top k pairs with overlap closest to target_overlap
    pairs = []
    for i in range(len(candidates)):
        for j in range(i+1, len(candidates)):
            faces1 = candidates[i][1]
            faces2 = candidates[j][1]
            overlap = len(faces1 & faces2)
            union = len(faces1 | faces2)
            if union == 0:
                continue
            overlap_ratio = overlap / union
            score = abs(overlap_ratio - target_overlap)
            pairs.append((
                score,
                i, j, overlap_ratio,
                (candidates[i][2], candidates[i][3], candidates[i][4]),
                (candidates[j][2], candidates[j][3], candidates[j][4])
            ))
    
    # Sort by score (distance from target_overlap)
    pairs.sort(key=lambda x: x[0])
    # Print mean, std, min, max, and percentiles of the scores
    scores = [entry[0] for entry in pairs]
    if scores:
        print(f"Mean score: {np.mean(scores):.4f}, Std score: {np.std(scores):.4f}")
        print(f"Min score: {np.min(scores):.4f}, Max score: {np.max(scores):.4f}")
        percentiles = np.percentile(scores, [0, 10, 25, 50, 75, 90, 100])
        print("Score percentiles:")
        for p, v in zip([0, 10, 25, 50, 75, 90, 100], percentiles):
            print(f"  {p:3d}th: {v:.4f}")
    else:
        print("No pairs to compute statistics.")
    
    # Collect top k unique camera viewpoints from the best pairs
    top_k = []
    used = set()
    for entry in pairs:
        # entry: (score, i, j, overlap_ratio, (el1, az1, dist1), (el2, az2, dist2))
        params1 = entry[4]
        params2 = entry[5]
        i, j = entry[1], entry[2]
        if i not in used:
            top_k.append(params1)
            used.add(i)
        if len(top_k) >= k:
            break
        if j not in used:
            top_k.append(params2)
            used.add(j)
        if len(top_k) >= k:
            break
    return top_k  # list of (el, az, dist)

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    randomize_lighting()

    if args.camera_type == 'heuristic':
        mesh_objs = [obj for obj in bpy.context.scene.objects if isinstance(obj.data, bpy.types.Mesh)]
        main_obj = mesh_objs[0]
        top_k = greedy_top_k_heuristic_viewpoints(main_obj)

        for idx, (score, (el, az, dist)) in enumerate(top_k):
            camera = set_camera_location(el, az, dist)

            # render the image
            render_path = os.path.join(args.output_dir, object_uid, f"{idx:03d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

            # save camera RT matrix
            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_path = os.path.join(args.output_dir, object_uid, f"{idx:03d}.npy")
            np.save(RT_path, RT)

            # save camera intrinsic matrix
            K = get_calibration_matrix_K_from_blender(camera.data)
            print(K)
            K_path = os.path.join(args.output_dir, object_uid, f"{idx:03d}_K.npy")
            np.save(K_path, K)
        print(top_k)
    else:
        # fixed camera viewpoints
        # azimuths = [30 + i * 60 for i in range(6)]  # [30, 90, 150, 210, 270, 330]
        # elevations = [30 if i % 2 == 0 else -20 for i in range(6)]  # [30, -20, 30, -20, 30, -20]
        azimuths = [(i * 360 / args.num_images) for i in range(args.num_images)]
        elevations = [30 if i % 2 == 0 else -20 for i in range(args.num_images)]

        for i in range(args.num_images):
            # # set the camera position
            # theta = (i / args.num_images) * math.pi * 2
            # phi = math.radians(60)
            # point = (
            #     args.camera_dist * math.sin(phi) * math.cos(theta),
            #     args.camera_dist * math.sin(phi) * math.sin(theta),
            #     args.camera_dist * math.cos(phi),
            # )
            # # reset_lighting()
            # cam.location = point

            # set camera
            if args.camera_type == 'random':
                camera = randomize_camera()
            elif args.camera_type == 'fixed':
                camera = set_camera_location(elevations[i], azimuths[i], args.camera_dist)
                
            # render the image
            render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

            # save camera RT matrix
            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
            np.save(RT_path, RT)

            # save camera intrinsic matrix
            K = get_calibration_matrix_K_from_blender(camera.data)
            print(K)
            K_path = os.path.join(args.output_dir, object_uid, f"{i:03d}_K.npy")
            np.save(K_path, K)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    # urllib.request.urlretrieve(object_url, tmp_local_path)
    import subprocess
    subprocess.run(["wget", "-O", tmp_local_path, object_url, "--no-check-certificate"], check=True)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


def test_heuristic(object_file: str) -> None:
    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    randomize_lighting()

    mesh_objs = [obj for obj in bpy.context.scene.objects if isinstance(obj.data, bpy.types.Mesh)]
    main_obj = mesh_objs[0]
    top_k = find_top_k_50_overlap_viewpoints(main_obj)

    os.makedirs(os.path.join(args.output_dir, object_uid), exist_ok=True)

    for idx, (el, az, dist) in enumerate(top_k):
        camera = set_camera_location(el, az, dist)

        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
        np.save(RT_path, RT)

if __name__ == "__main__":
    # try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        # test_heuristic(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    # except Exception as e:
    #     print("Failed to render", args.object_path)
    #     print(e)
