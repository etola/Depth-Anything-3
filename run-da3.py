#!/usr/bin/env python3

"""
Script to use Depth Anything to generate 3D scene from COLMAP calibrations and images.

This script takes a scene folder containing:
- scene_folder/images: directory with images
- scene_folder/sparse: COLMAP reconstruction data

"""



import os
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'code')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'externals/depth-anything-3')))

import time
from typing import List, Optional
import numpy as np
import geometric_utility
import torch
from pathlib import Path

from PIL import Image

from densification import DensificationProblem, DensificationConfig
from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction

def get_conf_thresh(
    prediction: Prediction,
    sky_mask: Optional[np.ndarray],
    conf_thresh: float,
    conf_thresh_percentile: float = 10.0,
    ensure_thresh_percentile: float = 90.0,
):
    assert prediction.conf is not None
    if sky_mask is not None and (~sky_mask).sum() > 10:
        conf_pixels = prediction.conf[~sky_mask]
    else:
        conf_pixels = prediction.conf
    lower = np.percentile(conf_pixels, conf_thresh_percentile)
    upper = np.percentile(conf_pixels, ensure_thresh_percentile)
    conf_thresh = float(min(max(conf_thresh, lower), upper))
    return conf_thresh

def as_homogeneous44(ext: np.ndarray) -> np.ndarray:
    """
    Accept (4,4) or (3,4) extrinsic parameters, return (4,4) homogeneous matrix.
    """
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4, dtype=ext.dtype)
        H[:3, :4] = ext
        return H
    raise ValueError(f"extrinsic must be (4,4) or (3,4), got {ext.shape}")


def depths_to_world_points_with_colors(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    single_frame: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors. Return point confidences
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all, conf_all = [], [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        valid = np.isfinite(d) & (d > 0)
        if conf is not None:
            valid &= conf[i] >= conf_thr
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix[vidx].T  # (3,M)
        Xc = rays * d_flat[vidx][None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        cols = images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8)  # (M,3)
        confs = conf[i].reshape(-1)[vidx].astype(np.float32) # (M,)

        pts_all.append(Xw)
        col_all.append(cols)
        conf_all.append(confs)

        if single_frame:
            break

    if len(pts_all) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.float32)

    return np.concatenate(pts_all, 0), np.concatenate(col_all, 0), np.concatenate(conf_all, 0)


def align_prediction_depth_to_sfm(
    reconstruction,
    image_id: int,
    pred_depth: np.ndarray,
    pred_conf: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    K_w: int,
    K_h: int,
    conf_thr: float,
    min_track_length: int = 3,
) -> tuple[float, float, int]:
    """
    Compute a robust scale and shift to align a single-image prediction depth map
    to the COLMAP SfM registration.

    Projects COLMAP SfM 3D points using the supplied *K* / *pose* (typically COLMAP's
    own calibration), scales the resulting pixel coordinates to the prediction depth
    resolution, samples the prediction, and runs RANSAC to find
    ``scale * pred_depth + shift ≈ sfm_depth``.

    Args:
        reconstruction: ColmapReconstruction with SfM points.
        image_id:       COLMAP image ID.
        pred_depth:     (H_pred, W_pred) predicted depth map (may differ from K resolution).
        pred_conf:      (H_pred, W_pred) prediction confidence map.
        K:              (3, 3) intrinsics matrix (for a K_h × K_w image).
        pose:           (4, 4) cam-from-world pose.
        K_w:            Image width that *K* corresponds to.
        K_h:            Image height that *K* corresponds to.
        conf_thr:       Confidence threshold for filtering unreliable predictions.
        min_track_length: Minimum SfM track length for a 3D point to be used.

    Returns:
        scale:      Multiplicative factor.
        shift:      Additive offset.
        n_inliers:  Number of RANSAC inlier correspondences (0 ⇒ alignment failed).
    """
    H_pred, W_pred = pred_depth.shape

    # 1) SfM points visible in this image
    sfm_pts_3d, _, _ = reconstruction.get_visible_3d_points(
        image_id, min_track_length=min_track_length
    )
    if len(sfm_pts_3d) == 0:
        print(f"  [align] No SfM points visible for image {image_id}")
        return 1.0, 0.0, 0

    # 2) Project SfM points using the supplied K / pose → (u, v, depth_sfm)
    #    u, v are at K's resolution (K_w × K_h)
    uvd = geometric_utility.project_pts3d_to_image(sfm_pts_3d, K, pose)
    u, v, depth_sfm = uvd[:, 0], uvd[:, 1], uvd[:, 2]

    # 3) Scale pixel coordinates from K resolution to prediction depth resolution
    sx = W_pred / K_w
    sy = H_pred / K_h
    u_pred = u * sx
    v_pred = v * sy

    px = np.round(u_pred).astype(int)
    py = np.round(v_pred).astype(int)
    valid = (px >= 0) & (px < W_pred) & (py >= 0) & (py < H_pred) & (depth_sfm > 0)

    if valid.sum() < 50:
        print(f"  [align] Only {valid.sum()} valid SfM projections for image {image_id}, skipping")
        return 1.0, 0.0, 0

    px, py, depth_sfm = px[valid], py[valid], depth_sfm[valid]

    # 4) Sample prediction depth / confidence at scaled locations
    depth_pred = pred_depth[py, px]
    conf_pred = pred_conf[py, px]

    good = (conf_pred >= conf_thr) & (depth_pred > 0)
    if good.sum() < 50:
        print(f"  [align] Only {good.sum()} good correspondences for image {image_id}, skipping")
        return 1.0, 0.0, 0

    src_samples = depth_pred[good]
    dst_samples = depth_sfm[good]
    print(f"  [align] Image {image_id}: {good.sum()} correspondences from {len(sfm_pts_3d)} SfM points")

    # 5) Robust RANSAC scale + shift:  scale * pred + shift ≈ sfm
    try:
        scale, shift, n_inliers = geometric_utility.compute_robust_depth_scale_and_shift(
            src_samples, dst_samples
        )
    except ValueError as e:
        print(f"  [align] Failed for image {image_id}: {e}")
        return 1.0, 0.0, 0

    print(f"  [align] Result: scale={scale:.4f}, shift={shift:.4f}, inliers={n_inliers}/{good.sum()}")
    return scale, shift, n_inliers


def process_image_batch(problem: DensificationProblem, model: DepthAnything3, image_ids: List[int], batch_id: int = -1, single_frame: bool = False):

    filter_black_bg = False
    filter_white_bg = False
    conf_thresh = 1.05
    conf_thresh_percentile = 40.0
    ensure_thresh_percentile = 90.0

    # prepare inputs
    image_list = []
    extrinsics_list = []
    intrinsics_list = []

    sfm = problem.reconstruction

    for image_id in image_ids:
        iid_data = problem.get_depth_data(image_id)
        cam = sfm.get_image_camera(image_id)
        K = cam.calibration_matrix()
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :] = sfm.get_image_cam_from_world(image_id).matrix()

        image_list.append(iid_data['image_path'])
        intrinsics_list.append(K)
        extrinsics_list.append(pose_4x4)

    extrinsics = np.stack(extrinsics_list)
    intrinsics = np.stack(intrinsics_list)

    batch_out_dir = os.path.join(problem.depth_data_folder, f"da3_{batch_id:06d}")
    os.makedirs(batch_out_dir, exist_ok=True)

    prediction: Prediction = model.inference(
        image=image_list,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        process_res=problem.config.resolution,
        export_dir=batch_out_dir,
        # infer_gs=True,
        export_format="glb-depth_vis"
    )

    assert prediction.intrinsics is not None
    assert prediction.extrinsics is not None
    assert prediction.conf is not None
    assert prediction.processed_images is not None
    assert prediction.depth is not None
    # 1) get processed images
    images_u8 = prediction.processed_images # (N,H,W,3) uint8

    # 2) sky processing
    # Percentile used to fill sky pixels with plausible depth values.
    sky_depth_def = 98.0
    if prediction.sky is not None:
        non_sky_mask = ~prediction.sky
        valid_depth = prediction.depth[non_sky_mask]
        if valid_depth.size > 0:
            max_depth = np.percentile(valid_depth, sky_depth_def)
            prediction.depth[prediction.sky] = max_depth

    # 3) Confidence threshold (if no conf, then no filtering)
    if filter_black_bg and prediction.processed_images is not None and prediction.conf is not None:
        prediction.conf[(prediction.processed_images < 16).all(axis=-1)] = 1.0
    if filter_white_bg and prediction.processed_images is not None and prediction.conf is not None:
        prediction.conf[(prediction.processed_images >= 240).all(axis=-1)] = 1.0
    conf_thr = get_conf_thresh(prediction, prediction.sky, conf_thresh, conf_thresh_percentile, ensure_thresh_percentile)

    # --- Single frame: align depth to SfM, use COLMAP K/pose directly ---
    img = images_u8[0]
    pred_depth = prediction.depth[0]
    pred_conf = prediction.conf[0]

    image_id = image_ids[0]
    iid_data = problem.get_depth_data(image_id)
    img_name = os.path.basename(iid_data['image_name'])
    target_w, target_h = iid_data['w'], iid_data['h']

    # Align prediction depth to SfM registration using COLMAP K / pose
    scale, shift, n_inliers = align_prediction_depth_to_sfm(
        problem.reconstruction, image_id,
        pred_depth, pred_conf,
        iid_data['K'], iid_data['pose'],
        target_w, target_h,
        conf_thr,
    )
    if n_inliers >= 50 and np.isfinite(scale) and scale > 0:
        print(f"  Applying depth alignment for image {image_id}: "
                f"scale={scale:.4f}, shift={shift:.4f}, inliers={n_inliers}")
        pred_depth = scale * pred_depth + shift
    else:
        print(f"  Skipping depth alignment for image {image_id}: "
                f"insufficient quality (inliers={n_inliers})")

    # Resize aligned depth & confidence to COLMAP resolution
    aligned_depth = np.array(
        Image.fromarray(pred_depth.astype(np.float32), mode='F')
        .resize((target_w, target_h), Image.BILINEAR)
    )
    conf_map = np.array(
        Image.fromarray(pred_conf.astype(np.float32), mode='F')
        .resize((target_w, target_h), Image.BILINEAR)
    )

    # Apply confidence mask
    # mask = (aligned_depth > 0) & (conf_map >= conf_thr)
    mask = (aligned_depth > 0)
    aligned_depth[~mask] = 0

    # Use aligned depth directly as the depth map (no compute_depthmap round-trip)
    iid_data['depth_map'] = aligned_depth
    iid_data['confidence_map'] = conf_map
    iid_data['mask'] = mask
    problem.save_depth_data(image_id)
    problem.export_dmap_as_tiff(image_id)
    problem.save_heatmap(image_id)

    # Save depth map as npy
    dmaps_dir = os.path.join(problem.config.output_folder, "dmaps")
    os.makedirs(dmaps_dir, exist_ok=True)
    img_name = Path(os.path.basename(iid_data['image_name']))
    npy_path = os.path.join(dmaps_dir, img_name.with_suffix('.npy'))
    np.save(npy_path, aligned_depth)
    print(f"Saved depth map npy to {npy_path}")

    # Unproject aligned depth to 3D using COLMAP K / pose
    pts3d_world, valid_mask = geometric_utility.depthmap_to_world_frame(
        aligned_depth, iid_data['K'], iid_data['pose']
    )
    points = pts3d_world[valid_mask].astype(np.float32)

    # Get colours at COLMAP resolution
    img_resized = np.array(Image.fromarray(img).resize((target_w, target_h), Image.BILINEAR))
    colors = img_resized[valid_mask].astype(np.uint8)

    points, colors, confs = depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        images_u8,
        prediction.conf,
        conf_thr,
    )

    geometric_utility.save_point_cloud(points, colors/255.0, os.path.join(problem.cloud_folder, f"{img_name.with_suffix('.ply')}"))
    torch.cuda.empty_cache()

def main():

    config = DensificationConfig()
    config.parse_args()
    config.target_w = 840
    config.target_h = 840
    config.image_mode = 'keep_aspect'

    problem = DensificationProblem(config)
    problem.initialize(load_images=True)

    # model_dir = "depth-anything/DA3-LARGE"
    model_dir = "depth-anything/DA3NESTED-GIANT-LARGE"
    model = DepthAnything3.from_pretrained(model_dir)
    model = model.to("cuda")

    single_frame_per_batch = True

    ref_images = problem.active_image_ids
    processed_images = set()
    batch_id = 0
    for ref_image_id in ref_images:
        if ref_image_id in processed_images:
            continue
        batch = problem.reconstruction.find_image_cluster(ref_image_id=ref_image_id, n_max_images=problem.config.batch_size, images_to_skip=processed_images)
        if len(batch) < 3:
            print(f"Skipping batch: {batch} because it has less than 3 images")
            continue
        if not single_frame_per_batch:
            processed_images.update(batch)
        print(f"Processing batch: {batch}")
        process_image_batch(problem, model, batch, batch_id, single_frame_per_batch)
        batch_id += 1

    # merge all saved point clouds into a single ply file
    merge_point_clouds(problem.config.output_folder)

    problem.save_current_state()
    # problem.initialize_from_folder()

    problem.apply_fusion()
    problem.export_fused_point_cloud(file_name="fused.ply", use_parallel=True)


def merge_point_clouds(output_folder: str):
    # get all ply files in the output folder
    ply_files = glob.glob(os.path.join(output_folder, "points-*.ply"))

    print(f"Merging {len(ply_files)} point clouds")

    # merge all ply files into a single ply file
    geometric_utility.merge_point_clouds(ply_files, os.path.join(output_folder, "merged.ply"), max_points=10000000)


if __name__ == "__main__":
    main()
