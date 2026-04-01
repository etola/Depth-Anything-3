
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import cv2

from numbers import Number
from typing import List, Tuple, Optional, Union

def uvd_to_world_frame(uvd_map: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Convert uvd map to world frame.

    Args:
        uvd_map: HxWx3 array containing [u, v, depth] coordinates OR Nx3 array
        intrinsics: 3x3 numpy array
        pose: 4x4 numpy array (cam_from_world)
    Returns:
        xyz_map: HxWx3 array containing [x, y, z] world coordinates (0 for invalid points)
                 OR Nx3 array if input was Nx3
    """
    # Check input shape and handle both HxWx3 and Nx3 formats
    if uvd_map.ndim == 2 and uvd_map.shape[1] == 3:
        # Nx3 format - flat array of points
        N = uvd_map.shape[0]
        xyz_output = np.zeros((N, 3), dtype=np.float32)

        # Extract valid points where depth > 0
        valid_mask = uvd_map[:, 2] > 0
        if not np.any(valid_mask):
            return xyz_output

        # Get valid uvd coordinates
        valid_u = uvd_map[valid_mask, 0]  # u coordinates
        valid_v = uvd_map[valid_mask, 1]  # v coordinates
        valid_d = uvd_map[valid_mask, 2]  # depth values

    elif uvd_map.ndim == 3 and uvd_map.shape[2] == 3:
        # HxWx3 format - image-like array
        H, W = uvd_map.shape[:2]
        xyz_output = np.zeros((H, W, 3), dtype=np.float32)

        # Extract valid points where depth > 0
        valid_mask = uvd_map[:, :, 2] > 0
        if not np.any(valid_mask):
            return xyz_output

        # Get valid uvd coordinates
        valid_u = uvd_map[valid_mask, 0]  # u coordinates
        valid_v = uvd_map[valid_mask, 1]  # v coordinates
        valid_d = uvd_map[valid_mask, 2]  # depth values
    else:
        raise ValueError(f"uvd_map must be either HxWx3 or Nx3, got shape {uvd_map.shape}")

    # Extract intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Convert to camera coordinates
    x_cam = (valid_u - cx) * valid_d / fx
    y_cam = (valid_v - cy) * valid_d / fy
    z_cam = valid_d

    # Stack into homogeneous coordinates (Nx4)
    cam_coords_homo = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=1)

    # Transform to world coordinates using inverse pose
    # pose is cam_from_world, so we need world_from_cam = inv(cam_from_world)
    world_from_cam = np.linalg.inv(pose)
    world_coords = (world_from_cam @ cam_coords_homo.T).T[:, :3]  # (N, 3)

    # Put world coordinates back into output array
    xyz_output[valid_mask] = world_coords

    return xyz_output


def compute_depthmap(points_3d: np.ndarray, intrinsics: np.ndarray, cam_from_world: np.ndarray, target_w: int, target_h: int, point_confidences: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Project 3D points to depth map in camera coordinate system.

    Args:
        points_3d: (N, 3) array of 3D world coordinates
        intrinsics: (3, 3) intrinsics matrix for a target_h x target_w image
        cam_from_world (4, 4): camera pose: transformation matrix that maps world coordinates to camera coordinates when multiplied
        target_w: target image width for depth map
        target_h: target image height for depth map

    Returns:
        depth_map: (target_h, target_w) numpy array with depth values
        confidence_map: (target_h, target_w) numpy array with confidence values (if point_confidences is provided)
    """
    if len(points_3d) == 0:
        return np.zeros((target_h, target_w), dtype=np.float32), np.zeros((target_h, target_w), dtype=np.float32)

    # Convert 3D points to homogeneous coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])

    # Transform to camera coordinates
    cam_coords = (cam_from_world[:3, :] @ points_3d_homo.T).T

    # Extract depth values (z-coordinates in camera frame)
    depths = cam_coords[:, 2]

    # Filter points behind camera
    valid_mask = depths > 0
    if not np.any(valid_mask):
        return np.zeros((target_h, target_w), dtype=np.float32), np.zeros((target_h, target_w), dtype=np.float32)

    cam_coords = cam_coords[valid_mask]
    depths = depths[valid_mask]
    confidences = None
    if point_confidences is not None:
        confidences = point_confidences[valid_mask]

    # Project to image coordinates
    proj_coords = (intrinsics @ cam_coords.T).T
    proj_coords = proj_coords / proj_coords[:, 2:3]  # Normalize by z

    # Convert to pixel coordinates
    pixel_x = proj_coords[:, 0].astype(int)
    pixel_y = proj_coords[:, 1].astype(int)

    # Filter points within image bounds
    valid_pixels = (
        (pixel_x >= 0) & (pixel_x < target_w) &
        (pixel_y >= 0) & (pixel_y < target_h)
    )

    depth_map = np.zeros((target_h, target_w), dtype=np.float32)
    cmap = None
    if  confidences is not None:
        cmap = np.zeros((target_h, target_w), dtype=np.float32)

    if np.any(valid_pixels):
        pixel_x = pixel_x[valid_pixels]
        pixel_y = pixel_y[valid_pixels]
        pixel_depths = depths[valid_pixels]
        if confidences is not None and cmap is not None:
            pixel_confs = confidences[valid_pixels]
            # Handle multiple points per pixel by taking the highest confidence
            np.maximum.at(cmap, (pixel_y, pixel_x), pixel_confs)

        # Handle multiple points per pixel by taking the closest depth
        # Replace 0s with infinity so that any actual depth will be smaller
        depth_map_working = np.where(depth_map == 0, np.inf, depth_map)
        # Use minimum.at to handle multiple points mapping to same pixel
        np.minimum.at(depth_map_working, (pixel_y, pixel_x), pixel_depths)
        # Replace any remaining infinities with 0 (shouldn't happen given our data)
        depth_map[:] = np.where(depth_map_working == np.inf, 0, depth_map_working)

    return depth_map, cmap



def depthmap_to_camera_frame(depthmap: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to a pointcloud in camera frame using numpy.

    Args:
        depthmap: HxW numpy array
        intrinsics: 3x3 numpy array

    Returns:
        pointmap in camera frame (HxWx3 array), and a mask specifying valid pixels.
    """
    height, width = depthmap.shape

    # Create pixel coordinate grids
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')

    # Extract intrinsics parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Convert to 3D points in camera frame
    depth_z = depthmap
    xx = (x_grid - cx) * depth_z / fx
    yy = (y_grid - cy) * depth_z / fy
    pts3d_cam = np.stack([xx, yy, depth_z], axis=-1)

    # Create valid mask for non-zero depth pixels
    valid_mask = depthmap > 0.0

    return pts3d_cam, valid_mask

def depthmap_to_world_frame(depthmap: np.ndarray, intrinsics: np.ndarray, cam_from_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to a pointcloud in world frame using numpy.

    Args:
        depthmap: HxW numpy array
        intrinsics: 3x3 numpy array
        cam_from_world: 4x4 numpy array

    Returns:
        pointmap in world frame (HxWx3 array), and a mask specifying valid pixels.
    """
    # Get 3D points in camera frame
    pts3d_cam, valid_mask = depthmap_to_camera_frame(depthmap, intrinsics)

    # Convert points from camera frame to world frame
    height, width = depthmap.shape

    # Convert to homogeneous coordinates
    pts3d_cam_homo = np.concatenate([
        pts3d_cam,
        np.ones((height, width, 1))
    ], axis=-1)

    cam_to_world = np.linalg.inv(cam_from_world)

    # Transform to world coordinates: pts_world = cam_to_world @ pts_cam_homo
    # Reshape for matrix multiplication: (H*W, 4) @ (4, 4) -> (H*W, 4)
    pts3d_cam_homo_flat = pts3d_cam_homo.reshape(-1, 4)
    pts3d_world_homo_flat = pts3d_cam_homo_flat @ cam_to_world.T

    # Reshape back and take only xyz coordinates
    pts3d_world = pts3d_world_homo_flat[:, :3].reshape(height, width, 3)

    return pts3d_world, valid_mask

def colorize_heatmap(data_map, colormap='plasma', data_range=None):
    """
    Colorize a data map (depth, confidence, etc.) for visualization and optionally save it.

    Args:
        data_map / confidence_map: (H, W) numpy array with data values
        colormap: matplotlib colormap name (default: 'plasma')
        data_range: optional tuple (min_value, max_value) for consistent scaling across multiple maps
    Returns:
        colorized_image: (H, W, 3) RGB array of colorized data map
    """
    # Handle case where data map is all zeros
    if np.max(data_map) == 0:
        # Create a black image for zero values
        colorized = np.zeros((data_map.shape[0], data_map.shape[1], 3), dtype=np.uint8)
        return colorized

    # Determine data range for normalization
    valid_mask = data_map > 0

    # Create normalized data map
    normalized_data = np.zeros_like(data_map, dtype=np.float32)

    if np.any(valid_mask):
        if data_range is not None:
            min_value, max_value = data_range
        else:
            min_value = np.min(data_map[valid_mask])
            max_value = np.max(data_map[valid_mask])
        if max_value > min_value:
            normalized_data[valid_mask] = np.clip((data_map[valid_mask].astype(np.float32) - min_value) / (max_value - min_value), 0, 1)
        else:
            normalized_data[valid_mask] = 1.0

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colorized = cmap(normalized_data)

    # Set invalid pixels to black
    colorized[~valid_mask] = [0, 0, 0, 1]

    # Convert to 8-bit RGB
    colorized_rgb = (colorized[:, :, :3] * 255).astype(np.uint8)

    return colorized_rgb

def save_point_cloud(pts, colors, save_path):
    print(f"Saving point cloud to {save_path}")

    # Validate inputs
    if len(pts) == 0:
        print(f"Warning: Empty point cloud, skipping save to {save_path}")
        return

    if len(pts) != len(colors):
        raise ValueError(f"Points and colors must have same length: pts={len(pts)}, colors={len(colors)}")

    # Check for NaN or Inf values
    pts_valid = np.isfinite(pts).all(axis=1)
    colors_valid = np.isfinite(colors).all(axis=1)
    valid_mask = pts_valid & colors_valid

    if not valid_mask.all():
        num_invalid = (~valid_mask).sum()
        print(f"Warning: Removing {num_invalid} invalid points (NaN/Inf) from point cloud")
        pts = pts[valid_mask]
        colors = colors[valid_mask]

    if len(pts) == 0:
        print(f"Warning: No valid points after filtering, skipping save to {save_path}")
        return

    # Ensure colors are in valid range [0, 1]
    colors = np.clip(colors, 0.0, 1.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Successfully saved {len(pts)} points")

def subsample_point_cloud(pts, colors, num_points):
    if len(pts) < num_points:
        return pts, colors
    indices = np.random.choice(len(pts), num_points, replace=False)
    return pts[indices], colors[indices]

def compute_consistent_map_info(points_3d: np.ndarray, valid_mask: np.ndarray, depth_map: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray, consistency_threshold: float) -> np.ndarray:
    """
    Checks the consistency of the point cloud with a depthmap given intrinsics and pose.

    Args:
        points_3d: HxWx3 array of 3D points in world coordinates
        valid_mask: HxW boolean mask indicating valid points
        depthmap: HxW numpy array
        intrinsics: 3x3 numpy array
        pose: 4x4 numpy array (cam_from_world)
        consistency_threshold: float, percentage of depth difference allowed for consistency
    Returns:
        consistency_map: HxWx3 array containing [u, v, depth] where u,v are partner image coordinates (all 0.0 = invalid)
    """
    # Initialize consistency map (default to invalid [u, v, depth])
    consistency_map = np.zeros_like(points_3d, dtype=np.float32)
    if depth_map is None:
        return consistency_map

    # Extract valid 3D points and their original coordinates
    valid_points_3d = points_3d[valid_mask]  # Shape: (N, 3)
    if len(valid_points_3d) == 0:
        return consistency_map

    # Get original pixel coordinates of valid points
    valid_coords = np.where(valid_mask)  # (y_coords, x_coords)
    valid_y, valid_x = valid_coords[0], valid_coords[1]

    # Transform 3D points to partner camera coordinates
    points_3d_homo = np.hstack([valid_points_3d, np.ones((len(valid_points_3d), 1))])
    cam_coords = (pose @ points_3d_homo.T).T[:, :3]  # (N, 3)

    # Filter points behind camera
    valid_depth_mask = cam_coords[:, 2] > 0
    if not np.any(valid_depth_mask):
        return consistency_map

    # Keep only points with valid depth
    cam_coords = cam_coords[valid_depth_mask]
    original_y = valid_y[valid_depth_mask]
    original_x = valid_x[valid_depth_mask]

    # Project to image coordinates
    proj_coords = (intrinsics @ cam_coords.T).T
    proj_coords = proj_coords / proj_coords[:, 2:3]  # Normalize by depth

    # Get pixel coordinates and depths
    partner_pixel_x = proj_coords[:, 0].astype(int)
    partner_pixel_y = proj_coords[:, 1].astype(int)
    projected_depths = cam_coords[:, 2]

    H,W = depth_map.shape

    # Filter points within image bounds
    in_bounds_mask = (
        (partner_pixel_x >= 0) & (partner_pixel_x < W) &
        (partner_pixel_y >= 0) & (partner_pixel_y < H)
    )

    if not np.any(in_bounds_mask):
        return consistency_map

    # Keep only in-bounds points
    partner_pixel_x = partner_pixel_x[in_bounds_mask]
    partner_pixel_y = partner_pixel_y[in_bounds_mask]
    projected_depths = projected_depths[in_bounds_mask]
    original_y = original_y[in_bounds_mask]
    original_x = original_x[in_bounds_mask]

    # Get partner's depth values at projected locations
    partner_depths = depth_map[partner_pixel_y, partner_pixel_x]

    # Compare depths where partner has valid measurements
    valid_partner_mask = partner_depths > 0
    if not np.any(valid_partner_mask):
        return consistency_map

    # Compute consistency for valid comparisons
    valid_partner_depths = partner_depths[valid_partner_mask]
    valid_projected_depths = projected_depths[valid_partner_mask]
    valid_original_y = original_y[valid_partner_mask]
    valid_original_x = original_x[valid_partner_mask]
    valid_partner_pixel_x = partner_pixel_x[valid_partner_mask]
    valid_partner_pixel_y = partner_pixel_y[valid_partner_mask]

    # Compute relative depth difference
    depth_diff = np.abs(valid_partner_depths - valid_projected_depths) / np.maximum(valid_projected_depths, 1e-6)

    # Check consistency threshold
    is_consistent = depth_diff < consistency_threshold

    # Set [u, v, depth] for consistent points, [0, 0, 0] for inconsistent
    consistent_u = np.where(is_consistent, valid_partner_pixel_x, 0.0)
    consistent_v = np.where(is_consistent, valid_partner_pixel_y, 0.0)
    consistent_depths = np.where(is_consistent, valid_partner_depths, 0.0)

    consistency_map[valid_original_y, valid_original_x, 0] = consistent_u
    consistency_map[valid_original_y, valid_original_x, 1] = consistent_v
    consistency_map[valid_original_y, valid_original_x, 2] = consistent_depths

    return consistency_map

def triangulate_matches(P_i, P_j, matches: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate matches using OpenCV (vectorized).

    Args:
        P_i: 3x4 camera projection matrix of the first image
        P_j: 3x4 camera projection matrix of the second image
        matches: N x 4 array of [x_i, y_i, u_j, v_j] coordinates

    Returns:
        points_3d: N x 3 array of 3D world coordinates
        z_i: N array of depth in camera i
        z_j: N array of depth in camera j
        valid_mask: N array of valid mask
    """

    if len(matches) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 1), np.array([]).reshape(0, 1), np.array([]).reshape(0, 1)

    # Extract 2D points from matches
    # cv2.triangulatePoints expects 2xN arrays
    points_i = matches[:, :2].T  # Shape: (2, N)
    points_j = matches[:, 2:].T  # Shape: (2, N)

    # Triangulate all points at once (vectorized)
    # Returns 4xN array of homogeneous coordinates
    points_4d = cv2.triangulatePoints(P_i, P_j, points_i, points_j)

    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]  # Shape: (3, N)
    points_3d = points_3d.T  # Shape: (N, 3)

    # Filter out points behind either camera
    # Check depth in camera coordinates (third row of projection matrix)
    points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # Shape: (N, 4)
    z_i = (P_i[2:3] @ points_homo.T).flatten()  # Depth in camera i
    z_j = (P_j[2:3] @ points_homo.T).flatten()  # Depth in camera j

    valid_mask = (z_i > 0) & (z_j > 0) & (np.abs(points_4d[3]) > 1e-6)

    return points_3d, z_i, z_j, valid_mask


def project_pts3d_to_image(pts3d, intrinsics, pose):
    """
    Project 3D points to image plane.

    Args:
        pts3d: N x 3 array of 3D points in world coordinates
        intrinsics: 3x3 intrinsics matrix
        pose: 4x4 pose matrix (cam_from_world)

    Returns:
        uvd: N x 3 array where each row is [u, v, d]
             u, v are pixel coordinates, d is depth
    """
    # Convert to homogeneous coordinates
    pts3d_homo = np.hstack([pts3d, np.ones((len(pts3d), 1))])  # Shape: (N, 4)

    # Project to camera coordinates
    P = intrinsics @ pose[:3, :]
    projected = (P @ pts3d_homo.T).T  # Shape: (N, 3) -> [x', y', z']

    # Extract depth before normalization
    d = projected[:, 2:3]  # Shape: (N, 1)

    # Normalize to get pixel coordinates
    uv = projected[:, :2] / d  # Shape: (N, 2)

    # Combine into [u, v, d]
    uvd = np.hstack([uv, d])  # Shape: (N, 3)

    return uvd

def apply_transform(src_pts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a similarity transform to a point cloud.
    The similarity transform maps src_pts to dst_pts: dst_pts = s*R*src_pts + t
    Args:
        src_pts: (N, 3) array of source points
        transform: (4, 4) similarity transformation matrix. Apply this transformation to src_pts to get dst_pts.
    Returns:
        dst_pts: (N, 3) array of destination points
    """
    if len(src_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Check for invalid input points
    if not np.isfinite(src_pts).all():
        print("Warning: Input points contain NaN/Inf values")
        valid_mask = np.isfinite(src_pts).all(axis=1)
        print(f"  {(~valid_mask).sum()} invalid points out of {len(src_pts)}")

    # Check for invalid transform
    if not np.isfinite(transform).all():
        print("Warning: Transform matrix contains NaN/Inf values")
        print(f"Transform:\n{transform}")
        raise ValueError("Invalid transform matrix")

    src_pts_homo = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    dst_pts_homo = (transform @ src_pts_homo.T).T
    dst_pts = dst_pts_homo[:, :3].copy()

    # Check output for issues
    if not np.isfinite(dst_pts).all():
        print("Warning: Transformed points contain NaN/Inf values")
        valid_mask = np.isfinite(dst_pts).all(axis=1)
        print(f"  {(~valid_mask).sum()} invalid points out of {len(dst_pts)}")

    return dst_pts

def compute_robust_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray, max_iterations: int = 1000, inlier_threshold: float = 0.1) -> np.ndarray:
    """
    Compute a robust similarity transform from source points to destination points.

    Uses RANSAC to robustly estimate rotation, translation, and uniform scale.
    The similarity transform maps src_pts to dst_pts: dst_pts = s*R*src_pts + t

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points
        max_iterations: maximum RANSAC iterations
        inlier_threshold: distance threshold for considering a point as an inlier

    Returns:
        transform: (4, 4) similarity transformation matrix. Apply this transformation to src_pts to get dst_pts.
    """
    assert len(src_pts) == len(dst_pts), "Point clouds must have same length"
    assert len(src_pts) >= 3, "Need at least 3 points to compute similarity transform"

    # Check for NaN/Inf in inputs
    if not np.isfinite(src_pts).all():
        print("Error: src_pts contains NaN/Inf values")
        return np.eye(4)
    if not np.isfinite(dst_pts).all():
        print("Error: dst_pts contains NaN/Inf values")
        return np.eye(4)

    N = len(src_pts)
    print(f"Computing robust similarity transform for {N} point pairs")
    best_inliers = 0
    best_transform = np.eye(4)

    # RANSAC loop
    for iteration in range(max_iterations):
        # Sample minimum number of points (3 for similarity transform)
        sample_indices = np.random.choice(N, size=3, replace=False)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        # Compute similarity transform from sample
        transform = compute_similarity_transform(src_sample, dst_sample)

        # Transform all source points
        transformed_pts = apply_transform(src_pts, transform)

        # Compute distances and count inliers
        distances = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        inliers = distances < inlier_threshold
        num_inliers = np.sum(inliers)

        # Update best model if this is better
        if num_inliers > best_inliers:
            best_inliers = num_inliers

            # Refine using all inliers
            if num_inliers >= 3:
                best_transform = compute_similarity_transform(
                    src_pts[inliers],
                    dst_pts[inliers]
                )
            else:
                best_transform = transform

            # Early termination if we have very good fit
            if num_inliers > 0.98 * N:
                break

    print(f"RANSAC completed: best inliers = {best_inliers}/{N} ({100*best_inliers/N:.1f}%)")

    # Verify the final transform is valid
    if not np.isfinite(best_transform).all():
        print("Error: Final transform contains NaN/Inf")
        return np.eye(4)

    return best_transform


def compute_robust_affine_transform_nonuniform_scale(src_pts: np.ndarray, dst_pts: np.ndarray, max_iterations: int = 1000, inlier_threshold: float = 0.1) -> np.ndarray:
    """
    Compute a robust affine transform with non-uniform scale using RANSAC.

    Uses RANSAC to robustly estimate rotation, translation, and per-axis scaling.
    The transform is: dst = R @ S @ src + t, where S is a diagonal scale matrix.

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points
        max_iterations: maximum RANSAC iterations
        inlier_threshold: distance threshold for considering a point as an inlier

    Returns:
        transform: (4, 4) affine transformation matrix with non-uniform scale
    """
    assert len(src_pts) == len(dst_pts), "Point clouds must have same length"
    assert len(src_pts) >= 3, "Need at least 3 points to compute affine transform"

    # Check for NaN/Inf in inputs
    if not np.isfinite(src_pts).all():
        print("Error: src_pts contains NaN/Inf values")
        return np.eye(4)
    if not np.isfinite(dst_pts).all():
        print("Error: dst_pts contains NaN/Inf values")
        return np.eye(4)

    N = len(src_pts)
    print(f"Computing robust affine transform (non-uniform scale) for {N} point pairs")
    best_inliers = 0
    best_transform = np.eye(4)

    # RANSAC loop
    for iteration in range(max_iterations):
        # Sample minimum number of points (3 for affine transform)
        sample_indices = np.random.choice(N, size=3, replace=False)
        src_sample = src_pts[sample_indices]
        dst_sample = dst_pts[sample_indices]

        # Compute affine transform from sample
        transform = compute_affine_transform_nonuniform_scale(src_sample, dst_sample)

        # Transform all source points
        transformed_pts = apply_transform(src_pts, transform)

        # Compute distances and count inliers
        distances = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        inliers = distances < inlier_threshold
        num_inliers = np.sum(inliers)

        # Update best model if this is better
        if num_inliers > best_inliers:
            best_inliers = num_inliers

            # Refine using all inliers
            if num_inliers >= 3:
                best_transform = compute_affine_transform_nonuniform_scale(
                    src_pts[inliers],
                    dst_pts[inliers]
                )
            else:
                best_transform = transform

            transformed_pts = apply_transform(src_pts, transform)
            distances = np.linalg.norm(transformed_pts - dst_pts, axis=1)
            inliers = distances < inlier_threshold
            num_inliers = np.sum(inliers)

            # Early termination if we have very good fit
            if num_inliers > 0.9 * N:
                break

    print(f"RANSAC completed: best inliers = {best_inliers}/{N} ({100*best_inliers/N:.1f}%)")

    # Verify the final transform is valid
    if not np.isfinite(best_transform).all():
        print("Error: Final transform contains NaN/Inf")
        return np.eye(4)

    return best_transform


def compute_affine_transform_nonuniform_scale(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute affine transform with non-uniform scale (different scale per axis).

    This computes rotation, translation, and per-axis scaling to align src_pts to dst_pts.
    The transform is: dst = R @ S @ src + t, where S is a diagonal scale matrix.

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points

    Returns:
        transform: (4, 4) affine transformation matrix
    """
    # Check for degenerate inputs
    if len(src_pts) < 3 or len(dst_pts) < 3:
        print(f"Warning: Not enough points for affine transform (need >= 3, got {len(src_pts)})")
        return np.eye(4)

    # Check for NaN/Inf in inputs
    if not np.isfinite(src_pts).all() or not np.isfinite(dst_pts).all():
        print(f"Warning: Input points contain NaN/Inf in compute_affine_transform_nonuniform_scale")
        return np.eye(4)

    # Compute centroids
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    # Check for NaN in centroids
    if not np.isfinite(src_centroid).all() or not np.isfinite(dst_centroid).all():
        print(f"Warning: Centroids contain NaN/Inf")
        return np.eye(4)

    # Center the point clouds
    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    # Compute covariance matrix
    H = src_centered.T @ dst_centered

    # Check for NaN in H
    if not np.isfinite(H).all():
        print(f"Warning: Covariance matrix contains NaN/Inf")
        return np.eye(4)

    # Compute rotation using SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Verify R is a valid rotation matrix
    if not np.isfinite(R).all():
        print(f"Warning: Rotation matrix contains NaN/Inf")
        return np.eye(4)

    # Rotate source points to align with destination
    src_rotated = (R @ src_centered.T).T

    # Compute per-axis scale
    # For each axis, compute the ratio of standard deviations
    scales = np.zeros(3)
    for i in range(3):
        src_std = np.std(src_rotated[:, i])
        dst_std = np.std(dst_centered[:, i])

        if src_std < 1e-10:
            scales[i] = 1.0
        else:
            scales[i] = dst_std / src_std

    # Check for unreasonable scales
    for i in range(3):
        if not np.isfinite(scales[i]) or scales[i] <= 0 or scales[i] > 1e6:
            print(f"Warning: Unreasonable scale factor on axis {i}: {scales[i]}")
            scales[i] = 1.0

    # Build scale matrix
    S_matrix = np.diag(scales)

    # Compute translation
    # t = dst_centroid - R @ S @ src_centroid
    t = dst_centroid - R @ S_matrix @ src_centroid

    # Check translation
    if not np.isfinite(t).all():
        print(f"Warning: Translation vector contains NaN/Inf")
        return np.eye(4)

    # Build 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R @ S_matrix
    transform[:3, 3] = t

    # Print scale factors for debugging
    print(f"Non-uniform scale factors: x={scales[0]:.4f}, y={scales[1]:.4f}, z={scales[2]:.4f}")

    return transform


def compute_scale_shift_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute scale and shift transform (uniform scale, translation) using closed-form solution.

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points

    Returns:
        transform: (4, 4) similarity transformation matrix
    """
    # Check for degenerate inputs
    if len(src_pts) < 3 or len(dst_pts) < 3:
        print(f"Warning: Not enough points for scale and shift transform (need >= 3, got {len(src_pts)})")
        return np.eye(4)

    # Compute centroids
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    # Center the point clouds
    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    # Compute scale
    src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))

    # Check for degenerate point clouds (all points at same location)
    if src_scale < 1e-10 and dst_scale < 1e-10:
        # Both point clouds are degenerate, just translate
        transform = np.eye(4)
        transform[:3, 3] = dst_centroid - src_centroid
        return transform
    elif src_scale < 1e-10:
        print(f"Warning: Source point cloud is degenerate (scale={src_scale})")
        scale = 1.0
    else:
        scale = dst_scale / src_scale

    # Normalize for rotation computation
    src_normalized = src_centered / (src_scale + 1e-10)
    dst_normalized = dst_centered / (dst_scale + 1e-10)

    # Compute translation
    t = dst_centroid - scale * src_centroid

    # Build 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = np.eye(3) * scale
    transform[:3, 3] = t

    return transform


def compute_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    Compute similarity transform (rotation, translation, uniform scale) using closed-form solution.
    Based on Umeyama's method.

    Args:
        src_pts: (N, 3) array of source points
        dst_pts: (N, 3) array of destination points

    Returns:
        transform: (4, 4) similarity transformation matrix
    """
    # Check for degenerate inputs
    if len(src_pts) < 3 or len(dst_pts) < 3:
        print(f"Warning: Not enough points for similarity transform (need >= 3, got {len(src_pts)})")
        return np.eye(4)

    # Compute centroids
    src_centroid = np.mean(src_pts, axis=0)
    dst_centroid = np.mean(dst_pts, axis=0)

    # Center the point clouds
    src_centered = src_pts - src_centroid
    dst_centered = dst_pts - dst_centroid

    # Compute scale
    src_scale = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
    dst_scale = np.sqrt(np.mean(np.sum(dst_centered**2, axis=1)))

    # Check for degenerate point clouds (all points at same location)
    if src_scale < 1e-10 and dst_scale < 1e-10:
        # Both point clouds are degenerate, just translate
        transform = np.eye(4)
        transform[:3, 3] = dst_centroid - src_centroid
        return transform
    elif src_scale < 1e-10:
        print(f"Warning: Source point cloud is degenerate (scale={src_scale})")
        scale = 1.0
    else:
        scale = dst_scale / src_scale

    # Normalize for rotation computation
    src_normalized = src_centered / (src_scale + 1e-10)
    dst_normalized = dst_centered / (dst_scale + 1e-10)

    # Compute rotation using SVD
    H = src_normalized.T @ dst_normalized

    # Check for NaN in H
    if not np.isfinite(H).all():
        print(f"Warning: H matrix contains NaN/Inf")
        return np.eye(4)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Verify R is a valid rotation matrix
    if not np.isfinite(R).all():
        print(f"Warning: Rotation matrix contains NaN/Inf")
        return np.eye(4)

    # Compute translation
    t = dst_centroid - scale * R @ src_centroid

    # Build 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = scale * R
    transform[:3, 3] = t

    return transform

def merge_point_clouds(ply_files: List[str], output_path: str, max_points: int = 10000000):
    # merge all ply files into a single ply file
    pcds = [o3d.io.read_point_cloud(ply_file) for ply_file in ply_files]
    merged_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged_pcd.points.extend(pcd.points)
        merged_pcd.colors.extend(pcd.colors)

    # merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)

    if len(merged_pcd.points) > max_points:
        sampling_ratio = max_points / len(merged_pcd.points)
        merged_pcd = merged_pcd.random_down_sample(sampling_ratio=sampling_ratio)

    o3d.io.write_point_cloud(output_path, merged_pcd)

def compute_robust_depth_scale_and_shift(
    src_depth_samples: np.ndarray,
    dst_depth_samples: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 0.02,
    early_exit_ratio: float = 0.95,
    inlier_count_th: int = 50,
    shift_th: float = 0.4,
) -> tuple[float, float, int]:
    """
    Compute robust scale and shift from source depth samples to destination depth samples.
    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
        max_iterations: maximum RANSAC iterations
        inlier_threshold: distance threshold for considering a point as an inlier
        early_exit_ratio: ratio of inliers to exit RANSAC
        inlier_count_th: minimum number of inliers to consider a model valid
        shift_th: maximum shift to consider a model valid
    Returns:
        scale: float
        shift: float
        inliers: int
    """

    N = len(src_depth_samples)
    if N != len(dst_depth_samples):
        raise ValueError(
            f"⚠️ Source and destination depth samples must have the same length: {N} != {len(dst_depth_samples)}"
        )
    if N < inlier_count_th:
        raise ValueError(f"⚠️ Need at least {inlier_count_th} points to compute scale and shift transform")

    # Filter invalid/degenerate samples up front.
    finite_mask = np.isfinite(src_depth_samples) & np.isfinite(dst_depth_samples)
    if not finite_mask.all():
        print("Depth samples contain NaN/Inf values; filtering invalid samples")
    src_depth_samples = src_depth_samples[finite_mask]
    dst_depth_samples = dst_depth_samples[finite_mask]

    positive_mask = (src_depth_samples > 1e-6) & (dst_depth_samples > 1e-6)
    if not positive_mask.all():
        print("Depth samples contain non-positive values; filtering invalid samples")
    src_depth_samples = src_depth_samples[positive_mask]
    dst_depth_samples = dst_depth_samples[positive_mask]

    N = len(src_depth_samples)
    if N < inlier_count_th:
        return 1.0, 0.0, 0

    best_inliers = 0
    best_scale = 1.0
    best_shift = 0.0

    inlier_set: list[int] = []

    # RANSAC loop
    min_src_std = 1e-6
    for iteration in range(max_iterations):
        if len(inlier_set) > 0.1 * N and iteration < 100:
            sample_indices = np.random.choice(inlier_set, size=3, replace=False)
        else:
            sample_indices = np.random.choice(N, size=3, replace=False)

        src_sample = src_depth_samples[sample_indices]
        dst_sample = dst_depth_samples[sample_indices]
        if np.std(src_sample) < min_src_std:
            continue

        # Compute scale and shift from sample
        scale, shift = compute_depth_scale_and_shift(src_sample, dst_sample)
        if abs(shift) > shift_th:
            continue

        # Transform all source depth samples
        transformed_depth_samples = scale * src_depth_samples + shift

        # compute the error in terms of abs percentage change wrt destination depth samples
        denom = np.maximum(np.abs(dst_depth_samples), 1e-3)
        error = np.abs(transformed_depth_samples - dst_depth_samples) / denom
        inliers = error < inlier_threshold
        num_inliers = np.sum(inliers)
        if num_inliers <= 3:
            continue

        # Update best model if this is better
        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_scale, best_shift = scale, shift
            inlier_set = [k for k in range(N) if inliers[k]]

            # Refine using all inliers
            best_scale, best_shift = compute_depth_scale_and_shift(
                src_depth_samples[inliers], dst_depth_samples[inliers]
            )
            transformed_depth_samples = best_scale * src_depth_samples + best_shift
            denom = np.maximum(np.abs(dst_depth_samples), 1e-3)
            error = np.abs(transformed_depth_samples - dst_depth_samples) / denom
            inliers = error < inlier_threshold
            num_inliers = np.sum(inliers)
            best_inliers = num_inliers
            inlier_set = [k for k in range(N) if inliers[k]]

        if best_inliers > early_exit_ratio * N:
            break

    if best_inliers < inlier_count_th or not np.isfinite(best_scale) or not np.isfinite(best_shift):
        fallback_scale, fallback_shift, fallback_inliers = compute_robust_depth_scale_and_shift_fallback(
            src_depth_samples, dst_depth_samples
        )
        return fallback_scale, fallback_shift, fallback_inliers

    return best_scale, best_shift, int(best_inliers)

def compute_depth_scale_and_shift(src_depth_samples: np.ndarray, dst_depth_samples: np.ndarray) -> tuple[float, float]:
    """
    Compute scale and shift from source depth samples to destination depth samples. The transform maps src_depth_samples
    to dst_depth_samples: dst_depth_samples = scale * src_depth_samples + shift

    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
    Returns:
        scale: float
        shift: float
    """

    if len(src_depth_samples) != len(dst_depth_samples):
        raise ValueError("Source and destination depth samples must have the same length")
    if len(src_depth_samples) < 3 or len(dst_depth_samples) < 3:
        raise ValueError("Need at least 3 points to compute scale and shift transform")

    # Compute centroids
    src_centroid = np.mean(src_depth_samples)
    dst_centroid = np.mean(dst_depth_samples)

    # Center the depth samples
    src_centered = src_depth_samples - src_centroid
    dst_centered = dst_depth_samples - dst_centroid
    src_centered = src_centered.reshape(-1, 1)
    dst_centered = dst_centered.reshape(-1, 1)

    # Compute scale
    scale = np.sqrt(np.mean(dst_centered**2, axis=0)) / np.sqrt(np.mean(src_centered**2, axis=0))[0]

    # Compute translation
    shift = dst_centroid - scale * src_centroid

    return float(scale[0]), float(shift[0])


def compute_robust_depth_scale_and_shift_fallback(
    src_depth_samples: np.ndarray,
    dst_depth_samples: np.ndarray,
    trim_ratio: float = 0.1,
) -> tuple[float, float, int]:
    """Fallback method for robust scale and shift estimation.

    dst_depth_samples = scale * src_depth_samples + shift

    The method:
    1) Estimate per-sample scale and shift using median-centered samples.
    2) Sort scales and shifts, trim outliers.
    3) Compute a single robust scale and shift from inliers.

    Args:
        src_depth_samples: Nx1 array of source depth samples
        dst_depth_samples: Nx1 array of destination depth samples
        trim_ratio: ratio of outliers to trim
    Returns:
        scale: float
        shift: float
        inliers: int
    """
    N = len(src_depth_samples)
    if N != len(dst_depth_samples):
        raise ValueError("Source and destination depth samples must have the same length")
    if N < 3:
        raise ValueError("Need at least 3 points to compute scale and shift transform")

    # Check for NaN/Inf in inputs
    valid_mask = np.isfinite(src_depth_samples) & np.isfinite(dst_depth_samples)
    if not valid_mask.all():
        logger.warning("Depth samples contain NaN/Inf values; filtering invalid samples for robust estimation")
    src = src_depth_samples[valid_mask].astype(np.float64)
    dst = dst_depth_samples[valid_mask].astype(np.float64)
    if len(src) < 3:
        return 1.0, 0.0, 0

    # Median-center to get per-sample scale/shift estimates.
    src_med = np.median(src)
    dst_med = np.median(dst)
    src_delta = src - src_med
    dst_delta = dst - dst_med

    # Avoid division by tiny deltas.
    delta_mask = np.abs(src_delta) > 1e-6
    src_valid = src[delta_mask]
    dst_valid = dst[delta_mask]
    if len(src_valid) < 3:
        return 1.0, 0.0, 0

    scale_candidates = dst_delta[delta_mask] / src_delta[delta_mask]
    shift_candidates = dst_valid - scale_candidates * src_valid

    # Sort scales and shifts and trim outliers.
    scales_sorted = np.sort(scale_candidates)
    shifts_sorted = np.sort(shift_candidates)
    lower_idx = int(trim_ratio * (len(scales_sorted) - 1))
    upper_idx = int((1.0 - trim_ratio) * (len(scales_sorted) - 1))
    scale_low, scale_high = scales_sorted[lower_idx], scales_sorted[upper_idx]
    shift_low, shift_high = shifts_sorted[lower_idx], shifts_sorted[upper_idx]

    inlier_mask = (
        (scale_candidates >= scale_low)
        & (scale_candidates <= scale_high)
        & (shift_candidates >= shift_low)
        & (shift_candidates <= shift_high)
    )
    if np.sum(inlier_mask) < 3:
        # Fall back to median-based estimates if trimming is too aggressive.
        robust_scale = float(np.median(scale_candidates))
        robust_shift = float(np.median(dst_valid - robust_scale * src_valid))
        return robust_scale, robust_shift, int(len(scale_candidates))

    src_inliers = src_valid[inlier_mask]
    dst_inliers = dst_valid[inlier_mask]

    # Robust final estimates: median scale then median shift on inliers.
    robust_scale = float(np.median(scale_candidates[inlier_mask]))
    robust_shift = float(np.median(dst_inliers - robust_scale * src_inliers))
    return robust_scale, robust_shift, int(np.sum(inlier_mask))

