
import os
from pathlib import Path
import numpy as np
import argparse
import glob
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import cv2
from PIL import Image
from tqdm import tqdm

from threedn_depth_data import ThreednDepthData
from colmap_utils import ColmapReconstruction, build_image_id_mapping, compute_image_depthmap, find_exact_image_match_from_extrinsics

import geometric_utility
from geometric_utility import depthmap_to_world_frame, colorize_heatmap, save_point_cloud, compute_depthmap, uvd_to_world_frame, compute_consistent_map_info

from image_utils import load_and_preprocess_image_keep_aspect, load_and_preprocess_image_square, load_and_preprocess_image_crop, load_and_preprocess_image_center_crop, load_and_preprocess_image_scale_then_crop
from image_utils import load_and_preprocess_image_keep_aspect_14_multiple, load_and_preprocess_image_square_padding, load_and_preprocess_image, load_and_resize_keep_aspect

from typing import List, Callable, Optional, Tuple

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class ParallelExecutor:
    """
    Generic parallel executor for running functions with image IDs in parallel.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker threads. If None, uses CPU count.
        """
        self.max_workers = max_workers

    def run_in_parallel(self, function, item_list: List,
                       progress_desc: str = "Processing",
                       max_workers: Optional[int] = None, **kwargs) -> List:
        """
        Execute a function in parallel for each item.

        Args:
            function: Function to execute. Should accept (item, **kwargs) as arguments.
            item_list: List of items to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            List of results from the function calls (in order of completion).
        """
        if not item_list:
            return []

        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(item_list), os.cpu_count() or 1)

        print(f"    {progress_desc}: {len(item_list)} items using {workers} workers...")

        results = []

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(function, item, **kwargs): item
                for item in item_list
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(item_list), desc=progress_desc, unit="item") as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Processing item {item} generated an exception: {exc}')
                        results.append(None)  # Add None for failed items
                        pbar.update(1)

        return results

    def run_in_parallel_no_return(self, function, item_list: List,
                                 progress_desc: str = "Processing",
                                 max_workers: Optional[int] = None, **kwargs) -> None:
        """
        Execute a function in parallel for each item without collecting results.
        More memory efficient when you don't need the return values.

        Args:
            function: Function to execute. Should accept (item, **kwargs) as arguments.
            item_list: List of items to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.
        """
        if not item_list:
            return

        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(item_list), os.cpu_count() or 1)

        print(f"    {progress_desc}: {len(item_list)} items using {workers} workers...")

        import time
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(function, item, **kwargs): item
                for item in item_list
            }

            print(f"    All {len(item_list)} tasks submitted, waiting for completion...")

            # Process completed tasks with progress bar
            with tqdm(total=len(item_list), desc=progress_desc, unit="item",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        future.result()  # Don't store the result
                        pbar.update(1)
                    except Exception as exc:
                        print(f'Processing item {item} generated an exception: {exc}')
                        pbar.update(1)

        elapsed = time.time() - start_time
        print(f"    Completed {progress_desc} in {elapsed:.2f} seconds ({elapsed/len(item_list):.2f}s per item)")

class DensificationConfig:
    """Configuration for densification process."""
    def __init__(self):
        self.scene_folder: str = ""
        self.colmap_folder: Optional[str] = None
        self.output_folder: str = ""
        self.image_load_size: int = 1024
        self.target_h: int = 518
        self.target_w = 518
        self.prior_type: str = 'none' # 'none', 'dmaps', 'sfm', 'reference'
        self.reference_reconstruction: Optional[ColmapReconstruction] = None
        self.min_track_length: int = 3
        self.memory_efficient_inference: bool = True
        self.export_resolution: int = 0
        self.conf_threshold: float = 0.0
        self.max_points: int = 1000000
        self.verbose: bool = False
        self.apache: bool = False
        self.smart_batching: bool = False
        self.sequential_batching: bool = True
        self.batch_size: int = 4
        self.seed: int = 42
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = torch.float32
        self.folder_based_filtering: bool = False # enable this to define neighboring images based on folder structure

        # parameters for vggt tracking
        self.vggt_num_track_frames: int = 4 # number of frames to track
        self.vggt_max_tracks_per_frame: int = 1000
        self.vggt_grid_spacing: int = 10
        self.vggt_min_track_length: int = 3 # minimum length of track to be considered valid
        self.vggt_min_confidence: float = 0.5
        self.vggt_min_visibility: float = 0.3

        # parameters for fusion
        self.run_fusion: bool = True
        self.fusion_max_partners: int = 8
        self.fusion_min_points: int = 10
        self.fusion_consistency_threshold: float = 0.05
        self.fusion_min_consistency_count: int = 2

        self.export_margin: int = 10 # margin in pixels to ignore for exporting points from the image

        # runner for depth completion
        self.run_depth_completion: Optional[Callable] = None

        self.image_mode: str = 'vggt' # 'vggt', 'square', 'keep_aspect', 'mapanything', 'configured'

        # world-mirror specific parameters
        self.fix_scale = False
        self.align_depth_to_prior = False
        self.export_predicted_raw_cloud = False

        self.apply_confidence_mask = True
        self.apply_edge_mask = True
        self.confidence_percentile = 10
        self.confidence_threshold = 5.0
        self.edge_normal_threshold = 5.0
        self.edge_depth_threshold = 0.03

        self.alignment_inlier_threshold: float = 0.1
        self.alignment_min_points: int = 4

        self.init_neighbours: bool = True


    def parse_args(self) -> None:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="Generate point cloud using MapAnything with COLMAP calibrations. Expects sparse/ and images/ folders under the scene_folder. requires dmaps under threedn/ folder if using dmaps prior"
        )
        parser.add_argument(
            "-sf", "--scene_folder",
            type=str,
            required=True,
            help="Scene folder containing 'images' and 'sparse' subdirectories",
        )
        parser.add_argument(
            "--colmap_folder",
            type=str,
            default="sparse_export",
            help="Colmap folder containing 'sparse' subdirectory",
        )
        parser.add_argument(
            "-ff", "--enable-folder-filter",
            action="store_true",
            default=False,
            help="Enable folder based filtering for similar images (default: False) -> use for pano images"
        )
        parser.add_argument(
            "-o", "--output_folder",
            type=str,
            default="output",
            help="Output folder for results (default: scene_folder/output/)",
        )
        parser.add_argument(
            "-s", "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility"
        )
        parser.add_argument(
            "-d", "--device",
            type=str,
            default=None,
            help="Device to use for inference (default: cuda if available, else cpu)",
        )
        parser.add_argument(
            "-m", "--memory_efficient_inference",
            action="store_true",
            default=False,
            help="Use memory efficient inference for reconstruction (trades off speed)",
        )
        parser.add_argument(
            "-r", "--resolution",
            type=int,
            default=840,
            help="Resolution for image processing (default: 518)",
        )
        parser.add_argument(
            "-e", "--export_resolution",
            type=int,
            default=0,
            help="Resolution to export 3dn depthmap format (default: 0 -> disables export)",
        )
        parser.add_argument(
            "-c", "--conf_threshold",
            type=float,
            default=0.0,
            help="Confidence threshold for depth filtering (default: 0.0)",
        )
        parser.add_argument(
            "-p", "--max_points",
            type=int,
            default=1000000,
            help="Maximum number of points in output point cloud (default: 1000000)",
        )
        parser.add_argument(
            "-b", "--batch_size",
            type=int,
            default=4,
            help="Number of images to process in each batch to manage memory usage (default: 4)",
        )
        parser.add_argument(
            "--smart_batching",
            action="store_true",
            default=False,
            help="Use COLMAP reconstruction quality for intelligent batch formation (default: True)",
        )
        parser.add_argument(
            "--sequential_batching",
            action="store_true",
            help="Use simple sequential batching instead of smart batching",
            default=True,
        )
        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            default=False,
            help="Enable verbose output and save colorized prior and predicted depth maps",
        )
        parser.add_argument("--min_track_length", "-t", type=int, default=3, help="Minimum track length for SfM depth to be used for prior depth information")

        args = parser.parse_args()

        if args.scene_folder is None or not os.path.isdir(args.scene_folder):
            raise ValueError("Scene folder is required and must exist")

        self.scene_folder = args.scene_folder
        self.colmap_folder = args.colmap_folder
        self.output_folder = os.path.join(args.scene_folder, args.output_folder)
        self.resolution = args.resolution
        self.min_track_length = args.min_track_length
        self.memory_efficient_inference = args.memory_efficient_inference
        self.export_resolution = args.export_resolution
        self.conf_threshold = args.conf_threshold
        self.max_points = args.max_points
        self.verbose = args.verbose
        self.smart_batching = args.smart_batching
        self.sequential_batching = args.sequential_batching
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.use_folder_filter = args.enable_folder_filter
        if args.device is not None:
            self.device = args.device

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == 'cpu':
            print("WARNING: running on CPU")
        if self.device == "cuda" and torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

class DensificationProblem:
    """
    Densification problem class for storing depth related data for a scene.

    for each image, in depth_data we store:

        'image_id'              : image_id,
        'image_name'            : image_name,
        'h'                      : height of the maps,
        'w'                      : width of the maps

        'image'                 : unnormalized scaled image in h x w x 3,
        'original_coords'       : array of shape (5) containing [x1, y1, x2, y2, width, height] for the image used by vggt

        'depth_map'             : depth_map in h x w or None,
        'confidence_map'        : confidence_map in h x w or None,
        'mask'                  : point mask in h x w or None, (stores with points should be considered)

        'normals'               : normal map in h x w x 3 or None,
        'normals_confidence'    : normal confidence map in h x w or None,

        'prior_depth_map'       : prior depth map in h x w or None,
        'fused_depth_map'       : fused depth map in h x w or None,
        'consistency_map'       : consistency map of the estimated depth map with the partner images map in h x w or None,

        'depth_range'           : (min_depth, max_depth) tuple for consistent scaling, or None if no valid depths,
        'confidence_range'      : (min_confidence, max_confidence) tuple for consistent scaling, or None if no valid confidences,

        'K'                     : scaled intrinsics for h x w image
        'pose'                  : 4x4 cam_from_world pose matrix for projecting world points to camera frame

        'predicted_K'           : predicted intrinsics for h x w image
        'predicted_pose'        : predicted 4x4 cam_from_world pose matrix for projecting world points to camera frame

        'partner_ids'           : list of image_ids of partner images used for fusion

    """

    def __init__(self, config: DensificationConfig):
        self.init(config)

    def init(self, config: DensificationConfig):
        """
        Initialize the depth data class.
        """
        self.config = config

        if not os.path.isdir(config.scene_folder):
            raise ValueError(f"Scene directory {config.scene_folder} does not exist")

        if config.colmap_folder is not None:
            sparse_dir = os.path.join(config.scene_folder, config.colmap_folder)
        else:
            sparse_dir = os.path.join(config.scene_folder, "sparse")
            if not os.path.isdir(sparse_dir):
                raise ValueError(f"Sparse directory {sparse_dir} does not exist")

        images_dir = os.path.join(config.scene_folder, "images")
        if not os.path.isdir(images_dir):
            raise ValueError(f"Images directory {images_dir} does not exist")

        self.reconstruction = ColmapReconstruction(sparse_dir)
        self.active_image_ids = self.reconstruction.get_all_image_ids()
        self.active_image_ids.sort()
        # self.reconstruction.compute_robust_bounding_box(min_visibility=3, padding_factor=0.5)

        self.reference_reconstruction = None
        if self.config.reference_reconstruction is not None:
            self.reference_reconstruction = ColmapReconstruction(self.config.reference_reconstruction)
            print("    Building image id mapping for reference reconstruction...")
            self.source_to_target_image_id_mapping = build_image_id_mapping(self.reconstruction, self.reference_reconstruction)
            valid_target_image_ids = self.reference_reconstruction.get_image_ids_with_valid_points()
            print(f"    Nbr of images with 3d points in reference reconstruction: {len(valid_target_image_ids)}/{self.reference_reconstruction.get_num_images()}")
            print("    Filtering active images with no 3d points in reference reconstruction...")
            # active image ids are the image ids that have a valid mapping from the source reconstruction to the reference reconstruction
            self.active_image_ids = [img_id for img_id in self.reconstruction.get_all_image_ids() if self.source_to_target_image_id_mapping[img_id] is not None and  self.source_to_target_image_id_mapping[img_id] in valid_target_image_ids]
            print(f"    Found {len(self.active_image_ids)}/{self.reconstruction.get_num_images()} active image ids")

        os.makedirs(self.config.output_folder, exist_ok=True)

        self.depth_data_folder = os.path.join(self.config.output_folder, "depth_data")
        os.makedirs(self.depth_data_folder, exist_ok=True)

        self.cloud_folder = os.path.join(self.config.output_folder, "point_clouds")
        os.makedirs(self.cloud_folder, exist_ok=True)

        self.dmap_folder = os.path.join(self.config.output_folder, "dmaps")
        os.makedirs(self.dmap_folder, exist_ok=True)

        self.tiff_folder = os.path.join(self.config.output_folder, "tiffs")
        os.makedirs(self.tiff_folder, exist_ok=True)

        self.scene_depth_data = {}        # stores all the depth data for each image
        self.parallel_executor = ParallelExecutor()  # Parallel execution helper
        self._lock = threading.Lock()  # Thread lock for safe dictionary access

    def clear(self) -> None:
        self.scene_depth_data = {}
        self.active_image_ids = self.reconstruction.get_all_image_ids()

    def initialize_image(self, image_id: int, mode: str) -> None:

        depth_data = self.get_depth_data(image_id)
        if depth_data is None:
            raise ValueError(f"Depth data for image {image_id} is not initialized")

        image_path = os.path.join(self.config.scene_folder, depth_data['image_name'])
        assert os.path.exists(image_path), f"Image {image_path} does not exist"

        if mode == 'vggt':
            target_size = round( max(self.config.target_w, self.config.target_h) / 14) * 14
            scaled_image, original_coords = load_and_preprocess_image_square_padding(image_path, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'vggt_crop':
            target_size = round( max(self.config.target_w, self.config.target_h) / 14) * 14
            scaled_image, original_coords = load_and_preprocess_image_crop(image_path, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'center_crop':
            target_size = round( max(self.config.target_w, self.config.target_h) / 14) * 14
            scaled_image, original_coords = load_and_preprocess_image_center_crop(image_path, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'scale_then_crop':
            target_size = round( max(self.config.target_w, self.config.target_h) / 14) * 14
            scaled_image, original_coords = load_and_preprocess_image_scale_then_crop(image_path, self.config.image_load_size, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'square':
            target_size = max(self.config.target_w, self.config.target_h)
            scaled_image, original_coords = load_and_preprocess_image_square(image_path, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'keep_aspect':
            target_size = max(self.config.target_w, self.config.target_h)
            scaled_image, original_coords = load_and_preprocess_image_keep_aspect(image_path, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'mapanything':
            target_size = max(self.config.target_w, self.config.target_h)
            scaled_image, original_coords = load_and_preprocess_image_keep_aspect_14_multiple(image_path, target_size)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'configured':
            scaled_image, original_coords = load_and_preprocess_image(image_path, depth_data['w'], depth_data['h'])
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        elif mode == 'ppd':
            scaled_image, original_coords = load_and_resize_keep_aspect(image_path, 1024, 768)
            depth_data['image'] = scaled_image
            depth_data['original_coords'] = original_coords
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # cv2.imwrite( os.path.join( self.config.output_folder, f"debug_scaled_image-{image_id}.png"), cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR))

        K = self.reconstruction.get_camera_calibration_matrix(image_id)
        K_updated = K.copy()

        img_w, img_h = scaled_image.shape[1], scaled_image.shape[0]

        if mode == 'center_crop': # focal length does not change
            tw = original_coords[2] - original_coords[0]
            th = original_coords[3] - original_coords[1]
            cx_shift = (original_coords[4] - tw) / 2
            cy_shift = (original_coords[5] - th) / 2
            K_updated[0, 2] -= cx_shift
            K_updated[1, 2] -= cy_shift
        elif mode == 'scale_then_crop':
            scale = self.config.image_load_size / max(original_coords[4], original_coords[5])
            scaled_w = int(original_coords[4] * scale)
            scaled_h = int(original_coords[5] * scale)
            K_updated[0, :] *= scale
            K_updated[1, :] *= scale

            tw = original_coords[2] - original_coords[0]
            th = original_coords[3] - original_coords[1]
            cx_shift = (scaled_w - tw) / 2
            cy_shift = (scaled_h - th) / 2
            K_updated[0, 2] -= cx_shift
            K_updated[1, 2] -= cy_shift

        else:
            scale_x = img_w / original_coords[4]
            scale_y = img_h / original_coords[5]
            K_updated[0, :] *= scale_x
            K_updated[1, :] *= scale_y

        depth_data['K'] = K_updated
        depth_data['w'] = img_w
        depth_data['h'] = img_h

        pose_4x4 = np.eye(4)
        pose_4x4[:3, :] = self.reconstruction.get_image_cam_from_world(image_id).matrix()
        depth_data['pose'] = pose_4x4

    def initialize_depth_data(self, image_id: int) -> None:
        if not self.reconstruction.has_image(image_id):
            raise ValueError(f"Image {image_id} not found in reconstruction")

        folder_name = os.path.dirname( self.reconstruction.get_image_name(image_id) )

        if folder_name != 'images':
            image_name = os.path.join('images', self.reconstruction.get_image_name(image_id))
        else:
            image_name = self.reconstruction.get_image_name(image_id)

        depth_data = {

            'image_id': image_id,
            'image_name': image_name,

            'w': None,
            'h': None,

            'image': None,
            'original_coords': None,

            'depth_map': None,
            'confidence_map': None,
            'mask': None,

            'normals': None,
            'normals_confidence': None,

            'prior_depth_map': None,
            'fused_depth_map': None,
            'consistency_map': None,

            'depth_range': None,
            'confidence_range': None,

            'K': None,
            'pose': None,

            'predicted_K': None,
            'predicted_pose': None,

            'partner_ids': [],
        }
        depth_data['image_path'] = os.path.join(self.config.scene_folder, depth_data['image_name'])
        assert os.path.exists(depth_data['image_path']), f"Image {depth_data['image_path']} does not exist"

        if self.config.init_neighbours:
            similar_image_ids = self.reconstruction.find_similar_images_for_image(image_id=image_id, min_points=10, use_folder_filter=self.config.use_folder_filter)
            if len(similar_image_ids) > 0:
                partner_ids = [pid for pid in similar_image_ids if pid in self.active_image_ids]
                depth_data['partner_ids'] = partner_ids[:self.config.fusion_max_partners]
            else:
                print(f"Warning: No similar images found for image {image_id} -- fusion will not be possible")

        # cam = self.reconstruction.get_image_camera(image_id)
        # K = cam.calibration_matrix()
        # img_w = cam.width
        # img_h = cam.height

        # depth_data['K'] = K
        # depth_data['w'] = img_w
        # depth_data['h'] = img_h

        # pose_4x4 = np.eye(4)
        # pose_4x4[:3, :] = self.reconstruction.get_image_cam_from_world(image_id).matrix()
        # depth_data['pose'] = pose_4x4

        with self._lock:
            self.scene_depth_data[image_id] = depth_data

    def get_depth_data(self, image_id: int) -> dict:
        assert image_id in self.scene_depth_data, f"Image {image_id} not found in scene depth data"
        return self.scene_depth_data[image_id]

    def get_active_image_ids(self) -> list:
        return self.active_image_ids

    def save_depth_data(self, image_id: int) -> None:
        depth_data = self.get_depth_data(image_id)
        img_name = Path(os.path.basename(depth_data['image_name'])).with_suffix('.npz')
        filepath = os.path.join(self.depth_data_folder, img_name)
        np.savez_compressed(filepath, **depth_data)

    def _load_depth_data_file(self, depth_data_file: str) -> dict:
        data = np.load(depth_data_file, allow_pickle=True, )
        image_id = int(data['image_id'])
        self.initialize_depth_data(image_id)
        depth_data = self.get_depth_data(image_id)
        for key in list(data.keys()):
            if data[key].ndim == 0:
                depth_data[key] = data[key].item()
            else:
                depth_data[key] = data[key]
        assert self.reconstruction.has_image(image_id)
        return depth_data

    def initialize_from_folder(self):
        self.active_image_ids = []
        depth_data_files = glob.glob(os.path.join(self.depth_data_folder, "*.npz"))
        if not depth_data_files:
            raise FileNotFoundError("No precomputed depth data found.")
        print(f"Loading {len(depth_data_files)} depth data files...")
        for df in tqdm(depth_data_files, desc="Loading depth data", unit="file"):
            depth_data = self._load_depth_data_file(df)
            image_id = depth_data['image_id']
            self.active_image_ids.append(image_id)
        self.active_image_ids.sort()

    def initialize(self, load_images: bool = True) -> None:
        print("Initializing for all active images for depth completion...")
        for img_id in self.active_image_ids:
            self.initialize_depth_data(img_id)
            # self.initialize_image(img_id, mode=self.config.image_mode)
        if not load_images:
            return
        self.parallel_executor.run_in_parallel_no_return(
            self.initialize_image,
            self.active_image_ids,
            progress_desc="Loading and Scaling Images",
            mode=self.config.image_mode
        )

    def set_priors_from_sfm(self, min_track_length) -> None:
        print("Initializing prior depth data for all active images using SfM...")
        # for img_id in self.active_image_ids:
        #     self.set_prior_depth_data_from_sfm(img_id, min_track_length=min_track_length)
        self.parallel_executor.run_in_parallel_no_return(
            self.set_prior_depth_data_from_sfm,
            self.active_image_ids,
            progress_desc="Initializing Prior Depth Data from SfM",
            min_track_length=min_track_length
        )

    def update_depth_ranges(self):
        # collect all depth ranges from all depth data
        depth_ranges = []
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_ranges.append(depth_data['depth_range'])
        depth_ranges = np.array(depth_ranges)
        depth_range = (np.min(depth_ranges[:, 0]), np.max(depth_ranges[:, 1]))

        # override all depth ranges with the collected depth range
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['depth_range'] = depth_range

    def update_confidence_ranges(self):
        cmaps = [self.get_depth_data(iid)['confidence_map'] for iid in self.active_image_ids]
        cmaps = np.array([cmap for cmap in cmaps if cmap is not None])
        cmaps = cmaps.reshape(-1)
        quantiles = [ np.quantile(cmaps, i / 100.0) for i in range(10,100,10) ]

        cmin = np.min(quantiles)
        cmax = np.max(quantiles)

        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['confidence_range'] = (cmin, cmax)

        print(f"Confidence quantiles: {quantiles}")


    def set_priors_from_reference(self) -> None:
        print("Initializing prior depth data for all active images using the reference reconstruction...")
        self.parallel_executor.run_in_parallel_no_return(
            self.set_prior_depth_data_from_reference,
            self.active_image_ids,
            progress_desc="Initializing Prior Depth Data from Reference",
            max_workers=4  # Reduce workers to avoid overwhelming system
        )

    def set_priors_from_threedn(self, keep_original_size: bool) -> None:
        print("threedn dmaps initializing...")
        threedn_dmap_files = glob.glob(os.path.join(self.config.scene_folder, 'threedn', "*.dmap"))
        self.parallel_executor.run_in_parallel_no_return(
            self.set_prior_depth_data_from_threedn,
            threedn_dmap_files,
            progress_desc="Initializing Prior Depth Data from Threedn",
            keep_original_size=keep_original_size
        )

    def get_coord_mask(self, depth_data: dict, export_margin: int = 0) -> np.ndarray:
        original_coords = depth_data['original_coords']
        coord_mask = np.zeros((depth_data['h'], depth_data['w']), dtype=bool)
        # expand area with export_margin
        xs, xe, ys, ye = original_coords[0], original_coords[2], original_coords[1], original_coords[3]
        xs += export_margin
        xe -= export_margin
        ys += export_margin
        ye -= export_margin
        coord_mask[int(ys):int(ye), int(xs):int(xe)] = True

        return coord_mask

    def extract_point_constraints(self, image_id: int, use_mask: bool = True, min_track_length: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract point constraints from the depth map.
        Args:
            image_id: ID of the image
            use_mask: Whether to use the mask
            min_track_length: Minimum track length for 3D points to include
        Returns:
            Tuple of points_3d, world_3d_points, uvd where:
                - points_3d: (N, 3) array of 3D points in world coordinates from the SFM reconstruction
                - world_3d_points: (N, 3) array of 3D points in world coordinates from the depth map corresponding to the points_3d
                - uvd: (N, 3) array of uvd coordinates from the depth map corresponding to the points_3d
        """
        depth_data = self.get_depth_data(image_id)
        depth_map = depth_data['predicted_depth_map']
        if use_mask:
            mask = depth_data['mask']
            depth_map[~mask] = 0

        coord_mask = self.get_coord_mask(depth_data)
        depth_map[~coord_mask] = 0

        points_3d, _, _ = self.reconstruction.get_visible_3d_points(image_id, min_track_length=min_track_length)

        # Convert 3D points to homogeneous coordinates
        points_3d_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])

        # Transform to camera coordinates
        cam_coords = (depth_data['pose'][:3, :] @ points_3d_homo.T).T
        projected_coords = (depth_data['K'] @ cam_coords.T).T
        projected_coords = projected_coords / projected_coords[:, 2:3]  # Normalize by z

        # Extract depth values (z-coordinates in camera frame)
        depths = projected_coords[:, 2]

        # sample the depth_map at cam_coords locations
        pix_x = projected_coords[:, 0].astype(int)
        pix_y = projected_coords[:, 1].astype(int)

        valid_point_mask = (pix_x >= 1) & (pix_x < depth_data['w']-1) & (pix_y >= 1) & (pix_y < depth_data['h']-1) & (depths > 0)

        points_3d = points_3d[valid_point_mask]
        pix_x = pix_x[valid_point_mask]
        pix_y = pix_y[valid_point_mask]

        sampled_depths = depth_map[pix_y, pix_x]

        valid_depth_mask = (sampled_depths > 0) & (sampled_depths < np.inf)
        points_3d = points_3d[valid_depth_mask]
        pix_x = pix_x[valid_depth_mask]
        pix_y = pix_y[valid_depth_mask]
        sampled_depths = sampled_depths[valid_depth_mask]

        # compute world 3d points from sampled depths
        uvd = np.stack([pix_x, pix_y, sampled_depths], axis=-1)
        world_3d_points = uvd_to_world_frame(uvd, depth_data['predicted_K'], depth_data['predicted_pose'])

        return points_3d, world_3d_points, uvd

    def set_prior_depth_data_from_sfm(self, image_id: int, min_track_length: int = 3) -> None:
        depth_data = self.get_depth_data(image_id)
        prior_depth_map, depth_range = compute_image_depthmap(self.reconstruction, image_id, depth_data['K'], depth_data['pose'], depth_data['w'], depth_data['h'], min_track_length=min_track_length)

        if prior_depth_map is None:
            print(f"Warning: No prior depth map found for image {image_id} with min_track_length={min_track_length}")
            return

        if self.config.verbose:
            n_prior_depths = np.sum(prior_depth_map > 0)
            # print(f"Image {image_id:06d} has {n_prior_depths} prior depths")
            if n_prior_depths > 0:
                valid_mask = prior_depth_map > 0
                img = depth_data['image'].copy()
                img[valid_mask] = np.array([255, 0, 0])
                Image.fromarray(img).save(os.path.join(self.depth_data_folder, f"prior_depth_{image_id:06d}.png"))

        coord_mask = self.get_coord_mask(depth_data)
        prior_depth_map[~coord_mask] = 0
        depth_data['prior_depth_map'] = prior_depth_map
        depth_data['depth_range'] = depth_range
        assert prior_depth_map.shape == (depth_data['h'], depth_data['w']), f"Prior depth map shape {prior_depth_map.shape} does not match target size {depth_data['h']}x{depth_data['w']}"

    def set_prior_depth_data_from_reference(self, image_id: int) -> None:
        if self.reference_reconstruction is None:
            return
        depth_data = self.get_depth_data(image_id)
        ref_image_id = self.source_to_target_image_id_mapping[image_id]
        assert ref_image_id is not None, f"No target image id found for image {image_id}"
        prior_depth_map, depth_range = compute_image_depthmap(self.reference_reconstruction, ref_image_id, depth_data['K'], depth_data['pose'], depth_data['w'], depth_data['h'], min_track_length=1)

        if prior_depth_map is not None:
            coord_mask = self.get_coord_mask(depth_data)
            prior_depth_map[~coord_mask] = 0
            depth_data['prior_depth_map'] = prior_depth_map
            depth_data['depth_range'] = depth_range
            assert prior_depth_map.shape == (depth_data['h'], depth_data['w']), f"Prior depth map shape {prior_depth_map.shape} does not match target size {depth_data['h']}x{depth_data['w']}"
        else:
            print(f"Warning: No prior depth map found for image {image_id} with min_track_length=1")

    def set_prior_depth_data_from_threedn(self, dmap_name: str, keep_original_size: bool = False) -> None:
        threedn_folder = os.path.join(self.config.scene_folder, "threedn")
        dmap_path = os.path.join(threedn_folder, dmap_name)
        if not os.path.isfile(dmap_path):
            raise ValueError(f"Threedn dmap {dmap_path} does not exist")

        threedn_dmap = ThreednDepthData()
        threedn_dmap.load(dmap_path)

        K = np.array(threedn_dmap.K, dtype=np.float64).reshape(3, 3)
        R = np.array(threedn_dmap.R, dtype=np.float64).reshape(3, 3)
        C = np.array(threedn_dmap.C, dtype=np.float64)
        t = -R @ C

        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = R
        pose_4x4[:3, 3] = t.flatten()

        image_id = find_exact_image_match_from_extrinsics(self.reconstruction, R, t)
        if image_id is None:
            raise ValueError(f"No exact image match found for threedn dmap {dmap_path}")

        img_path = os.path.join(self.config.scene_folder, str(threedn_dmap.image_name))

        assert len(threedn_dmap.depth_size) == 2, f"Depth size must be a tuple of length 2, got {threedn_dmap.depth_size}"
        W = int(threedn_dmap.depth_size[0])
        H = int(threedn_dmap.depth_size[1])

        depth_data = self.get_depth_data(image_id)
        depth_data['image_name'] = os.path.join('images', os.path.basename(img_path))

        if keep_original_size:
            depth_data['w'] = W
            depth_data['h'] = H
        else:
            depth_data['w'] = self.config.target_w
            depth_data['h'] = self.config.target_h

        prior_dmap = threedn_dmap.depthMap.reshape(H, W)
        if not keep_original_size:
            prior_dmap = cv2.resize(prior_dmap, (self.config.target_w, self.config.target_h), interpolation=cv2.INTER_LINEAR)

        coord_mask = self.get_coord_mask(depth_data)
        prior_dmap[~coord_mask] = 0

        depth_data['prior_depth_map'] = prior_dmap
        depth_data['depth_range'] = threedn_dmap.depth_range

        K[0,:] *= depth_data['w'] / W
        K[1,:] *= depth_data['h'] / H
        depth_data['K'] = K
        depth_data['pose'] = pose_4x4



    def update_depth_data(self, image_id: int, depth_map: np.ndarray, confidence_map: Optional[np.ndarray] = None) -> None:
        depth_data = self.get_depth_data(image_id)

        w = depth_data['w']
        h = depth_data['h']

        coord_mask = self.get_coord_mask(depth_data)

        if depth_map.ndim == 4:
            depth_map = depth_map.squeeze(0).squeeze(-1)
        elif depth_map.ndim == 3: # 1xHxW
            depth_map = depth_map.squeeze(0)
        depth_map[~coord_mask] = 0
        depth_data['depth_map'] = depth_map

        if depth_data['mask'] is None:
            depth_data['mask'] = (depth_map > 0).astype(bool)
        else:
            depth_data['mask'] &= (depth_map > 0).astype(bool)

        if confidence_map is not None:
            if confidence_map.ndim == 4: # 1xHxWx1
                confidence_map = confidence_map.squeeze(0).squeeze(-1)
            elif confidence_map.ndim == 3: # 1xHxW
                confidence_map = confidence_map.squeeze(0)

            confidence_map[~coord_mask] = 0
            depth_data['confidence_map'] = confidence_map
            depth_data['confidence_range'] = (np.min(confidence_map[depth_map > 0]), np.max(confidence_map[depth_map > 0]))
            assert confidence_map.shape == (h, w), f"Confidence map shape {confidence_map.shape} does not match target size {h}x{w}"

        assert depth_map.shape == (h, w), f"Depth map shape {depth_map.shape} does not match target size {h}x{w}"

    def save_current_state(self, max_workers: int = 4) -> None:
        """
        Save all depth data files in parallel.
        Args:
            max_workers: Maximum number of worker threads for parallel saving.
        """
        img_ids = list(self.scene_depth_data.keys())
        self.parallel_executor.run_in_parallel_no_return(
            self.save_depth_data,
            img_ids,
            progress_desc="Saving depth data",
            max_workers=max_workers
        )

        if self.config.verbose:
            self.parallel_executor.run_in_parallel_no_return(
                self.save_heatmap,
                img_ids,
                progress_desc="Saving heatmaps",
                what_to_save="all",
                max_workers=max_workers
            )


    def get_batches_geometric(self, batch_size: int) -> List[List[int]]:
        """
        Split image IDs into batches using COLMAP reconstruction quality metrics.
        Each batch consists of a reference image and its best partner images.
        Once images are used in a batch, they cannot be reference images for future batches.

        Args:
            reconstruction: ColmapReconstruction object
            image_ids: List of image IDs
            batch_size: Maximum number of images per batch

        Returns:
            List of batches, where each batch is a list of image IDs
        """

        # Ensure image point mappings are built
        self.reconstruction._ensure_image_point_maps()

        batches = []
        used_as_reference = set()  # Images that have been used as reference images

        # Sort image IDs for consistent processing order
        remaining_candidates = sorted(self.active_image_ids)

        while len(used_as_reference) < len(self.active_image_ids):

            reference_image_id = None
            for img_id in remaining_candidates:
                if img_id not in used_as_reference:
                    reference_image_id = img_id
                    break

            if reference_image_id is None:
                break

            best_partners = self.reconstruction.find_similar_images_for_image(
                reference_image_id,
                min_points=10,  # Lower threshold for more flexibility
            )

            valid_partners = [pid for pid in best_partners if pid in self.active_image_ids]
            if len(valid_partners) == 0:
                print(f"No valid partners found for image {reference_image_id}")
                used_as_reference.add(reference_image_id)
                continue

            # Start batch with reference image
            batch = [reference_image_id]

            # Add best partners up to batch_size
            for partner in valid_partners:
                if len(batch) >= batch_size:
                    break
                if partner not in batch:  # Avoid duplicates
                    batch.append(partner)


            # If batch is too small, fill with any remaining images that haven't been references
            if len(batch) < batch_size:
                for img_id in remaining_candidates:
                    if len(batch) >= batch_size:
                        break
                    if img_id not in batch and img_id not in used_as_reference:
                        batch.append(img_id)

            # Mark ALL images in this batch as used (cannot be reference images anymore)
            for img_id in batch:
                used_as_reference.add(img_id)

            batches.append(batch)

        all_batched_images = set()
        for batch in batches:
            all_batched_images.update(batch)

        remaining_unprocessed = [img_id for img_id in self.active_image_ids if img_id not in all_batched_images]

        if remaining_unprocessed:
            batches.append(remaining_unprocessed)

        return batches

    def get_batches_sequential(self, batch_size: int) -> List[List[int]]:
        """
        Simple sequential split into batches.
        Args:
            batch_size: Maximum number of images per batch
        Returns:
            List of batches, where each batch is a list of image IDs
        """
        batches = []
        for i in range(0, len(self.active_image_ids), batch_size):
            batch = self.active_image_ids[i:i + batch_size]
            batches.append(batch)
        return batches

    def save_heatmap(self, image_id: int, what_to_save: str = "all") -> None:
        if what_to_save not in ["depth_map", "confidence_map", "prior_depth_map", "all"]:
            raise ValueError(f"Invalid what_to_save: {what_to_save}")

        depth_data = self.get_depth_data(image_id)

        img_name = os.path.basename(depth_data['image_name']).replace('.jpg', '')

        if what_to_save == "depth_map" and depth_data['depth_map'] is not None:
            rgb = colorize_heatmap(depth_data['depth_map'], data_range=depth_data['depth_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"depth_{img_name}.png"))

        elif what_to_save == "confidence_map" and depth_data['confidence_map'] is not None:
            rgb = colorize_heatmap(depth_data['confidence_map'], data_range=depth_data['confidence_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"confidence_{img_name}.png"))

        elif what_to_save == "prior_depth_map" and depth_data['prior_depth_map'] is not None:
            rgb = colorize_heatmap(depth_data['prior_depth_map'], data_range=depth_data['depth_range'])
            Image.fromarray(rgb).save(os.path.join(self.depth_data_folder, f"prior_depth_{img_name}.png"))

        elif what_to_save == "all":

            H = depth_data['h']
            W = depth_data['w']

            img = depth_data['image']

            empty = np.zeros((H, W, 3), dtype=np.uint8)
            depth_rgb = colorize_heatmap(depth_data['depth_map'], data_range=depth_data['depth_range']) if depth_data['depth_map'] is not None else empty
            conf_rgb  = colorize_heatmap(depth_data['confidence_map'], data_range=depth_data['confidence_range']) if depth_data['confidence_map'] is not None else empty
            prior_rgb = colorize_heatmap(depth_data['prior_depth_map'], data_range=depth_data['depth_range']) if depth_data['prior_depth_map'] is not None else empty
            # fusion_rgb = colorize_heatmap(depth_data['fused_depth_map'], data_range=depth_data['depth_range']) if depth_data['fused_depth_map'] is not None else empty
            _consistency_rgb = colorize_heatmap(depth_data['consistency_map'], data_range=(0, self.config.fusion_max_partners-1)) if depth_data['consistency_map'] is not None else empty


            masked_rgb = img.copy()
            masked_depth = depth_rgb.copy()

            if depth_data['mask'] is not None:
                masked_rgb[~depth_data['mask']] = 0
                masked_depth[~depth_data['mask']] = 0
            else:
                masked_rgb = empty
                masked_depth = depth_rgb

            # generate a legend image
            legend_image = np.zeros((H, 50), dtype=np.float32)
            for i in range(self.config.fusion_max_partners):
                legend_image[i*H//self.config.fusion_max_partners:(i+1)*H//self.config.fusion_max_partners, :] = i
            legend_rgb = colorize_heatmap(legend_image, data_range=(0, self.config.fusion_max_partners-1))

            combined = np.concatenate([img, prior_rgb, depth_rgb, conf_rgb, masked_rgb, masked_depth, legend_rgb], axis=1)
            Image.fromarray(combined).save(os.path.join(self.depth_data_folder, f"data_{img_name}.png"))

    def save_cloud(self, image_id: int, use_prior_depth: bool=False, use_fused_depth: bool=False, use_mask: bool=False, consistent_points: bool=False, file_name: Optional[str] = None) -> None:
        pts, colors, _ = self.get_point_cloud(image_id, use_prior_depth=use_prior_depth, use_fused_depth=use_fused_depth, use_mask=use_mask, consistency_threshold=self.config.fusion_min_consistency_count if consistent_points else 0)
        if pts is None:
            return
        if file_name is None:
            file_name = f"pointcloud_{image_id:06d}.ply"
        pc_path = os.path.join(self.cloud_folder, file_name)
        save_point_cloud(pts, colors, pc_path)

    def get_point_cloud(self, image_id: int, use_predicted_depth: bool=False, use_prior_depth: bool=False, use_fused_depth: bool=False, use_predicted_pose: bool=False, conf_threshold: float=0.0, consistency_threshold: int=0, use_mask: bool=False, get_color: bool=True) -> tuple:

        depth_data = self.get_depth_data(image_id)

        if use_prior_depth:
            depth_map = depth_data['prior_depth_map']
        elif use_fused_depth:
            depth_map = depth_data['fused_depth_map']
        elif use_predicted_depth:
            depth_map = depth_data['predicted_depth_map']
        else:
            depth_map = depth_data['depth_map']

        if depth_map is None:
            return None, None, None

        confidence_map = depth_data['confidence_map']  # (H, W)
        consistency_map = depth_data['consistency_map']  # (H, W)

        if use_predicted_pose:
            cam_from_world = depth_data['predicted_pose']
            camera_intrinsics = depth_data['predicted_K']
        else:
            cam_from_world = depth_data['pose']  # (4, 4) - camera_from_world pose
            camera_intrinsics = depth_data['K']  # (3, 3) - scaled intrinsics
        scaled_image = depth_data['image']

        depth_map_filtered = depth_map.copy()
        coord_mask = self.get_coord_mask(depth_data, export_margin=self.config.export_margin)
        depth_map_filtered[~coord_mask] = 0

        if use_mask:
            mask = depth_data['mask']
            if mask is not None:
                depth_map_filtered[~mask] = 0

        # Filter by confidence threshold
        if conf_threshold > 0.0 and confidence_map is not None:
            valid_mask = confidence_map >= conf_threshold
            depth_map_filtered[~valid_mask] = 0

        # Filter by consistency threshold
        if consistency_threshold > 0 and consistency_map is not None:
            valid_mask = consistency_map >= consistency_threshold
            depth_map_filtered[~valid_mask] = 0

        if self.config.export_margin > 0:
            H, W = depth_map.shape
            M = self.config.export_margin
            export_mask = np.zeros((H, W), dtype=bool)
            export_mask[M:H-M, M:W-M] = True
            depth_map_filtered[~export_mask] = 0

        if depth_map_filtered is None:
            return None, None, None

        # Check if we have valid depth values
        if np.max(depth_map_filtered) == 0:
            print(f"Warning: No valid depth values after filtering: Image Id {image_id:06d}")
            return None, None, None

        # Compute 3D points from depth using the saved camera parameters
        pts3d, valid_mask = depthmap_to_world_frame(depth_map_filtered, camera_intrinsics, cam_from_world)

        if not valid_mask.any():
            print(f"Warning: No valid points found in depth map: Image Id {image_id:06d}")
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), {}

        # Extract valid points
        pts3d = pts3d[valid_mask]

        if get_color:
            colors = np.ones((len(pts3d), 3), dtype=np.float32) * 0.5  # Gray color
            if scaled_image is not None:
                simg_array = np.array(scaled_image, dtype=np.float32) / 255.0  # Shape: (518, 518, 3)
                colors = simg_array[valid_mask]  # Shape: (N_valid_pixels, 3)
        else:
            colors = None

        return pts3d, colors, valid_mask

    def uvd_to_world_frame(self, image_id: int, uvd_map: np.ndarray) -> np.ndarray:
        """
        Convert uvd map to world frame.

        Args:
            image_id: ID of the image
            uvd_map: HxWx3 array containing [u, v, depth] coordinates
        Returns:
            xyz_map: HxWx3 array containing [x, y, z] world coordinates (0 for invalid points)
        """
        depth_data = self.get_depth_data(image_id)
        assert depth_data is not None, f"Depth data for image {image_id} not found"
        return uvd_to_world_frame(uvd_map, depth_data['K'], depth_data['pose'])

    def fuse_for_image(self, image_id: int):
        depth_data = self.get_depth_data(image_id)
        assert depth_data is not None, f"Depth data for image {image_id} not found"

        pts3d, valid_mask = depthmap_to_world_frame(depth_data['depth_map'], depth_data['K'], depth_data['pose'])
        if depth_data['mask'] is not None:
            valid_mask = valid_mask & depth_data['mask']

        coord_mask = self.get_coord_mask(depth_data, export_margin=self.config.export_margin)
        valid_mask &= coord_mask

        # project each valid point onto partner images and check if the projected point's depth is consistent with
        # the partner's depth map.
        accumulated_pts3d = np.zeros_like(pts3d, dtype=np.float32)
        accumulated_pts3d[valid_mask] = pts3d[valid_mask]
        accumulated_valid_mask = (valid_mask).astype(np.uint8)

        partner_image_ids = depth_data['partner_ids']
        partner_uvds = {}

        for partner_id in partner_image_ids:
            partner_data = self.get_depth_data(partner_id)
            partner_uvd = self.compute_consistency_map_depths(pts3d, valid_mask, partner_id, consistency_threshold=self.config.fusion_consistency_threshold)
            partner_valid_mask = valid_mask & (partner_uvd[:,:,2] > 0).astype(np.uint8)
            partner_uvd[partner_valid_mask,2] = 0
            partner_uvds[partner_id] = partner_uvd
            X = uvd_to_world_frame(partner_uvd, partner_data['K'], partner_data['pose'])
            accumulated_valid_mask += partner_valid_mask.astype(np.uint8)
            accumulated_pts3d[partner_valid_mask>0] += X[partner_valid_mask>0]

        # export 3d points that have accumulated valid mask > fusion_min_consistency_count
        consistency_mask = accumulated_valid_mask > self.config.fusion_min_consistency_count
        depth_data['consistency_mask'] = accumulated_valid_mask
        if not np.any(consistency_mask):
            return

        # average point by the number of accumulated valid masks per pixel
        fused_pts3d = np.zeros_like(pts3d, dtype=np.float32)
        fused_pts3d[consistency_mask] = accumulated_pts3d[consistency_mask] / accumulated_valid_mask[consistency_mask][:, None]
        fused_dmap, _ = compute_depthmap(fused_pts3d[consistency_mask], depth_data['K'], depth_data['pose'], target_w=depth_data['w'], target_h=depth_data['h'])

        depth_data['fused_depth_map'] = fused_dmap

    def apply_fusion(self):

        # print(f"Fusing depth maps for {len(self.active_image_ids)} images")
        # for image_id in self.active_image_ids:
        #     self.fuse_for_image(image_id)

        self.parallel_executor.run_in_parallel_no_return(
            self.fuse_for_image,
            self.active_image_ids,
            progress_desc="Fusing depth maps",
            max_workers=8
        )

    def export_fused_point_cloud(self, stepping: int = 1, file_name: str = "fused.ply", use_parallel: bool = True):
        print(f"Exporting fused point cloud with stepping {stepping} and file name {file_name}")
        if use_parallel:
            return self._export_fused_point_cloud_parallel(stepping, file_name)
        else:
            return self._export_fused_point_cloud_sequential(stepping, file_name)

    def _export_fused_point_cloud_sequential(self, stepping: int = 1, file_name: str = "fused.ply"):
        pts_list = []
        colors_list = []

        # Initialize exported masks
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['exported_mask'] = np.zeros_like(depth_data['depth_map'], dtype=bool)

        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            if depth_data['fused_depth_map'] is not None:
                dmap = depth_data['fused_depth_map']
                if dmap is None:
                    continue

                exported_mask = depth_data['exported_mask']
                image_pts3d, valid_mask = depthmap_to_world_frame(dmap, depth_data['K'], depth_data['pose'])

                # Remove already exported points
                valid_mask &= ~exported_mask

                # Mark points that have been exported to partner images (before applying stepping)
                for partner_id in depth_data['partner_ids']:
                    partner_data = self.get_depth_data(partner_id)
                    uvd = self.compute_consistency_map_depths(image_pts3d, valid_mask, partner_id, consistency_threshold=self.config.fusion_consistency_threshold)
                    uv = uvd[uvd[:,:,2] > 0, :2]
                    if len(uv) > 0:
                        uv = uv.astype(int)
                        partner_data['exported_mask'][uv[:, 1], uv[:, 0]] = True

                # Now apply stepping mask to choose which points to export from this image
                if stepping > 1:
                    stepping_mask = np.zeros((depth_data['h'], depth_data['w']), dtype=bool)
                    stepping_mask[::stepping, ::stepping] = True
                    valid_mask &= stepping_mask

                # Extract points and colors
                pts3d = image_pts3d[valid_mask]
                if len(pts3d) > 0:
                    current_colors = np.ones((len(pts3d), 3), dtype=np.float32) * 0.5  # Gray color
                    if depth_data['image'] is not None:
                        simg_array = np.array(depth_data['image'], dtype=np.float32) / 255.0
                        current_colors = simg_array[valid_mask]

                    pts_list.append(pts3d)
                    colors_list.append(current_colors)

        # Save results
        pts = np.vstack(pts_list)
        colors = np.vstack(colors_list)
        print(f"Exported {len(pts)} points to {file_name}")
        save_point_cloud(pts, colors, os.path.join(self.cloud_folder, file_name))

    def _export_fused_point_cloud_parallel(self, stepping: int = 1, file_name: str = "fused.ply"):
        from threading import Lock

        # Initialize exported masks
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['exported_mask'] = np.zeros_like(depth_data['depth_map'], dtype=bool)

        # Thread-safe data collection
        pts_list = []
        colors_list = []
        data_lock = Lock()
        mask_locks = {img_id: Lock() for img_id in self.active_image_ids}

        def process_image(image_id: int):
            """Process a single image in parallel"""
            depth_data = self.get_depth_data(image_id)
            if depth_data['fused_depth_map'] is None:
                return None

            dmap = depth_data['fused_depth_map']
            if dmap is None:
                return None

            # Get exported mask with lock
            with mask_locks[image_id]:
                exported_mask = depth_data['exported_mask'].copy()

            image_pts3d, valid_mask = depthmap_to_world_frame(dmap, depth_data['K'], depth_data['pose'])

            # Remove already exported points
            valid_mask &= ~exported_mask

            # Process partner images with locks to avoid race conditions (before applying stepping)
            for partner_id in depth_data['partner_ids']:
                if partner_id in mask_locks:  # Only process if partner is active
                    partner_data = self.get_depth_data(partner_id)
                    uvd = self.compute_consistency_map_depths(image_pts3d, valid_mask, partner_id, consistency_threshold=self.config.fusion_consistency_threshold)
                    uv = uvd[uvd[:,:,2] > 0, :2]
                    if len(uv) > 0:
                        uv = uv.astype(int)
                        # Thread-safe update of partner's exported_mask
                        with mask_locks[partner_id]:
                            partner_data['exported_mask'][uv[:, 1], uv[:, 0]] = True

            # Now apply stepping mask to choose which points to export from this image
            if stepping > 1:
                stepping_mask = np.zeros((depth_data['h'], depth_data['w']), dtype=bool)
                stepping_mask[::stepping, ::stepping] = True
                valid_mask &= stepping_mask

            # Extract points and colors
            pts3d = image_pts3d[valid_mask]
            if len(pts3d) > 0:
                current_colors = np.ones((len(pts3d), 3), dtype=np.float32) * 0.5
                if depth_data['image'] is not None:
                    simg_array = np.array(depth_data['image'], dtype=np.float32) / 255.0
                    current_colors = simg_array[valid_mask]

                # Thread-safe collection
                with data_lock:
                    pts_list.append(pts3d)
                    colors_list.append(current_colors)

            return len(pts3d) if len(pts3d) > 0 else 0

        # Run in parallel
        self.parallel_executor.run_in_parallel_no_return(
            process_image,
            self.active_image_ids,
            progress_desc="Exporting point cloud",
            max_workers=4
        )

        pts = np.vstack(pts_list)
        colors = np.vstack(colors_list)
        print(f"Exported {len(pts)} points to {file_name}")
        save_point_cloud(pts, colors, os.path.join(self.cloud_folder, file_name))

    def compute_consistency_map_depths(self, points_3d: np.ndarray, valid_mask: np.ndarray, partner_id: int, consistency_threshold: float) -> np.ndarray:
        """
        Compute consistency map by projecting valid 3D points to partner camera and comparing depths.

        Args:
            points_3d: HxWx3 array of 3D points in world coordinates
            valid_mask: HxW boolean mask indicating valid points
            partner_id: ID of partner image for consistency check

        Returns:
            consistency_map: HxWx3 array containing [u, v, depth] where u,v are partner image coordinates (all 0.0 = invalid)
        """
        partner_data = self.get_depth_data(partner_id)
        depth_map = partner_data['depth_map']
        intrinsics = partner_data['K']
        pose = partner_data['pose']  # cam_from_world for partner
        return compute_consistent_map_info(points_3d, valid_mask, depth_map, intrinsics, pose, consistency_threshold)

    def _save_single_result(self, image_id: int, tag: str = "") -> None:
        """Save all results for a single image."""
        self.save_depth_data(image_id)
        self.save_heatmap(image_id, what_to_save="all")
        self.save_cloud(image_id, file_name=f"{tag}cloud{image_id:06d}.ply")
        self.save_cloud(image_id, consistent_points=True, file_name=f"{tag}consistent_cloud{image_id:06d}.ply")
        self.save_cloud(image_id, use_prior_depth=True, file_name=f"{tag}prior_cloud{image_id:06d}.ply")
        self.save_cloud(image_id, use_fused_depth=True, file_name=f"{tag}fused_cloud{image_id:06d}.ply")

    def save_results(self) -> None:
        self.parallel_executor.run_in_parallel_no_return(
            self._save_single_result,
            self.active_image_ids,
            progress_desc="Saving results",
            max_workers=8
        )

    def transfer_fused_to_prior(self) -> None:
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['prior_depth_map'] = depth_data['fused_depth_map']
            depth_data['depth_map'] = depth_data['fused_depth_map']
            depth_data['fused_depth_map'] = None
            depth_data['mask'] = None
            depth_data['confidence_map'] = None
            depth_data['consistency_map'] = None

    def transfer_depthmap_to_prior(self) -> None:
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['prior_depth_map'] = depth_data['depth_map']
            depth_data['depth_map'] = None
            depth_data['fused_depth_map'] = None
            depth_data['mask'] = None
            depth_data['confidence_map'] = None
            depth_data['consistency_map'] = None

    def transfer_prior_to_depth(self) -> None:
        for image_id in self.active_image_ids:
            depth_data = self.get_depth_data(image_id)
            depth_data['depth_map'] = depth_data['prior_depth_map']
            depth_data['prior_depth_map'] = None

    def is_precomputed_depth_data_present(self) -> bool:
        depth_data_files = glob.glob(os.path.join(self.depth_data_folder, "*.npz"))
        return len(depth_data_files) > 0

    def fuse_threedn_dmaps(self):
        threedn_folder = os.path.join(self.config.scene_folder, "threedn")
        if os.path.exists(threedn_folder):
            print("Threedn folder found, initializing from threedn")
            self.set_priors_from_threedn(keep_original_size=True)
            self.config.image_mode = 'configured'
            self.initialize(load_images=True)
            self.transfer_prior_to_depth()
            self.apply_fusion()
            self.export_fused_point_cloud(file_name="lidar_fused.ply", use_parallel=True)
        else:
            raise RuntimeError("Threedn folder not found, cannot fuse lidar dmaps")

    def run_densification(self):

        assert self.config.run_depth_completion is not None, "run_depth_completion function is not set"

        if self.is_precomputed_depth_data_present():
            print("Precomputed depth data found, initializing from folder")
            self.initialize_from_folder()

        else:

            if self.config.prior_type == 'none':
                self.initialize(load_images=True)
            elif self.config.prior_type == 'reference':
                assert self.config.reference_reconstruction is not None, "reference_reconstruction is required when prior_type is reference"
                self.initialize(load_images=True)
                self.set_priors_from_reference()
            elif self.config.prior_type == 'sfm':
                self.initialize(load_images=True)
                self.set_priors_from_sfm(min_track_length=3)
            elif self.config.prior_type == 'dmaps':
                self.initialize(load_images=True)
                self.set_priors_from_threedn(keep_original_size=False)
                self.transfer_prior_to_depth()
                self.apply_fusion()
                self.transfer_fused_to_prior()
            else:
                raise ValueError(f"Invalid prior type: [{self.config.prior_type}]")

            if self.config.smart_batching:
                print("Using smart batching based on COLMAP reconstruction quality...")
                image_batches = self.get_batches_geometric(self.config.batch_size)
                print(f"Processing {len(image_batches)} smart batches with max batch size {self.config.batch_size}")
            else:
                print("Using sequential batching...")
                image_batches = self.get_batches_sequential(self.config.batch_size)
                print(f"Processing {len(image_batches)} sequential batches with batch size {self.config.batch_size}")

            self.config.run_depth_completion(self, self.config, image_batches)
            self.save_current_state()

        # if self.config.export_resolution > 0:
        #     print(f"Exporting 3dn depthmaps at resolution {self.config.export_resolution}")
        #     self.scale_all_depth_data(self.config.export_resolution, is_square=False)
        #     self.export_dmaps(max_workers=8, verbose=self.config.verbose)

        self.apply_fusion()
        self.export_fused_point_cloud(file_name="fused.ply", use_parallel=True)

        if self.config.verbose:
            self.save_results()

    def export_dmap_as_tiff(self, image_id: int, verbose: bool = False) -> None:
        depth_data = self.get_depth_data(image_id)
        dmap = depth_data['depth_map']
        if dmap is None:
            print(f"Warning: No depth map found for image {image_id}")
            return
        img_name = Path(os.path.basename(depth_data['image_name'])).with_suffix('.tiff')
        print(f"Saving dmap to {os.path.join(self.tiff_folder, img_name)}")
        cv2.imwrite(os.path.join(self.tiff_folder, img_name), dmap)

    def export_as_threedn_depth_data(self, image_id: int, export_id: int, verbose: bool = False) -> None:
        depth_data = self.get_depth_data(image_id)

        threedn_depth_data = ThreednDepthData()

        K_export = depth_data['K']
        export_width = depth_data['w']
        export_height = depth_data['h']

        print(f"Exporting dmap for image {image_id} with size {export_width}x{export_height}: {export_id:04d}: {depth_data['image_name']}")

        dmap = depth_data['depth_map']
        if dmap is None:
            print(f"Warning: No depth map found for image {image_id}")
            return

        mask = depth_data['mask']
        if mask is not None:
            dmap[~mask] = 0

        threedn_depth_data.magic = "DR"
        threedn_depth_data.image_name = os.path.join('images', os.path.basename(depth_data['image_name']))
        threedn_depth_data.image_size = (export_width, export_height)
        threedn_depth_data.depth_size = (export_width, export_height)
        threedn_depth_data.depth_range = np.min(dmap[dmap>0]), np.max(dmap[dmap>0])

        cam_from_world = depth_data['pose']
        R = cam_from_world[:3, :3]
        t = cam_from_world[:3, 3]
        C = -R.T @ t

        threedn_depth_data.K = K_export.flatten().tolist()
        threedn_depth_data.R = R.flatten().tolist()
        threedn_depth_data.C = C.flatten().tolist()
        threedn_depth_data.flags = ThreednDepthData.HAS_DEPTH
        threedn_depth_data.depthMap = dmap.flatten()

        partner_ids = depth_data['partner_ids']
        threedn_depth_data.neighbors = partner_ids
        threedn_depth_data.hsize = threedn_depth_data.headersize()
        threedn_depth_data.save(os.path.join(self.dmap_folder, f"depth{export_id:04d}.dmap"))

    def export_all_dmaps(self) -> None:

        self.active_image_ids.sort()

        for i, image_id in enumerate(self.active_image_ids):
            self.export_as_threedn_depth_data(image_id, export_id=i, verbose=False)
            self.export_dmap_as_tiff(image_id, verbose=False)

    def scale_all_depth_data(self, max_image_size: int, is_square: bool = False):

        new_w = 0
        new_h = 0
        if is_square:
            new_w = max_image_size
            new_h = max_image_size
        else:
            for image_id in self.active_image_ids:
                camera = self.reconstruction.get_image_camera(image_id)
                original_width, original_height = camera.width, camera.height
                if new_w == 0 and new_h == 0:
                    new_w = int(original_width  * max_image_size / max(original_width, original_height))
                    new_h = int(original_height * max_image_size / max(original_width, original_height))
                else:
                    img_w = int(original_width  * max_image_size / max(original_width, original_height))
                    img_h = int(original_height * max_image_size / max(original_width, original_height))

                    if img_w != new_w or img_h != new_h:
                        raise ValueError(f"Image {image_id} has size {img_w}x{img_h} which does not match target size {new_w}x{new_h}")

        self.target_w = new_w
        self.target_h = new_h

        self.parallel_executor.run_in_parallel_no_return(
            self.scale_depth_data,
            self.active_image_ids,
            progress_desc="Scaling depth data",
            max_workers=4,
            new_w=new_w,
            new_h=new_h
        )

    def scale_depth_data(self, image_id: int, new_w: int, new_h: int) -> None:

        depth_data = self.get_depth_data(image_id)
        if depth_data['depth_map'] is None:
            depth_data['w'] = new_w
            depth_data['h'] = new_h
            depth_data['scaled_image'] = None
            return

        # Extract camera intrinsics and scale for target size
        camera = self.reconstruction.get_image_camera(image_id)
        original_width, original_height = camera.width, camera.height

        if new_w == original_width and new_h == original_height:
            return

        if depth_data['h'] == new_h and depth_data['w'] == new_w:
            return

        depth_data['w'] = new_w
        depth_data['h'] = new_h
        depth_data['image'] = None
        self.initialize_image(image_id, mode='configured')

        depth_data['depth_map'] = cv2.resize(depth_data['depth_map'], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if depth_data['confidence_map'] is not None:
            depth_data['confidence_map'] = cv2.resize(depth_data['confidence_map'], (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        valid_mask = (depth_data['depth_map'] > 0).astype(bool)
        depth_data['mask'] = valid_mask
        depth_data['prior_depth_map'] = None
        depth_data['fused_depth_map'] = None
        depth_data['consistency_map'] = None
        depth_data['normals'] = None
        depth_data['normals_confidence'] = None
        depth_data['depth_range'] = None


    def register_depth_map_to_sfm(self, image_id: int):

        iid_data = self.get_depth_data(image_id)

        # generate constraints from the depth map
        points_3d, map_points, uvd = self.extract_point_constraints(image_id)

        if len(points_3d) <= self.config.alignment_min_points:
            print(f"Warning: Not enough points to align depth map for image {image_id} [{len(points_3d)}] - deleting depth_map")
            iid_data['depth_map'] = None
            iid_data['confidence_map'] = None
            iid_data['normals'] = None
            iid_data['normals_confidence'] = None
            return

        transform = geometric_utility.compute_robust_similarity_transform(map_points, points_3d, inlier_threshold=self.config.alignment_inlier_threshold)
        pts, _colors, _valid_mask = self.get_point_cloud(image_id, use_predicted_depth=True, use_predicted_pose=True, use_mask=True)
        transformed_pts = geometric_utility.apply_transform(pts, transform)
        dmap, _ = geometric_utility.compute_depthmap(transformed_pts, iid_data['K'], iid_data['pose'], iid_data['w'], iid_data['h'])

        self.update_depth_data(image_id, dmap, None)

    def register_depth_maps(self):
        self.parallel_executor.run_in_parallel_no_return(
            self.register_depth_map_to_sfm,
            self.active_image_ids,
            progress_desc="Registering depth maps",
        )
