from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Predictions:
    """
    Unified predictions object for multi-view depth/3D estimation models.

    All arrays use numpy format with the following shapes:
        depth: (N, H, W) - depth values per pixel
        conf: (N, H, W) - confidence scores per pixel
        extrinsics: (N, 3, 4) - camera poses (W2C format)
        intrinsics: (N, 3, 3) - camera intrinsic matrices
        processed_images: (N, H, W, 3) - RGB images (uint8)
        mask: (N, H, W) - optional valid pixel mask (bool)
        world_points: (N, H, W, 3) - optional pre-computed world coordinates

    Where N = number of frames, H = height, W = width

    Note: intrinsics is optional when world_points is provided directly.
    """
    depth: np.ndarray
    conf: np.ndarray
    extrinsics: np.ndarray
    processed_images: np.ndarray
    intrinsics: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    world_points: Optional[np.ndarray] = None


@dataclass
class PredictionsMA:
    """
    MapAnything-native predictions format.

    Stores outputs exactly as MapAnything produces them for direct reuse
    in subsequent chunk inference (sliding_prior mode). This avoids
    unnecessary conversions (C2W<->W2C, resolution scaling, etc.).

    Native forward() format (for prior injection):
        depth_along_ray: (N, H, W, 1) - depth along ray (Euclidean distance)
        ray_directions: (N, H, W, 3) - unit ray directions in camera frame
        cam_quats: (N, 4) - camera quaternions (wxyz format)
        cam_trans: (N, 3) - camera translations

    Derived format (at inference resolution):
        depth: (N, H, W) - Z-depth at inference resolution
        intrinsics: (N, 3, 3) - camera intrinsics at inference resolution
        camera_poses_c2w: (N, 4, 4) - C2W format poses
        conf: (N, H, W) - confidence scores
        mask: (N, H, W) - valid pixel mask (bool)
        processed_images: (N, H, W, 3) - RGB images (uint8)
        world_points: (N, H, W, 3) - world frame 3D points

    For DA3-streaming compatibility:
        extrinsics: (N, 3, 4) - W2C format for any_streaming.py
    """
    # Native forward() format for prior injection
    depth_along_ray: np.ndarray
    ray_directions: np.ndarray
    cam_quats: np.ndarray
    cam_trans: np.ndarray
    # Derived format
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_poses_c2w: np.ndarray
    conf: np.ndarray
    mask: np.ndarray
    processed_images: np.ndarray
    world_points: np.ndarray
    # DA3-streaming compatibility (computed once during storage)
    extrinsics: np.ndarray