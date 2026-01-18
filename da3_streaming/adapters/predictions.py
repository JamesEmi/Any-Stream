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
    
    Where N = number of frames, H = height, W = width
    """
    depth: np.ndarray
    conf: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    processed_images: np.ndarray
    mask: Optional[np.ndarray] = None
    world_points: Optional[np.ndarray] = None  # (N, H, W, 3) pre-computed world points