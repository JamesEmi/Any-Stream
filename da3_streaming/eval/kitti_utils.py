"""KITTI odometry dataset utilities for pose loading and format conversion."""

import numpy as np


def load_kitti_poses(pose_path: str) -> np.ndarray:
    """Load KITTI ground-truth poses from a text file.

    Each line contains 12 values representing a 3x4 [R|t] matrix (row-major).
    Returns [N, 4, 4] homogeneous transforms.
    """
    raw = np.loadtxt(pose_path).reshape(-1, 3, 4)
    N = raw.shape[0]
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    poses[:, :3, :4] = raw
    poses[:, 3, 3] = 1.0
    return poses


def load_kitti_timestamps(times_path: str) -> np.ndarray:
    """Load KITTI timestamps (seconds) from times.txt. Returns [N] float64."""
    return np.loadtxt(times_path, dtype=np.float64)


def poses_to_evo(poses_4x4: np.ndarray, timestamps: np.ndarray = None):
    """Convert [N,4,4] poses to an evo PoseTrajectory3D.

    If timestamps is None, uses integer indices.
    """
    from evo.core.trajectory import PoseTrajectory3D

    N = poses_4x4.shape[0]
    if timestamps is None:
        timestamps = np.arange(N, dtype=np.float64)
    return PoseTrajectory3D(poses_se3=list(poses_4x4), timestamps=timestamps)


def orthogonalize_rotations(poses_4x4: np.ndarray) -> np.ndarray:
    """Re-orthogonalize rotation matrices via SVD to ensure valid SO(3)."""
    out = poses_4x4.copy()
    for i in range(len(out)):
        R = out[i, :3, :3]
        U, _, Vt = np.linalg.svd(R)
        # Ensure proper rotation (det = +1)
        d = np.linalg.det(U @ Vt)
        S = np.diag([1.0, 1.0, d])
        out[i, :3, :3] = U @ S @ Vt
    return out


def save_kitti_poses(poses_4x4: np.ndarray, out_path: str):
    """Save [N,4,4] poses in KITTI format (12 values per line).
    Rotation matrices are re-orthogonalized before saving.
    """
    poses_4x4 = orthogonalize_rotations(poses_4x4)
    with open(out_path, "w") as f:
        for P in poses_4x4:
            vals = P[:3, :4].flatten()
            f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")
