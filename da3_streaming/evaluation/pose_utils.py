"""Trajectory I/O utilities.

`load_kitti_poses` / `save_poses` use the KITTI 12-float [R|t] format.
`load_tartanair_poses` reads TartanAir 7-float (x y z qx qy qz qw, scalar-last).
`load_poses` auto-dispatches on column count so callers don't need to know the format.
"""

import numpy as np


def load_kitti_poses(pose_path: str) -> np.ndarray:
    """Load KITTI 12-float [R|t] (3x4 row-major) per line. Returns [N, 4, 4]."""
    raw = np.loadtxt(pose_path).reshape(-1, 3, 4)
    N = raw.shape[0]
    poses = np.zeros((N, 4, 4), dtype=np.float64)
    poses[:, :3, :4] = raw
    poses[:, 3, 3] = 1.0
    return poses


def load_tartanair_poses(pose_path: str) -> np.ndarray:
    """Load TartanAir pose_lcam_front.txt (N rows: x y z qx qy qz qw, scalar-last).
    Returns [N, 4, 4] homogeneous transforms in TartanAir's own world frame.
    """
    from scipy.spatial.transform import Rotation as R

    raw = np.loadtxt(pose_path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    assert raw.shape[1] == 7, f"expected 7 cols (x y z qx qy qz qw), got {raw.shape}"
    t = raw[:, 0:3]
    q = raw[:, 3:7]                         # scipy from_quat is scalar-last
    Rmat = R.from_quat(q).as_matrix()
    N = raw.shape[0]
    P = np.zeros((N, 4, 4), dtype=np.float64)
    P[:, :3, :3] = Rmat
    P[:, :3, 3] = t
    P[:, 3, 3] = 1.0
    return P


def load_poses(pose_path: str) -> np.ndarray:
    """Load a pose file in any supported format (auto-dispatched on column count).
    Returns [N, 4, 4] homogeneous transforms.
    """
    raw = np.loadtxt(pose_path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    cols = raw.shape[1]
    if cols == 12:
        return load_kitti_poses(pose_path)
    if cols == 7:
        return load_tartanair_poses(pose_path)
    raise ValueError(
        f"{pose_path}: expected 12 cols (KITTI [R|t]) or 7 cols (TartanAir xyz+quat), "
        f"got shape {raw.shape}"
    )


def load_kitti_timestamps(times_path: str) -> np.ndarray:
    """Load KITTI timestamps (seconds) from times.txt. Returns [N] float64."""
    return np.loadtxt(times_path, dtype=np.float64)


def poses_to_evo(poses_4x4: np.ndarray, timestamps: np.ndarray = None):
    """Convert [N,4,4] poses to an evo PoseTrajectory3D. Defaults to integer index timestamps."""
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
        d = np.linalg.det(U @ Vt)
        S = np.diag([1.0, 1.0, d])
        out[i, :3, :3] = U @ S @ Vt
    return out


def save_poses(poses_4x4: np.ndarray, out_path: str):
    """Save [N,4,4] poses in KITTI 12-float [R|t] format. Re-orthogonalizes rotations first."""
    poses_4x4 = orthogonalize_rotations(poses_4x4)
    with open(out_path, "w") as f:
        for P in poses_4x4:
            vals = P[:3, :4].flatten()
            f.write(" ".join(f"{v:.6e}" for v in vals) + "\n")
