#!/usr/bin/env python3
"""
Visualize the metric quality of DA3-Streaming point cloud reconstruction.

Overlays:
1. GPS ground truth trajectory (green)
2. Predicted camera poses (blue)
3. Casualty locations (red spheres with controllable transparency)

Computes alignment metrics (Sim3) and reports scale/rotation/translation errors.

Usage:
    python visualize_metric_quality.py \
        --ply_path ./results/drone/veh5_all_12k_default/pcd/combined_pcd.ply \
        --poses_path ./results/drone/veh5_all_12k_default/camera_poses.txt \
        --image_dir /path/to/images \
        --gps_csv /path/to/gps.csv \
        --casualty_csv /path/to/casualties.csv \
        --output_path ./results/drone/veh5_all_12k_default/metric_viz.ply
"""

import argparse
import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pymap3d as pm
except ImportError:
    raise ImportError("Install pymap3d: pip install pymap3d")

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Install open3d: pip install open3d")


@dataclass
class GpsSample:
    t_ns: int
    t_s: float
    lat: float
    lon: float
    alt: float


def read_gps_csv(csv_path: str) -> List[GpsSample]:
    """Read GPS data from CSV file."""
    rows: List[GpsSample] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            t_ns = int(r.get("timestamp_ns") or 0)
            t_s = float(r.get("timestamp_s") or 0.0)
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            alt = float(r["altitude"])
            rows.append(GpsSample(t_ns=t_ns, t_s=t_s, lat=lat, lon=lon, alt=alt))
    if not rows:
        raise ValueError(f"No rows parsed from {csv_path}")
    return rows


def extract_ts_ns(path: str) -> Optional[int]:
    """Extract timestamp in nanoseconds from image filename."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    m = re.match(r"^(\d{12,})$", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def build_enu_interpolator(gps_rows: List[GpsSample]):
    """Build an ENU interpolator from GPS samples."""
    t_ns = np.array([g.t_ns for g in gps_rows], dtype=np.int64)
    order = np.argsort(t_ns)
    gps_sorted = [gps_rows[i] for i in order]
    t_ns = t_ns[order]

    g0 = gps_sorted[0]
    e_list, n_list, u_list = [], [], []
    for g in gps_sorted:
        e, n, u = pm.geodetic2enu(g.lat, g.lon, g.alt, g0.lat, g0.lon, g0.alt)
        e_list.append(e)
        n_list.append(n)
        u_list.append(u)
    e_arr = np.asarray(e_list, dtype=np.float64)
    n_arr = np.asarray(n_list, dtype=np.float64)
    u_arr = np.asarray(u_list, dtype=np.float64)

    def interp(ts_ns):
        ts = np.asarray(ts_ns, dtype=np.int64)
        e = np.interp(ts, t_ns, e_arr, left=np.nan, right=np.nan)
        n = np.interp(ts, t_ns, n_arr, left=np.nan, right=np.nan)
        u = np.interp(ts, t_ns, u_arr, left=np.nan, right=np.nan)
        return e, n, u

    return interp, {"t_ns": t_ns, "origin": (g0.lat, g0.lon, g0.alt)}


def read_casualty_csv(csv_path: str) -> List[Tuple[str, float, float, float]]:
    """
    Read casualty locations from CSV.
    Expected columns: casualty_id, lat, lon (altitude optional)
    Returns: list of (id, lat, lon, alt) tuples
    """
    casualties = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            cid = r.get("casualty_id", str(len(casualties)))
            lat = float(r.get("lat") or r.get("latitude"))
            lon = float(r.get("lon") or r.get("longitude"))
            alt = float(r.get("alt") or r.get("altitude") or 0.0)
            casualties.append((cid, lat, lon, alt))
    return casualties


def load_predicted_poses(poses_path: str) -> np.ndarray:
    """
    Load predicted camera poses from DA3 output.
    Each line is a flattened 4x4 C2W matrix (16 numbers).
    Returns: (N, 4, 4) array of C2W matrices.
    """
    poses = []
    with open(poses_path, "r") as f:
        for line in f:
            values = [float(x) for x in line.strip().split()]
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)
    return np.array(poses)


def get_image_paths(image_dir: str) -> List[str]:
    """Get sorted list of image paths."""
    extensions = (".jpg", ".jpeg", ".png")
    paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(extensions)
    ]
    return paths


def compute_gps_trajectory_enu(
    image_paths: List[str], gps_csv: str
) -> Tuple[np.ndarray, List[int], GpsSample]:
    """
    Compute GPS trajectory in ENU coordinates for each image.
    Returns: (N, 3) ENU positions, list of valid indices, ENU origin (chronologically first GPS point)
    """
    gps_rows = read_gps_csv(gps_csv)
    interp_enu, meta = build_enu_interpolator(gps_rows)

    # Get the chronologically first GPS point (used as ENU origin)
    t_ns = np.array([g.t_ns for g in gps_rows], dtype=np.int64)
    origin = gps_rows[np.argmin(t_ns)]

    enu_positions = []
    valid_indices = []

    for i, path in enumerate(image_paths):
        ts = extract_ts_ns(path)
        if ts is None:
            continue
        e, n, u = interp_enu(ts)
        if np.isfinite(e) and np.isfinite(n) and np.isfinite(u):
            enu_positions.append([e, n, u])
            valid_indices.append(i)

    return np.array(enu_positions), valid_indices, origin


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Sim3 alignment (Umeyama) from src to dst points.

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points
        with_scale: if True, compute scale; otherwise scale=1

    Returns:
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector

    Transforms src to dst: dst = s * R @ src + t
    """
    assert src.shape == dst.shape
    n, dim = src.shape

    # Centroids
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Centered points
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Covariance
    H = src_centered.T @ dst_centered / n

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Scale
    if with_scale:
        var_src = np.sum(src_centered**2) / n
        s = np.sum(S) / var_src
    else:
        s = 1.0

    # Translation
    t = dst_mean - s * R @ src_mean

    return s, R, t


def compute_alignment_metrics(
    pred_positions: np.ndarray, gt_positions: np.ndarray
) -> Dict:
    """
    Compute Sim3 alignment and error metrics.

    Returns dict with:
        - scale: estimated scale factor
        - rotation: (3, 3) rotation matrix
        - translation: (3,) translation vector
        - rmse: root mean square error after alignment
        - ate: absolute trajectory error per point
        - scale_error_std: std of per-segment scale ratios
    """
    # Align predicted to ground truth
    s, R, t = umeyama_alignment(pred_positions, gt_positions, with_scale=True)

    # Apply alignment
    pred_aligned = s * (pred_positions @ R.T) + t

    # Compute errors
    errors = np.linalg.norm(pred_aligned - gt_positions, axis=1)
    rmse = np.sqrt(np.mean(errors**2))

    # Compute per-segment scale consistency
    gt_dists = np.linalg.norm(np.diff(gt_positions, axis=0), axis=1)
    pred_dists = np.linalg.norm(np.diff(pred_positions, axis=0), axis=1)

    # Avoid division by zero
    valid_mask = pred_dists > 1e-6
    scale_ratios = gt_dists[valid_mask] / pred_dists[valid_mask]
    scale_error_std = np.std(scale_ratios) if len(scale_ratios) > 0 else 0.0

    return {
        "scale": s,
        "rotation": R,
        "translation": t,
        "rmse": rmse,
        "ate": errors,
        "scale_error_std": scale_error_std,
        "pred_aligned": pred_aligned,
    }


def create_trajectory_lineset(
    positions: np.ndarray, color: List[float]
) -> o3d.geometry.LineSet:
    """Create a line set for trajectory visualization."""
    n = len(positions)
    lines = [[i, i + 1] for i in range(n - 1)]
    colors = [color for _ in range(len(lines))]

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(positions)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset


def create_sphere(
    center: np.ndarray, radius: float, color: List[float], resolution: int = 20
) -> o3d.geometry.TriangleMesh:
    """Create a sphere mesh."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def create_trajectory_spheres(
    positions: np.ndarray, color: List[float], radius: float = 0.5
) -> List[o3d.geometry.TriangleMesh]:
    """Create spheres at each trajectory position."""
    spheres = []
    for pos in positions:
        sphere = create_sphere(pos, radius, color)
        spheres.append(sphere)
    return spheres


def create_trajectory_points(
    positions: np.ndarray, color: List[float]
) -> o3d.geometry.PointCloud:
    """Create a point cloud for trajectory visualization (more efficient than spheres)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(positions), 1)))
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="Visualize metric quality of DA3-Streaming reconstruction"
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        required=True,
        help="Path to combined point cloud PLY file",
    )
    parser.add_argument(
        "--poses_path",
        type=str,
        required=True,
        help="Path to camera_poses.txt from DA3",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to input images directory",
    )
    parser.add_argument(
        "--gps_csv",
        type=str,
        required=True,
        help="Path to GPS CSV file",
    )
    parser.add_argument(
        "--casualty_csv",
        type=str,
        default=None,
        help="Optional path to casualty locations CSV",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="metric_viz.ply",
        help="Output path for visualization PLY",
    )
    parser.add_argument(
        "--casualty_radius",
        type=float,
        default=0.5,
        help="Radius of casualty spheres in meters",
    )
    parser.add_argument(
        "--casualty_alpha",
        type=float,
        default=1.0,
        help="Transparency of casualty spheres (0=transparent, 1=opaque)",
    )
    parser.add_argument(
        "--casualty_height_offset",
        type=float,
        default=0.0,
        help="Height offset for casualty markers in meters (added to altitude)",
    )
    parser.add_argument(
        "--traj_radius",
        type=float,
        default=0.3,
        help="Radius of trajectory marker spheres",
    )
    parser.add_argument(
        "--subsample_traj",
        type=int,
        default=10,
        help="Subsample trajectory points (plot every Nth point)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open interactive Open3D viewer",
    )
    parser.add_argument(
        "--no_alignment",
        action="store_true",
        help="Skip Sim3 alignment (assume already aligned)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DA3-Streaming Metric Quality Visualization")
    print("=" * 60)

    # Load point cloud
    print(f"\nLoading point cloud from: {args.ply_path}")
    pcd = o3d.io.read_point_cloud(args.ply_path)
    print(f"  Points: {len(pcd.points)}")

    # Load predicted poses
    print(f"\nLoading predicted poses from: {args.poses_path}")
    pred_poses = load_predicted_poses(args.poses_path)
    print(f"  Poses: {len(pred_poses)}")

    # Extract predicted camera positions (translation from C2W)
    pred_positions = pred_poses[:, :3, 3]

    # Get image paths and compute GPS trajectory
    print(f"\nLoading images from: {args.image_dir}")
    image_paths = get_image_paths(args.image_dir)
    print(f"  Images: {len(image_paths)}")

    print(f"\nComputing GPS trajectory from: {args.gps_csv}")
    gps_positions_enu, valid_indices, enu_origin = compute_gps_trajectory_enu(image_paths, args.gps_csv)
    print(f"  Valid GPS matches: {len(valid_indices)}")
    print(f"  ENU origin: lat={enu_origin.lat:.7f}, lon={enu_origin.lon:.7f}, alt={enu_origin.alt:.2f}")

    # Match predicted and GPS positions by index
    pred_matched = pred_positions[valid_indices]
    gt_matched = gps_positions_enu

    print(f"\nMatched trajectory points: {len(pred_matched)}")

    # Compute alignment metrics
    if not args.no_alignment and len(pred_matched) >= 3:
        print("\n" + "-" * 40)
        print("Computing Sim3 Alignment (Umeyama)")
        print("-" * 40)

        metrics = compute_alignment_metrics(pred_matched, gt_matched)

        print(f"\n  Scale factor: {metrics['scale']:.6f}")
        print(f"  RMSE after alignment: {metrics['rmse']:.4f} m")
        print(f"  Scale consistency (std of ratios): {metrics['scale_error_std']:.4f}")
        print(f"  Mean ATE: {np.mean(metrics['ate']):.4f} m")
        print(f"  Max ATE: {np.max(metrics['ate']):.4f} m")

        # Apply alignment to all predicted positions
        s, R, t = metrics["scale"], metrics["rotation"], metrics["translation"]
        pred_positions_aligned = s * (pred_positions @ R.T) + t

        # Also transform the point cloud
        pcd_points = np.asarray(pcd.points)
        pcd_points_aligned = s * (pcd_points @ R.T) + t
        pcd.points = o3d.utility.Vector3dVector(pcd_points_aligned)
    else:
        pred_positions_aligned = pred_positions
        metrics = None

    # Subsample trajectories for visualization
    subsample = args.subsample_traj
    pred_viz = pred_positions_aligned[::subsample]
    gt_viz = gps_positions_enu[::subsample]

    # Create visualization geometries
    geometries = [pcd]
    trajectory_meshes = []  # Track trajectory spheres separately
    casualty_meshes = []    # Track casualty spheres separately

    # GPS ground truth trajectory (green)
    print("\nCreating GPS trajectory visualization (green)...")
    gt_lineset = create_trajectory_lineset(gps_positions_enu, [0.0, 0.8, 0.0])
    geometries.append(gt_lineset)

    gt_spheres = create_trajectory_spheres(gt_viz, [0.0, 1.0, 0.0], args.traj_radius)
    geometries.extend(gt_spheres)
    trajectory_meshes.extend(gt_spheres)

    # Predicted trajectory (blue)
    print("Creating predicted trajectory visualization (blue)...")
    pred_lineset = create_trajectory_lineset(pred_positions_aligned, [0.0, 0.0, 0.8])
    geometries.append(pred_lineset)

    pred_spheres = create_trajectory_spheres(pred_viz, [0.0, 0.0, 1.0], args.traj_radius)
    geometries.extend(pred_spheres)
    trajectory_meshes.extend(pred_spheres)

    # Casualty locations (red spheres)
    if args.casualty_csv is not None:
        print(f"\nLoading casualty locations from: {args.casualty_csv}")
        casualties_gps = read_casualty_csv(args.casualty_csv)
        print(f"  Casualties: {len(casualties_gps)}")

        # Convert to ENU using same origin as GPS trajectory (chronologically first GPS point)
        casualty_positions = []
        casualty_ids = []
        for cid, lat, lon, alt in casualties_gps:
            e, n, u = pm.geodetic2enu(lat, lon, alt, enu_origin.lat, enu_origin.lon, enu_origin.alt)
            u += args.casualty_height_offset  # Apply height offset
            casualty_positions.append([e, n, u])
            casualty_ids.append(cid)
        casualty_positions = np.array(casualty_positions)

        print("Creating casualty markers (red spheres)...")
        # Apply alpha to color (darker red = more transparent effect in PLY)
        casualty_color = [0.0, 0.0, 1.0]
        for cid, pos in zip(casualty_ids, casualty_positions):
            sphere = create_sphere(pos, args.casualty_radius, casualty_color)
            geometries.append(sphere)
            casualty_meshes.append(sphere)
            print(f"  Casualty {cid}: ENU = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    # Save visualization files
    print(f"\nSaving visualization to: {args.output_path}")

    output_base = os.path.splitext(args.output_path)[0]

    # Save aligned point cloud
    pcd_output = f"{output_base}_pointcloud.ply"
    o3d.io.write_point_cloud(pcd_output, pcd)
    print(f"  Saved point cloud: {pcd_output}")

    # Save trajectory markers (green GPS + blue predicted)
    if trajectory_meshes:
        traj_mesh = o3d.geometry.TriangleMesh()
        for m in trajectory_meshes:
            traj_mesh += m
        traj_output = f"{output_base}_trajectory.ply"
        o3d.io.write_triangle_mesh(traj_output, traj_mesh)
        print(f"  Saved trajectory markers: {traj_output}")

    # Save casualty markers (red spheres)
    if casualty_meshes:
        cas_mesh = o3d.geometry.TriangleMesh()
        for m in casualty_meshes:
            cas_mesh += m
        cas_output = f"{output_base}_casualties.ply"
        o3d.io.write_triangle_mesh(cas_output, cas_mesh)
        print(f"  Saved casualty markers: {cas_output}")

    # Save metrics to text file
    if metrics is not None:
        metrics_output = f"{output_base}_metrics.txt"
        with open(metrics_output, "w") as f:
            f.write("DA3-Streaming Metric Quality Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Scale factor: {metrics['scale']:.6f}\n")
            f.write(f"RMSE after alignment: {metrics['rmse']:.4f} m\n")
            f.write(f"Scale consistency (std): {metrics['scale_error_std']:.4f}\n")
            f.write(f"Mean ATE: {np.mean(metrics['ate']):.4f} m\n")
            f.write(f"Max ATE: {np.max(metrics['ate']):.4f} m\n")
            f.write(f"Median ATE: {np.median(metrics['ate']):.4f} m\n")
            f.write(f"\nAlignment Transform (pred -> gt):\n")
            f.write(f"  Scale: {metrics['scale']}\n")
            f.write(f"  Rotation:\n")
            for row in metrics["rotation"]:
                f.write(f"    [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]\n")
            f.write(f"  Translation: [{metrics['translation'][0]:.4f}, "
                    f"{metrics['translation'][1]:.4f}, {metrics['translation'][2]:.4f}]\n")
        print(f"  Saved metrics: {metrics_output}")

    # Interactive visualization
    if args.visualize:
        print("\nOpening interactive viewer...")
        print("  Green = GPS ground truth")
        print("  Blue = Predicted trajectory")
        print("  Red = Casualty locations")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="DA3-Streaming Metric Quality",
            width=1920,
            height=1080,
        )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
