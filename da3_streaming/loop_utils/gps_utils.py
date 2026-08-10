"""
GPS helpers shared by the offline driver and the ROS2 deployment node.

Extracted from viz_ply_cas.py so that consumers which only need GPS handling
(any_streaming_rt, and anystream_ros' AnyStreamer) do not pull in open3d, which
viz_ply_cas imports at module scope for its visualisation output.

Dependencies here are deliberately limited to numpy + pymap3d.
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import pymap3d as pm
except ImportError:
    raise ImportError("Install pymap3d: pip install pymap3d")


@dataclass
class GpsSample:
    t_ns: int
    t_s: float
    lat: float
    lon: float
    alt: float
    cov_enu: Optional[np.ndarray] = None  # 3x3 ENU covariance, or None if unavailable


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
            cov = None
            if "cov_ee" in r:
                cov = np.array([
                    [float(r["cov_ee"]), float(r["cov_en"]), float(r["cov_eu"])],
                    [float(r["cov_ne"]), float(r["cov_nn"]), float(r["cov_nu"])],
                    [float(r["cov_ue"]), float(r["cov_un"]), float(r["cov_uu"])],
                ], dtype=np.float64)
            rows.append(GpsSample(t_ns=t_ns, t_s=t_s, lat=lat, lon=lon, alt=alt, cov_enu=cov))
    if not rows:
        raise ValueError(f"No rows parsed from {csv_path}")
    return rows


def extract_ts_ns(path: str) -> Optional[int]:
    """Extract timestamp in nanoseconds from image filename."""
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    m = re.search(r"(\d{12,})", stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def build_enu_interpolator(gps_rows: List[GpsSample]):
    """Build an ENU interpolator from GPS samples.

    Returns (interp, meta) where interp is an object with:
      interp(ts_ns) → (e, n, u)
      interp.covariance(ts_ns) → 3x3 ndarray or None (if the source had no covariance)
    """
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

    # Covariance: already in ENU frame from the source, so interpolate element-wise.
    has_cov = gps_sorted[0].cov_enu is not None
    if has_cov:
        cov_flat = np.array([g.cov_enu.ravel() for g in gps_sorted], dtype=np.float64)  # (N, 9)

    class _Interp:
        def __call__(self, ts_ns):
            ts = np.asarray(ts_ns, dtype=np.int64)
            e = np.interp(ts, t_ns, e_arr, left=np.nan, right=np.nan)
            n = np.interp(ts, t_ns, n_arr, left=np.nan, right=np.nan)
            u = np.interp(ts, t_ns, u_arr, left=np.nan, right=np.nan)
            return e, n, u

        def covariance(self, ts_ns):
            if not has_cov:
                return None
            ts = np.asarray(ts_ns, dtype=np.int64)
            interped = np.array([np.interp(ts, t_ns, cov_flat[:, i], left=np.nan, right=np.nan)
                                 for i in range(9)], dtype=np.float64)
            return interped.reshape(3, 3)

    return _Interp(), {"t_ns": t_ns, "origin": (g0.lat, g0.lon, g0.alt)}


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
