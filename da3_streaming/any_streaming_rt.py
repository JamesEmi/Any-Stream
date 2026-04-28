# Copyright (c) 2025 CMU Airlab and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import gc
import glob
import json
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import rerun as rr
import torch
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
    precompute_scale_chunks_with_depth,
    warmup_numba,
)

from safetensors.torch import load_file
from depth_anything_3.api import DepthAnything3
from viz_ply_cas import read_gps_csv, build_enu_interpolator, extract_ts_ns, umeyama_alignment
from eval.pose_utils import load_poses, save_poses

# OpenCV world (Y-down, Z-forward) → Z-up viz. Applied at log time to all
# model-frame quantities (pointcloud, trajectory, camera Transform3D).
# Matches the pattern in /home/ubuntu/anyslam/lingbot-map/demo.py.
R_Z_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


class Any_StreamingRT:
    def __init__(self, image_dir, save_dir, config, gps_csv=None, poses=None,
                 frame_stride=1):
        self.config = config
        self.poses_path = poses
        self.frame_stride = max(1, int(frame_stride))

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir

        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(self.pcd_dir, exist_ok=True)

        # self.all_camera_poses = []
        # self.all_camera_intrinsics = []
        # self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        print("Loading model...")
        t_load = time.time()

        model_type = self.config["Weights"]["model"]

        self.acc_pts: list = []     # list of [M,3] float32 arrays (global model frame)
        self.acc_cols: list = []    # list of [M,3] uint8 arrays
        self._acc_chunk_sim3: list = []  # (chunk_idx, s, R, t) used when accumulating acc_pts[i]
        self.sim3_list: list = []   # relative (s,R,t) between consecutive chunks
        self.prev_predictions = None
        self.chunk_indices: list = []
        self._rr_time = 0

        # GPS alignment state
        self.gps_csv = gps_csv
        self.gps_interp = None
        self.gps_meta = None
        if gps_csv is not None:
            gps_rows = read_gps_csv(gps_csv)
            self.gps_interp, self.gps_meta = build_enu_interpolator(gps_rows)
            print(f"GPS loaded: {len(gps_rows)} samples")
        self._poses_cache: np.ndarray | None = None        # loaded once, reused every chunk
        self._intrinsics_prior: np.ndarray | None = None  # set after chunk 0 when Model.inject_pred_intrinsics=true
        self._inject_pred_intrinsics: bool = bool(
            self.config.get("Model", {}).get("inject_pred_intrinsics", False)
        )
        self.acc_cam_positions: list = []   # list of [K,3] arrays, non-overlap camera positions
        self.frame_image_paths: list = []   # parallel to acc_cam_positions (flattened)
        self.acc_frame_c2w: list = []       # list of [K,4,4] c2w poses per chunk (world frame)
        self._acc_frame_c2w_local: list = []  # list of [K,4,4] c2w in chunk-local coords (for PGO recompute)
        self.acc_frame_global_indices: list = []  # list of global frame indices (flattened)

        if model_type == "DA3":
            with open(self.config["Weights"]["DA3_CONFIG"]) as f:
                config = json.load(f)
            self.model = DepthAnything3(**config)
            weight = load_file(self.config["Weights"]["DA3"])
            self.model.load_state_dict(weight, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)

        elif model_type == "MapAnything":
            from adapters.mapanything import MapAnythingAdapter
            self.model = MapAnythingAdapter(device=self.device)
            self.model.load()

        elif model_type == "VGGT":
            from adapters.vggt import VGGTAdapter
            self.model = VGGTAdapter(device=self.device)
            self.model.load()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model_type = model_type
        load_time = time.time() - t_load
        print(f"Model loaded in {load_time:.2f}s")

        self.log_file = os.path.join(save_dir, "timing.log")
        with open(self.log_file, "w") as f:
            f.write(f"model_load: {load_time:.2f}s\n")

        self.total_infer_time = 0.0
        self.total_align_time = 0.0
        self.start_time = time.time()

        self.chunk_indicess: list = []

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]
        # self.loop_predict_list = []
        # # self.loop_enable = self.config["Model"]["loop_enable"]

        # if self.loop_enable:
        #     loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
        #     self.loop_detector = LoopDetector(
        #         image_dir=image_dir, output=loop_info_save_path, config=self.config
        #     )
        #     self.loop_detector.load_model()

        print("init done.")

    def _infer_chunk(self, image_paths, chunk_idx):
        print(f"Loaded {len(image_paths)} images")

        ref_view_strategy = self.config["Model"]["ref_view_strategy"]
        t0 = time.time()
        torch.cuda.empty_cache()
        with torch.no_grad():
            if self.model_type == "DA3":
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model.inference(
                        image_paths,
                        ref_view_strategy=ref_view_strategy
                    )
                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0  # Conf correction for DA3

            elif self.model_type == "MapAnything":
                predictions = self.model.infer(
                    image_paths,
                    intrinsics_prior=self._intrinsics_prior,
                )

            elif self.model_type == "VGGT":
                predictions = self.model.infer(image_paths)

        infer_time = time.time() - t0
        self.total_infer_time += infer_time
        print(f"  [chunk {chunk_idx}] inference: {infer_time:.2f}s  mean depth: {np.mean(predictions.depth):.3f}")

        # After the first chunk, freeze a per-sequence intrinsics prior for MA reuse.
        if (self._inject_pred_intrinsics
                and self.model_type == "MapAnything"
                and self._intrinsics_prior is None
                and predictions.intrinsics is not None
                and predictions.intrinsics.shape[0] > 0):
            K_stack = np.asarray(predictions.intrinsics, dtype=np.float64)  # (N, 3, 3)
            K_prior = np.median(K_stack, axis=0)
            K_prior[2] = [0.0, 0.0, 1.0]   # enforce exact bottom row
            self._intrinsics_prior = K_prior
            fx, fy = K_prior[0, 0], K_prior[1, 1]
            cx, cy = K_prior[0, 2], K_prior[1, 2]
            print(f"  [inject_pred_intrinsics] prior from chunk {chunk_idx} "
                  f"(median of {K_stack.shape[0]} frames): "
                  f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        with open(self.log_file, "a") as f:
            f.write(f"chunk_{chunk_idx}_infer: {infer_time:.2f}s\n")
        torch.cuda.empty_cache()
        return predictions

    def _align_chunks(self, pred1, pred2, chunk_idx):
        # Build point maps from depth (same as lines 609-625 of original)
        pm1 = depth_to_point_cloud_vectorized(pred1.depth, pred1.intrinsics, pred1.extrinsics)
        pm2 = depth_to_point_cloud_vectorized(pred2.depth, pred2.intrinsics, pred2.extrinsics)

        ov = self.overlap
        point_map1 = pm1[-ov:]
        point_map2 = pm2[:ov]
        conf1 = pred1.conf[-ov:]
        conf2 = pred2.conf[:ov]

        # scale+se3 branch (copy from align_2pcds lines 385-397)
        align_method = self.config["Model"]["align_method"]
        scale_factor = None
        if align_method == "scale+se3":
            d1 = np.squeeze(pred1.depth[-ov:])
            d2 = np.squeeze(pred2.depth[:ov])
            c1 = np.squeeze(pred1.conf[-ov:])
            c2 = np.squeeze(pred2.conf[:ov])
            scale_factor, quality, method = precompute_scale_chunks_with_depth(
                d1, c1, d2, c2, method=self.config["Model"]["scale_compute_method"]
            )

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        t0 = time.time()
        s, R, t = weighted_align_point_maps(
            point_map1, conf1, point_map2, conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
        )
        align_time = time.time() - t0
        self.total_align_time += align_time
        print(f"  [align] chunk {chunk_idx-1}→{chunk_idx}: s={s:.4f} ({align_time:.2f}s)")
        with open(self.log_file, "a") as f:
            f.write(f"chunk_{chunk_idx-1}_to_{chunk_idx}_align: {align_time:.2f}s\n")
        return s, R, t

    def _build_filter_mask(self, predictions, sl) -> np.ndarray:
        """
        Build a boolean mask [N, H, W] for the sliced frames.

        Strategy:
          - If predictions.mask exists (MapAnything) → start with that mask
          - If use_conf_filtering: True in config OR no mask exists → apply conf threshold
          Both conditions can stack: mask AND conf both applied when use_conf_filtering=True.
        """
        conf = predictions.conf[sl]
        use_conf = self.config["Model"]["Pointcloud_Save"].get("use_conf_filtering", True)
        mask_src = getattr(predictions, 'mask', None)

        if mask_src is not None:
            # MapAnything non-ambiguous mask
            filter_mask = mask_src[sl].astype(bool)
            if use_conf:
                conf_thresh = (np.mean(conf)
                               * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"])
                filter_mask = filter_mask & (conf >= conf_thresh)

        else:
            # DA3 / Pi3: confidence only
            conf_thresh = (np.mean(conf)
                           * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"])
            filter_mask = conf >= conf_thresh

        return filter_mask  # [N, H, W] bool

    
    @staticmethod
    def _w2c_to_camera_positions(extrinsics_w2c: np.ndarray) -> np.ndarray:
        """Convert W2C extrinsics [N, 3, 4] to camera positions [N, 3] in world frame."""
        N = extrinsics_w2c.shape[0]
        w2c_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
        w2c_4x4[:, :3, :4] = extrinsics_w2c
        w2c_4x4[:, 3, 3] = 1.0
        c2w = np.linalg.inv(w2c_4x4)
        return c2w[:, :3, 3]

    @staticmethod
    def _w2c_to_c2w(extrinsics_w2c: np.ndarray) -> np.ndarray:
        """Convert W2C extrinsics [N, 3, 4] to C2W [N, 4, 4]."""
        N = extrinsics_w2c.shape[0]
        w2c_4x4 = np.zeros((N, 4, 4), dtype=np.float64)
        w2c_4x4[:, :3, :4] = extrinsics_w2c
        w2c_4x4[:, 3, 3] = 1.0
        return np.linalg.inv(w2c_4x4)

    @staticmethod
    def _apply_sim3_to_c2w(c2w: np.ndarray, s, R, t) -> np.ndarray:
        """Apply Sim(3) transform to c2w poses [N, 4, 4]. Returns [N, 4, 4]."""
        out = c2w.copy()
        # Transform position: p' = s * R @ p + t
        out[:, :3, 3] = (s * (c2w[:, :3, 3] @ R.T) + t)
        # Transform rotation: R' = R_sim3 @ R_c2w (scale doesn't affect rotation)
        out[:, :3, :3] = np.einsum("ij,njk->nik", R, c2w[:, :3, :3])
        return out

    def _extract_new_points(self, predictions, chunk_idx: int, s_abs, R_abs, t_abs,
                            chunk_paths: list = None):
        """
        Extract world-space points for the non-overlap frames of this chunk,
        apply cumulative SIM(3), filter (mask and/or conf), and subsample.
        Returns (pts [M,3], cols [M,3], cam_pos [K,3], frame_paths [K]).
        """
        n = len(predictions.depth)
        ov = self.overlap

        if chunk_idx == 0:
            sl = slice(0, n - ov)
        else:
            sl = slice(ov, n)

        depth  = predictions.depth[sl]
        images = predictions.processed_images[sl]

        if len(depth) == 0:
            empty_pts = np.zeros((0, 3), dtype=np.float32)
            empty_cols = np.zeros((0, 3), dtype=np.uint8)
            empty_cam = np.zeros((0, 3), dtype=np.float32)
            return empty_pts, empty_cols, empty_cam, []

        # Get world points — use model's own if available, else backproject
        if getattr(predictions, 'world_points', None) is not None:
            pts = predictions.world_points[sl]  # [N, H, W, 3]
        else:
            K = predictions.intrinsics[sl]
            E = predictions.extrinsics[sl]
            pts = depth_to_point_cloud_vectorized(depth, K, E)  # [N, H, W, 3]

        # Extract camera positions for non-overlap frames
        cam_pos = self._w2c_to_camera_positions(predictions.extrinsics[sl])  # [K, 3]

        # Apply cumulative SIM(3) for all chunks after the first
        if chunk_idx > 0:
            pts = apply_sim3_direct_torch(pts, s_abs, R_abs, t_abs)  # [N, H, W, 3] numpy
            cam_pos = (s_abs * (cam_pos @ R_abs.T) + t_abs).astype(np.float32)

        # Non-overlap frame paths
        if chunk_paths is not None:
            if chunk_idx == 0:
                frame_paths = chunk_paths[:n - ov]
            else:
                frame_paths = chunk_paths[ov:]
        else:
            frame_paths = []

        # Build filter mask (handles mask vs conf logic)
        filter_mask = self._build_filter_mask(predictions, sl)  # [N, H, W] bool

        pts_flat    = pts.reshape(-1, 3)
        cols_flat   = images.reshape(-1, 3).astype(np.uint8)
        mask_flat   = filter_mask.reshape(-1)

        pts_valid  = pts_flat[mask_flat]
        cols_valid = cols_flat[mask_flat]

        # Subsample - TODO - be careful about what value to pick here for MapAnything vs DA3
        sample_ratio = self.config["Model"]["Pointcloud_Save"]["sample_ratio"]
        if len(pts_valid) > 0 and sample_ratio < 1.0:
            n_keep = max(1, int(len(pts_valid) * sample_ratio))
            idx = np.random.choice(len(pts_valid), size=n_keep, replace=False)
            pts_valid  = pts_valid[idx]
            cols_valid = cols_valid[idx]

        return pts_valid, cols_valid, cam_pos, frame_paths
    
    def _log_to_rerun(self):
        if not self.acc_pts:
            return
        all_pts  = np.concatenate(self.acc_pts,  axis=0)
        all_cols = np.concatenate(self.acc_cols, axis=0)

        # Cap for Rerun perf
        max_pts = 3_000_000
        if len(all_pts) > max_pts:
            idx = np.random.choice(len(all_pts), size=max_pts, replace=False)
            all_pts  = all_pts[idx]
            all_cols = all_cols[idx]

        rr.set_time("stable_time", sequence=self._rr_time)
        self._rr_time += 1
        rr.log("map/pointcloud", rr.Points3D(
            positions=(R_Z_UP @ all_pts.T).T, colors=all_cols
        ))
        print(f"  [rerun] {len(all_pts):,} pts logged  "
              f"(total acc: {sum(len(p) for p in self.acc_pts):,})")

        # Rolling predicted trajectory (model frame, white)
        if self.acc_cam_positions:
            traj = np.concatenate(self.acc_cam_positions, axis=0).astype(np.float32)
            traj_viz = (R_Z_UP @ traj.T).T
            rr.log("trajectories/pred", rr.Points3D(
                positions=traj_viz,
                colors=np.full((len(traj_viz), 3), [255, 255, 255], dtype=np.uint8),
            ))
            if len(traj_viz) >= 2:
                rr.log("trajectories/pred_line", rr.LineStrips3D([traj_viz], colors=[[255, 255, 255]]))

    def _compute_gps_alignment(self):
        """
        Compute Umeyama alignment from accumulated local camera positions to GPS ENU.
        Returns (s, R, t, local_matched, gps_matched) or None.
        """
        if self.gps_interp is None or not self.acc_cam_positions:
            return None

        all_cam_pos = np.concatenate(self.acc_cam_positions, axis=0)

        local_matched = []
        gps_matched = []
        for i, path in enumerate(self.frame_image_paths):
            ts = extract_ts_ns(path)
            if ts is None:
                continue
            e, n, u = self.gps_interp(ts)
            if np.isfinite(e) and np.isfinite(n) and np.isfinite(u):
                local_matched.append(all_cam_pos[i])
                gps_matched.append([float(e), float(n), float(u)])

        if len(local_matched) < 3:
            print(f"  [GPS] Only {len(local_matched)} matches, need >= 3. Skipping.")
            return None

        local_matched = np.array(local_matched, dtype=np.float64)
        gps_matched = np.array(gps_matched, dtype=np.float64)

        s, R, t = umeyama_alignment(local_matched, gps_matched, with_scale=True)

        aligned = s * (local_matched @ R.T) + t
        rmse = np.sqrt(np.mean(np.sum((aligned - gps_matched) ** 2, axis=1)))
        print(f"  [GPS] Umeyama: s={s:.4f}, RMSE={rmse:.3f}m, {len(local_matched)} matches")

        return s, R, t, local_matched, gps_matched

    def _apply_gps_transform(self, s_gps, R_gps, t_gps):
        """Apply GPS SIM(3) to all accumulated points. Returns (global_pts, global_cols)."""
        if not self.acc_pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        all_pts = np.concatenate(self.acc_pts, axis=0)
        all_cols = np.concatenate(self.acc_cols, axis=0)
        global_pts = (s_gps * (all_pts.astype(np.float64) @ R_gps.T) + t_gps).astype(np.float32)
        return global_pts, all_cols

    def _log_to_rerun_gps(self, s_gps, R_gps, t_gps, gps_matched):
        """Log GPS-aligned point cloud and trajectories to Rerun."""
        global_pts, global_cols = self._apply_gps_transform(s_gps, R_gps, t_gps)

        max_pts = 3_000_000
        if len(global_pts) > max_pts:
            idx = np.random.choice(len(global_pts), size=max_pts, replace=False)
            global_pts = global_pts[idx]
            global_cols = global_cols[idx]

        rr.set_time("stable_time", sequence=self._rr_time - 1)  # same timestep as local log
        rr.log("map/global_pointcloud", rr.Points3D(positions=global_pts, colors=global_cols))

        # Predicted trajectory in GPS frame (blue)
        all_cam = np.concatenate(self.acc_cam_positions, axis=0)
        cam_global = (s_gps * (all_cam.astype(np.float64) @ R_gps.T) + t_gps).astype(np.float32)
        rr.log("map/trajectory_pred", rr.Points3D(
            positions=cam_global,
            colors=np.full((len(cam_global), 3), [0, 0, 255], dtype=np.uint8),
        ))
        if len(cam_global) >= 2:
            rr.log("map/trajectory_pred_line", rr.LineStrips3D(
                [cam_global], colors=[[0, 0, 255]],
            ))

        # GPS ground truth (green)
        gps_f32 = gps_matched.astype(np.float32)
        rr.log("map/trajectory_gps", rr.Points3D(
            positions=gps_f32,
            colors=np.full((len(gps_f32), 3), [0, 255, 0], dtype=np.uint8),
        ))
        if len(gps_f32) >= 2:
            rr.log("map/trajectory_gps_line", rr.LineStrips3D(
                [gps_f32], colors=[[0, 255, 0]],
            ))

        print(f"  [rerun/gps] {len(global_pts):,} pts, "
              f"{len(cam_global)} pred poses, {len(gps_f32)} gps poses")

    def _save_poses(self, out_path: str):
        """Save accumulated per-frame c2w poses in KITTI format (3x4 per line)."""
        if not self.acc_frame_c2w:
            print("  No poses to save.")
            return
        all_c2w = np.concatenate(self.acc_frame_c2w, axis=0)  # [N, 4, 4]
        save_poses(all_c2w, out_path)
        print(f"Saved {len(all_c2w)} poses → {out_path}")

    def _recompute_poses_from_sim3(self):
        """Recompute acc_frame_c2w from stored per-chunk local c2w and updated sim3_list.
        Called after PGO to update poses with optimized transforms.
        """
        if not self.sim3_list:
            return
        cumulative = accumulate_sim3_transforms(self.sim3_list)
        new_c2w = []
        for i, c2w_local in enumerate(self._acc_frame_c2w_local):
            if i == 0:
                new_c2w.append(c2w_local.copy())  # copy to avoid aliasing _acc_frame_c2w_local[0]
            else:
                s_abs, R_abs, t_abs = cumulative[i - 1]
                new_c2w.append(self._apply_sim3_to_c2w(c2w_local, s_abs, R_abs, t_abs))
        self.acc_frame_c2w = new_c2w

    def _retransform_pointcloud(self):
        """Retransform acc_pts from pre-PGO chunk transforms to post-PGO ones.

        _acc_chunk_sim3[i] = (chunk_idx, s_old, R_old, t_old) records the absolute
        transform used when points were accumulated. After PGO updates sim3_list,
        we undo the old transform (back to local chunk frame) and apply the new one.
        """
        if not self.sim3_list or not self._acc_chunk_sim3:
            return
        cumulative = accumulate_sim3_transforms(self.sim3_list)
        # cumulative[k] = absolute transform for chunk k+1; chunk 0 is identity
        def new_transform(chunk_idx):
            if chunk_idx == 0:
                return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            s, R, t = cumulative[chunk_idx - 1]
            return float(s), R.astype(np.float64), t.astype(np.float64)

        new_pts = []
        new_sim3 = []
        for pts, (ci, s_old, R_old, t_old) in zip(self.acc_pts, self._acc_chunk_sim3):
            s_old = float(s_old)
            R_old = R_old.astype(np.float64)
            t_old = t_old.astype(np.float64)
            # undo old: pts_local = (1/s_old) * R_old.T @ (pts - t_old)
            pts_local = (1.0 / s_old) * ((pts.astype(np.float64) - t_old) @ R_old)
            s_new, R_new, t_new = new_transform(ci)
            pts_new = (s_new * (pts_local @ R_new.T) + t_new).astype(np.float32)
            new_pts.append(pts_new)
            new_sim3.append((ci, s_new, R_new.astype(np.float32), t_new.astype(np.float32)))

        self.acc_pts = new_pts
        self._acc_chunk_sim3 = new_sim3

    def _apply_gps_absolutes(self, abs_gps):
        """Update acc_frame_c2w, acc_cam_positions, acc_pts and _acc_chunk_sim3
        in place using per-chunk absolute GPS-frame Sim3s from optimize_gps.

        abs_gps[k] = (s_k_gps, R_k_gps, t_k_gps), where s_k_gps = s_g * s_k^m
        is the chunk's metric scale (constant — PGO does not touch chunk
        interiors), and (R_k_gps, t_k_gps) is the post-PGO chunk anchor in
        GPS frame.

        After this call, every acc_* lives in GPS frame and self.sim3_list is
        unchanged (still the depth-model relative chain in model frame).
        """
        assert len(abs_gps) == len(self._acc_frame_c2w_local), (
            f"abs_gps len {len(abs_gps)} != local frames len {len(self._acc_frame_c2w_local)}"
        )

        # 1) Recompute per-frame c2w and per-chunk camera positions in GPS frame.
        new_c2w = []
        new_cam_pos = []
        for k, c2w_local in enumerate(self._acc_frame_c2w_local):
            s_k, R_k, t_k = abs_gps[k]
            c2w_gps = self._apply_sim3_to_c2w(
                c2w_local,
                float(s_k),
                R_k.astype(np.float64),
                t_k.astype(np.float64),
            )
            new_c2w.append(c2w_gps)
            new_cam_pos.append(c2w_gps[:, :3, 3].astype(np.float32))
        self.acc_frame_c2w = new_c2w
        self.acc_cam_positions = new_cam_pos

        # 2) Retransform acc_pts: undo whatever transform was last applied
        # (frame-agnostic), then apply the new GPS-frame chunk transform.
        if not self._acc_chunk_sim3:
            return

        new_pts = []
        new_chunk_sim3 = []
        for pts, (ci, s_old, R_old, t_old) in zip(self.acc_pts, self._acc_chunk_sim3):
            s_old = float(s_old)
            R_old = R_old.astype(np.float64)
            t_old = t_old.astype(np.float64)
            # undo old: pts_local = (1/s_old) * R_old.T @ (pts - t_old)
            pts_local = (1.0 / s_old) * ((pts.astype(np.float64) - t_old) @ R_old)

            s_new, R_new, t_new = abs_gps[ci]
            s_new = float(s_new)
            R_new_d = R_new.astype(np.float64)
            t_new_d = t_new.astype(np.float64)

            pts_new = (s_new * (pts_local @ R_new_d.T) + t_new_d).astype(np.float32)
            new_pts.append(pts_new)
            new_chunk_sim3.append(
                (ci, s_new, R_new_d.astype(np.float32), t_new_d.astype(np.float32))
            )

        self.acc_pts = new_pts
        self._acc_chunk_sim3 = new_chunk_sim3

    def _log_gt_rerun_gps_frame(self, poses: np.ndarray):
        """
        Log GT trajectory directly in GPS frame (no Umeyama inverse).

        Used after GPS-PGO when acc_* already lives in GPS frame.
        """
        n_gt = min(len(poses), sum(len(p) for p in self.acc_cam_positions))
        gt_pos_all = poses[:n_gt, :3, 3].astype(np.float32)
        rr.log("trajectories/gt", rr.Points3D(
            positions=gt_pos_all,
            colors=np.full((len(gt_pos_all), 3), [0, 255, 0], dtype=np.uint8),
        ), static=True)
        if len(gt_pos_all) >= 2:
            rr.log("trajectories/gt_line", rr.LineStrips3D(
                [gt_pos_all], colors=[[0, 255, 0]],
            ), static=True)

    def _log_gps_anchors_rerun(self, anchor_positions: np.ndarray):
        """
        Log per-chunk GPS anchor positions used as PGO unaries (red markers).

        anchor_positions: [K, 3] array of GPS-frame positions, one per
        anchored chunk. NaN rows are filtered out.
        """
        if anchor_positions is None or len(anchor_positions) == 0:
            return
        valid = np.isfinite(anchor_positions).all(axis=1)
        pts = anchor_positions[valid].astype(np.float32)
        if len(pts) == 0:
            return
        rr.log("trajectories/gps_anchors", rr.Points3D(
            positions=pts,
            colors=np.full((len(pts), 3), [255, 0, 0], dtype=np.uint8),
            radii=np.full(len(pts), 0.5, dtype=np.float32),
        ), static=True)

    def _log_gps_traj_rerun_gps_frame(self):
        """Log GPS CSV trajectory (raw ENU) as a static green line.

        Used after the GPS-frame refactor — no Umeyama inverse needed because
        acc_* now lives in GPS frame too.
        """
        gps_enu = []
        for path in self.frame_image_paths:
            ts = extract_ts_ns(path)
            if ts is None:
                continue
            e, n, u = self.gps_interp(ts)
            if np.isfinite(e) and np.isfinite(n) and np.isfinite(u):
                gps_enu.append([float(e), float(n), float(u)])
        if len(gps_enu) < 2:
            return
        gps_enu = np.array(gps_enu, dtype=np.float32)
        rr.log("trajectories/gt", rr.Points3D(
            positions=gps_enu,
            colors=np.full((len(gps_enu), 3), [0, 255, 0], dtype=np.uint8),
        ), static=True)
        rr.log("trajectories/gt_line", rr.LineStrips3D(
            [gps_enu], colors=[[0, 255, 0]],
        ), static=True)

    def _fit_gt_alignment(self, poses: np.ndarray):
        """Compute Umeyama alignment from predicted chunk positions to GT.

        Returns (s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales)
        where  s_g * R_g @ pred + t_g ≈ gt  (model → GPS frame).

        Only uses chunks that have actually been processed (len(sim3_list) + 1), so this
        is safe to call mid-stream without contaminating the fit with extrapolated future positions.
        """
        n_chunks_total = len(self.chunk_indices)
        # Cap to the number of processed chunks: sim3_list has one entry per chunk-to-chunk
        # transition, so processed count = len(sim3_list) + 1.
        n_chunks = min(n_chunks_total, len(self.sim3_list) + 1)

        chunk_rep_frames = []
        for ci, (start, end) in enumerate(self.chunk_indices[:n_chunks]):
            ov = self.overlap
            rep = (start + (end - ov - start) // 2) if ci == 0 else \
                  ((start + ov) + (end - start - ov) // 2)
            chunk_rep_frames.append(min(rep, len(poses) - 1))

        chunk_gt = np.array([poses[f] for f in chunk_rep_frames])
        chunk_gt_pos = chunk_gt[:, :3, 3]
        chunk_gt_rot = chunk_gt[:, :3, :3]

        cumulative = accumulate_sim3_transforms(self.sim3_list) if self.sim3_list else []
        pred_positions = [np.zeros(3, dtype=np.float64)]
        pred_scales    = [1.0]
        for (s_c, R_c, t_c) in cumulative[:n_chunks - 1]:  # only processed transitions
            pred_positions.append(t_c.astype(np.float64))
            pred_scales.append(float(s_c))
        pred_pos    = np.array(pred_positions[:n_chunks])
        pred_scales = pred_scales[:n_chunks]

        s_g, R_g, t_g = umeyama_alignment(pred_pos, chunk_gt_pos, with_scale=True)
        return s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales

    def _log_gt_rerun(self, poses: np.ndarray,
                      s_g: float, R_g: np.ndarray, t_g: np.ndarray):
        """Log GT trajectory in model frame as a static green line."""
        n_gt = min(len(poses), sum(len(p) for p in self.acc_cam_positions))
        gt_pos_all = poses[:n_gt, :3, 3]
        gt_model = ((1.0 / s_g) * ((gt_pos_all - t_g) @ R_g)).astype(np.float32)
        rr.log("trajectories/gt", rr.Points3D(
            positions=gt_model,
            colors=np.full((len(gt_model), 3), [0, 255, 0], dtype=np.uint8),
        ), static=True)
        if len(gt_model) >= 2:
            rr.log("trajectories/gt_line", rr.LineStrips3D(
                [gt_model], colors=[[0, 255, 0]],
            ), static=True)

    def _shift_all_poses(self, delta: np.ndarray):
        """Rigidly shift acc_cam_positions, acc_frame_c2w, and acc_pts by delta (3,).

        Used to move the model origin to the GPS start position before PGO, so that
        pred and GPS share the same starting point in all visualisations.
        sim3_list is NOT touched — relative transforms are invariant to global translation.
        """
        delta32 = delta.astype(np.float32)
        for cam_pos in self.acc_cam_positions:
            cam_pos += delta32
        for c2w_chunk in self.acc_frame_c2w:
            c2w_chunk[:, :3, 3] += delta32
        self.acc_pts = [p + delta32 for p in self.acc_pts]

    def _log_gps_traj_rerun(self, s_g: float, R_g: np.ndarray, t_g: np.ndarray):
        """Log GPS CSV trajectory (ENU → model frame) as a static green line."""
        gps_enu = []
        for path in self.frame_image_paths:
            ts = extract_ts_ns(path)
            if ts is None:
                continue
            e, n, u = self.gps_interp(ts)
            if np.isfinite(e) and np.isfinite(n) and np.isfinite(u):
                gps_enu.append([float(e), float(n), float(u)])
        if len(gps_enu) < 2:
            return
        gps_enu = np.array(gps_enu, dtype=np.float64)
        gt_model = ((1.0 / s_g) * ((gps_enu - t_g) @ R_g)).astype(np.float32)
        rr.log("trajectories/gt", rr.Points3D(
            positions=gt_model,
            colors=np.full((len(gt_model), 3), [0, 255, 0], dtype=np.uint8),
        ), static=True)
        if len(gt_model) >= 2:
            rr.log("trajectories/gt_line", rr.LineStrips3D(
                [gt_model], colors=[[0, 255, 0]],
            ), static=True)

    def _fit_gps_csv_alignment(self):
        """Compute Umeyama alignment from predicted chunk positions to GPS CSV ENU.

        Returns (s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales)
        with the same layout as _fit_gt_alignment.  chunk_gt_rot is identity (GPS has
        no orientation); chunk_gt_pos[k] is NaN if GPS is unavailable for that chunk.
        Returns None if fewer than 3 valid GPS observations exist.
        """
        n_chunks_total = len(self.chunk_indices)
        n_chunks = min(n_chunks_total, len(self.sim3_list) + 1)

        # Build global_frame_index → image_path mapping from accumulated lists
        gfi_to_path = {
            gfi: self.frame_image_paths[i]
            for i, gfi in enumerate(self.acc_frame_global_indices)
        }

        chunk_gt_pos = np.full((n_chunks, 3), np.nan, dtype=np.float64)
        chunk_gt_rot = np.tile(np.eye(3, dtype=np.float64), (n_chunks, 1, 1))

        for ci, (start, end) in enumerate(self.chunk_indices[:n_chunks]):
            ov = self.overlap
            rep = (start + (end - ov - start) // 2) if ci == 0 else \
                  ((start + ov) + (end - start - ov) // 2)
            path = gfi_to_path.get(rep)
            if path is None:
                continue
            ts = extract_ts_ns(path)
            if ts is None:
                continue
            e, n, u = self.gps_interp(ts)
            if np.isfinite(e) and np.isfinite(n) and np.isfinite(u):
                chunk_gt_pos[ci] = [float(e), float(n), float(u)]

        cumulative = accumulate_sim3_transforms(self.sim3_list) if self.sim3_list else []
        pred_positions = [np.zeros(3, dtype=np.float64)]
        pred_scales    = [1.0]
        for (s_c, R_c, t_c) in cumulative[:n_chunks - 1]:
            pred_positions.append(t_c.astype(np.float64))
            pred_scales.append(float(s_c))
        pred_pos    = np.array(pred_positions[:n_chunks])
        pred_scales = pred_scales[:n_chunks]

        valid = np.all(np.isfinite(chunk_gt_pos), axis=1)
        n_valid = int(valid.sum())
        print(f"  [GPS-CSV] {n_valid}/{n_chunks} chunks have valid GPS positions")
        if n_valid < 3:
            print("  [GPS-CSV] Not enough valid GPS points for Umeyama fit. Skipping PGO.")
            return None

        s_g, R_g, t_g = umeyama_alignment(pred_pos[valid], chunk_gt_pos[valid], with_scale=True)
        print(f"  [GPS-CSV] Umeyama: scale={s_g:.4f}  "
              f"|R_g-I|={np.linalg.norm(R_g - np.eye(3)):.4f}  "
              f"|t_g|={np.linalg.norm(t_g):.3f}m")
        return s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales

    def _run_gps_pgo(self, alignment_data: tuple, poses=None):
        """GPS-based Pose Graph Optimization using position-only anchor constraints.

        Constrains only the translation of each anchored chunk; rotation and scale
        are governed solely by the sequential (odometry) edges and remain free.

        alignment_data: 7-tuple returned by _fit_gt_alignment or _fit_gps_csv_alignment:
          (s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales)
        poses: if provided, GT trajectory is logged to Rerun after PGO.

        Umeyama: s_g * R_g @ pred + t_g ≈ gt  maps model frame → GPS (metric) frame.
        Inverse: pos_model = (1/s_g) * R_g.T @ (pos_gps - t_g)
        """
        from loop_utils.sim3loop import Sim3LoopOptimizer

        print("\n=== GPS-PGO ===")
        s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales = alignment_data
        print(f"  [GPS-PGO] Umeyama: scale={s_g:.4f}  "
              f"|R_g-I|={np.linalg.norm(R_g - np.eye(3)):.4f}  "
              f"|t_g|={np.linalg.norm(t_g):.3f}m")

        if not self.sim3_list:
            print("  sim3_list empty (single chunk). Skipping PGO.")
            return

        n_chunks = len(self.chunk_indices)
        pgo_cfg = self.config.get("GPS_PGO", {})
        gps_every_k = pgo_cfg.get("gps_every_k", 5)

        # ── Build per-frame 5-DOF GPS measurements ─────────
        # Each measurement binds chunk k's Similarity3 to one GPS sample:
        #   p_obs: GPS position in metric frame
        #   v_obs: GPS heading unit vector (adjacent-sample diff)
        #   c_loc: frame camera position in chunk-local coords
        #   v_loc: local camera motion direction (adjacent-sample diff)

        gps_every_n_frames = pgo_cfg.get("gps_anchor_freq", 1)
        gps_measurements = []
        ov = self.overlap

        gps_pos_all = np.full((max(self.acc_frame_global_indices) + 2, 3), np.nan,
                              dtype=np.float64)
        gps_cov_all = None  # None → isotropic fallback (KITTI); ndarray → per-frame (drone)

        # Need per-frame GPS positions.
        if poses is not None:
            # KITTI - per frame GT poses, dense.
            n_poses = min(poses.shape[0], gps_pos_all.shape[0])
            gps_pos_all[:n_poses] = poses[:n_poses, :3, 3].astype(np.float64)
        elif self.gps_interp is not None:
            # drone - interpolate gps csv per frame via timestamp
            gps_cov_all = np.full((gps_pos_all.shape[0], 3, 3), np.nan, dtype=np.float64) 
            # gfi_to_path maps global-frame-index → image path (for ts extraction).
            gfi_to_path = {
                gfi: self.frame_image_paths[i]
                for i, gfi in enumerate(self.acc_frame_global_indices)
            }
            for gfi, path in gfi_to_path.items():
                ts = extract_ts_ns(path)
                if ts is None:
                    continue
                e, n, u = self.gps_interp(ts)
                if np.isfinite(e) and np.isfinite(n) and np.isfinite(u):
                    gps_pos_all[gfi] = [float(e), float(n), float(u)]
                    cov = self.gps_interp.covariance(ts)
                    if cov is None:
                        raise RuntimeError(
                            f"GPS CSV path active but no position_covariance at ts={ts}. "
                            f"Ensure the GPS CSV includes covariance columns and "
                            f"build_enu_interpolator exposes .covariance(ts)."
                        )
                    gps_cov_all[gfi] = np.asarray(cov, dtype=np.float64).reshape(3, 3)

        else:
            print("  [GPS-PGO-Sim3] No GPS source (poses=None, gps_interp=None). Skipping.")
            return
        

        for k, (start, end) in enumerate(self.chunk_indices):
            c2w_local = self._acc_frame_c2w_local[k]  # (n_local, 4, 4)
            n_local = c2w_local.shape[0]
            if n_local < 2:
                continue
            # local→global frame mapping for this chunk (matches accumulation logic)
            if k == 0:
                global_start = start
            else:
                global_start = start + ov

            c_loc_all = c2w_local[:, :3, 3].astype(np.float64)   # (n_local, 3)
            # local velocity direction by central diff (fwd/bwd at boundaries)
            v_loc_all = np.zeros_like(c_loc_all)
            v_loc_all[1:-1] = c_loc_all[2:] - c_loc_all[:-2]
            v_loc_all[0] = c_loc_all[1] - c_loc_all[0]
            v_loc_all[-1] = c_loc_all[-1] - c_loc_all[-2]

            for i_local in range(0, n_local, gps_every_n_frames):
                g = global_start + i_local
                # need g-1 and g+1 for GPS heading differencing
                if g <= 0 or g >= gps_pos_all.shape[0] - 1:
                    continue
                p_obs = gps_pos_all[g]
                p_prev = gps_pos_all[g - 1]
                p_next = gps_pos_all[g + 1]
                if not (np.all(np.isfinite(p_obs)) and
                        np.all(np.isfinite(p_prev)) and
                        np.all(np.isfinite(p_next))):
                    continue
                v_obs_raw = p_next - p_prev

                # Per-measurement covariance: use sensor cov when available,
                # else fall back to isotropic from config (KITTI path).
                if pgo_cfg.get("gps_cov_isotropic", False):
                    cov = None  # force isotropic fallback (uses gps_sigma_t)
                else:
                    cov = gps_cov_all[g] if gps_cov_all is not None else None

                gps_measurements.append({
                    "chunk_k": k,
                    "p_obs": p_obs,
                    "v_obs": v_obs_raw,
                    "c_loc": c_loc_all[i_local],
                    "v_loc": v_loc_all[i_local],
                    "cov": cov,
                })

        print(f"  [GPS-PGO-Sim3] {len(gps_measurements)} per-frame GPS measurements "
              f"(every {gps_every_n_frames} frame(s), {n_chunks} chunks)")
        if not gps_measurements:
            print("  No GPS measurements generated. Skipping PGO.")
            return
        
        anchor_pts = np.array([m["p_obs"] for m in gps_measurements], dtype=np.float64)
        self._log_gps_anchors_rerun(anchor_pts)
        optimizer = Sim3LoopOptimizer(self.config, device="cpu")
        abs_gps = optimizer.optimize_gps_sim3(
            self.sim3_list,
            gps_measurements,
            umeyama=(s_g, R_g, t_g),
        )

        self._apply_gps_absolutes(abs_gps)

        print("  GPS-PGO complete. Poses updated.\n")

        if poses is not None:
            self._log_gt_rerun_gps_frame(poses)

    def _run_gps_anchor_warp(self, poses_path: str):
        """Stage-2 GPS correction: piecewise-linear trajectory deformation.

        sim3_list is NOT modified. Per-chunk translation corrections are computed
        by linearly interpolating between GPS anchor offsets, then applied directly
        to acc_cam_positions, acc_frame_c2w, and acc_pts.

        Correction at anchor chunk k:  d_k = GPS_k_model - pred_pos_k
          where GPS_k_model = (1/s_g) * R_g.T @ (chunk_gt_pos_k - t_g)

        Between consecutive anchors a and b:
          d_i = lerp(d_a, d_b, (i-a)/(b-a))

        chunk 0 is always fixed (d_0 = 0).
        After the last anchor, correction is held constant.
        """
        print("\n=== GPS Anchor Warp ===")
        poses = load_poses(poses_path)
        if self.frame_stride > 1:
            poses = poses[::self.frame_stride]
        print(f"Loaded {len(poses)} GT poses from {poses_path}"
              + (f" (stride={self.frame_stride})" if self.frame_stride > 1 else ""))

        n_chunks = len(self.chunk_indices)
        pgo_cfg = self.config.get("GPS_PGO", {})
        gps_every_k = pgo_cfg.get("gps_every_k", 5)

        s_g, R_g, t_g, chunk_gt_pos, chunk_gt_rot, pred_pos, pred_scales = \
            self._fit_gt_alignment(poses)
        print(f"  Umeyama: scale={s_g:.4f}  |t_g|={np.linalg.norm(t_g):.3f}m")

        def gps_model(k):
            return (1.0 / s_g) * (R_g.T @ (chunk_gt_pos[k] - t_g))

        # Anchor chunks: chunk 0 fixed at d=0, GPS anchors at every gps_every_k
        anchor_corrections = {0: np.zeros(3, dtype=np.float64)}
        for k in range(1, n_chunks, gps_every_k):
            anchor_corrections[k] = gps_model(k) - pred_pos[k]
        anchors = sorted(anchor_corrections.keys())

        # Build per-chunk corrections by linear interpolation between anchors
        chunk_corrections = np.zeros((n_chunks, 3), dtype=np.float64)
        for i in range(len(anchors) - 1):
            a, b = anchors[i], anchors[i + 1]
            d_a, d_b = anchor_corrections[a], anchor_corrections[b]
            for k in range(a, b + 1):
                alpha = (k - a) / (b - a)
                chunk_corrections[k] = (1.0 - alpha) * d_a + alpha * d_b
        # Hold last anchor correction constant to end of sequence
        for k in range(anchors[-1], n_chunks):
            chunk_corrections[k] = anchor_corrections[anchors[-1]]

        # Apply to acc_cam_positions
        self.acc_cam_positions = [
            cam_pos + chunk_corrections[i].astype(np.float32)
            for i, cam_pos in enumerate(self.acc_cam_positions)
        ]

        # Apply to acc_frame_c2w (update translation column of each [M,4,4] array)
        new_c2w = []
        for i, c2w_chunk in enumerate(self.acc_frame_c2w):
            c2w_new = c2w_chunk.copy()
            c2w_new[:, :3, 3] += chunk_corrections[i].astype(np.float32)
            new_c2w.append(c2w_new)
        self.acc_frame_c2w = new_c2w

        # Apply to acc_pts (each entry tagged with its chunk_idx via _acc_chunk_sim3)
        new_pts, new_sim3 = [], []
        for pts, (ci, s, R, t) in zip(self.acc_pts, self._acc_chunk_sim3):
            corr = chunk_corrections[ci].astype(np.float32)
            new_pts.append(pts + corr)
            new_sim3.append((ci, s, R, (t.astype(np.float64) + chunk_corrections[ci]).astype(np.float32)))
        self.acc_pts = new_pts
        self._acc_chunk_sim3 = new_sim3

        self._log_gt_rerun(poses, s_g, R_g, t_g)
        print("  GPS anchor warp complete.\n")

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        if not img_list:
            raise ValueError(f"No images found in {self.img_dir}")
        n_total = len(img_list)
        if self.frame_stride > 1:
            img_list = img_list[::self.frame_stride]
            print(f"Found {n_total} images; using {len(img_list)} after stride={self.frame_stride}")
        else:
            print(f"Found {n_total} images")

        # Build chunk indices (mirrors Any_Streaming.get_chunk_indices)
        if len(img_list) <= self.chunk_size:
            self.chunk_indices = [(0, len(img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(img_list) - self.overlap + step - 1) // step
            self.chunk_indices = []
            for i in range(num_chunks):
                s_i = i * step
                e_i = min(s_i + self.chunk_size, len(img_list))
                self.chunk_indices.append((s_i, e_i))

        n_chunks = len(self.chunk_indices)
        print(f"Processing {len(img_list)} frames in {n_chunks} chunks "
              f"(size={self.chunk_size}, overlap={self.overlap}, "
              f"step={self.chunk_size - self.overlap})")

        s_abs = 1.0
        R_abs = np.eye(3, dtype=np.float32)
        t_abs = np.zeros(3, dtype=np.float32)

        for chunk_idx, (start, end) in enumerate(self.chunk_indices):
            print(f"\n[Chunk {chunk_idx}/{n_chunks-1}]  frames {start}–{end-1}")
            chunk_paths = img_list[start:end]

            cur_pred = self._infer_chunk(chunk_paths, chunk_idx)

            if chunk_idx > 0:
                s_rel, R_rel, t_rel = self._align_chunks(self.prev_predictions, cur_pred, chunk_idx)
                self.sim3_list.append((s_rel, R_rel, t_rel))
                cumulative = accumulate_sim3_transforms(self.sim3_list)
                s_abs, R_abs, t_abs = cumulative[-1]

            pts, cols, cam_pos, frame_paths = self._extract_new_points(
                cur_pred, chunk_idx, s_abs, R_abs, t_abs, chunk_paths
            )
            if len(pts) > 0:
                self.acc_pts.append(pts)
                self.acc_cols.append(cols)
                self._acc_chunk_sim3.append((chunk_idx, s_abs, R_abs, t_abs))
            self.acc_cam_positions.append(cam_pos)
            self.frame_image_paths.extend(frame_paths)

            # Collect per-frame c2w poses for evaluation
            n_frames = len(cur_pred.depth)
            ov = self.overlap
            if chunk_idx == 0:
                sl = slice(0, n_frames - ov)
                global_indices = list(range(start, start + n_frames - ov))
            else:
                sl = slice(ov, n_frames)
                global_indices = list(range(start + ov, end))
            c2w_local = self._w2c_to_c2w(cur_pred.extrinsics[sl])
            self._acc_frame_c2w_local.append(c2w_local.copy())
            if chunk_idx > 0:
                c2w_world = self._apply_sim3_to_c2w(c2w_local, s_abs, R_abs, t_abs)
            else:
                c2w_world = c2w_local
            self.acc_frame_c2w.append(c2w_world)
            self.acc_frame_global_indices.extend(global_indices)

            # Log GT trajectory up to current chunk end frame in model frame (visualization only)
            if chunk_idx >= 1 and self.poses_path is not None:
                if self._poses_cache is None:
                    self._poses_cache = load_poses(self.poses_path)
                    if self.frame_stride > 1:
                        self._poses_cache = self._poses_cache[::self.frame_stride]
                try:
                    s_g, R_g, t_g, chunk_gt_pos, *_ = self._fit_gt_alignment(self._poses_cache)
                    cur_end = self.chunk_indices[chunk_idx][1]
                    gt_pos = self._poses_cache[:cur_end, :3, 3]
                    gt_model = ((1.0 / s_g) * ((gt_pos - t_g) @ R_g)).astype(np.float32)
                    gt_model_viz = (R_Z_UP @ gt_model.T).T
                    rr.set_time("stable_time", sequence=self._rr_time)
                    rr.log("trajectories/gt", rr.Points3D(
                        positions=gt_model_viz,
                        colors=np.full((len(gt_model_viz), 3), [0, 255, 0], dtype=np.uint8),
                    ))
                    if len(gt_model_viz) >= 2:
                        rr.log("trajectories/gt_line", rr.LineStrips3D(
                            [gt_model_viz], colors=[[0, 255, 0]]
                        ))
                except Exception as e:
                    print(f"  [GPS-vis] Skipped chunk {chunk_idx}: {e}")

            self._log_to_rerun()

            # Per-frame image + depth + confidence, on a dedicated "frame_idx"
            # timeline so each frame is scrubbable independently of the
            # per-chunk "stable_time" timeline used for trajectories/pointcloud.
            # frame_paths/global_indices cover only non-overlap frames; sl
            # slices cur_pred.{depth,conf} to match.
            if frame_paths:
                try:
                    from PIL import Image as PILImage
                    depth_np = np.asarray(cur_pred.depth[sl])   # [K, H, W]
                    conf_np  = np.asarray(cur_pred.conf[sl])    # [K, H, W]
                    K_pred_np = np.asarray(cur_pred.intrinsics[sl])  # [K, 3, 3]
                    K_prior = self._intrinsics_prior  # (3,3) or None
                    for local_i, (fp, gfi) in enumerate(zip(frame_paths, global_indices)):
                        rr.set_time("frame_idx", sequence=int(gfi))
                        img = PILImage.open(fp).convert("RGB")
                        rr.log("camera/image", rr.Image(np.array(img)))
                        rr.log("depth",        rr.DepthImage(depth_np[local_i].astype(np.float32)))
                        rr.log("confidence",   rr.DepthImage(conf_np[local_i].astype(np.float32)))
                        K = K_pred_np[local_i]
                        rr.log("intrinsics/fx/pred", rr.Scalars(float(K[0, 0])))
                        rr.log("intrinsics/fy/pred", rr.Scalars(float(K[1, 1])))
                        rr.log("intrinsics/cx/pred", rr.Scalars(float(K[0, 2])))
                        rr.log("intrinsics/cy/pred", rr.Scalars(float(K[1, 2])))
                        if K_prior is not None:
                            rr.log("intrinsics/fx/prior", rr.Scalars(float(K_prior[0, 0])))
                            rr.log("intrinsics/fy/prior", rr.Scalars(float(K_prior[1, 1])))
                            rr.log("intrinsics/cx/prior", rr.Scalars(float(K_prior[0, 2])))
                            rr.log("intrinsics/cy/prior", rr.Scalars(float(K_prior[1, 2])))

                        # Camera frustum in world frame (same view as map/pointcloud)
                        c2w = c2w_world[local_i]  # (4, 4)
                        h_pin, w_pin = depth_np[local_i].shape[:2]
                        t_c2w = c2w[:3, 3].astype(np.float32)
                        R_c2w = c2w[:3, :3].astype(np.float32)
                        rr.log(
                            "map/camera",
                            rr.Transform3D(
                                translation=R_Z_UP @ t_c2w,
                                mat3x3=R_Z_UP @ R_c2w,
                            ),
                        )
                        rr.log(
                            "map/camera/pinhole",
                            rr.Pinhole(
                                image_from_camera=K.astype(np.float32),
                                height=int(h_pin),
                                width=int(w_pin),
                                camera_xyz=rr.ViewCoordinates.RDF,
                                image_plane_distance=1.0,
                            ),
                        )
                        rr.log("map/camera/pinhole/rgb", rr.Image(np.array(img)))
                except Exception as e:
                    print(f"  [rerun] per-frame log failed: {e}")

            # GPS alignment (optional)
            gps_result = self._compute_gps_alignment()
            if gps_result is not None:
                s_gps, R_gps, t_gps, local_matched, gps_matched = gps_result
                self._log_to_rerun_gps(s_gps, R_gps, t_gps, gps_matched)

            self.prev_predictions = cur_pred

        total_time = time.time() - self.start_time
        with open(self.log_file, "a") as f:
            f.write("\n--- SUMMARY ---\n")
            f.write(f"total_infer_time: {self.total_infer_time:.2f}s\n")
            f.write(f"total_align_time: {self.total_align_time:.2f}s\n")
            f.write(f"total_wall_time: {total_time:.2f}s\n")
            f.write(f"num_images: {len(img_list)}\n")
            f.write(f"num_chunks: {n_chunks}\n")
            f.write(f"avg_infer_per_chunk: {self.total_infer_time / n_chunks:.2f}s\n")
            f.write(f"avg_infer_per_image: {self.total_infer_time / len(img_list):.2f}s\n")

        print(f"\n=== Timing Summary ===")
        print(f"Inference:  {self.total_infer_time:.2f}s")
        print(f"Alignment:  {self.total_align_time:.2f}s")
        print(f"Wall time:  {total_time:.2f}s")
        print(f"Per image:  {self.total_infer_time / len(img_list):.3f}s")

        # Save baseline poses (before PGO)
        baseline_pose_path = os.path.join(self.output_dir, "poses_pred_baseline.txt")
        self._save_poses(baseline_pose_path)

        # ── Pre-compute Umeyama alignment so the baseline trajectory can be
        # logged in GPS frame, matching the post-PGO cyan line and the GT.
        pgo_cfg = self.config.get("GPS_PGO", {})
        _poses = None
        _alignment = None
        if self.poses_path is not None:
            _poses = load_poses(self.poses_path)
            if self.frame_stride > 1:
                _poses = _poses[::self.frame_stride]
            _alignment = self._fit_gt_alignment(_poses)
        elif self.gps_interp is not None and self.sim3_list:
            _alignment = self._fit_gps_csv_alignment()

        # Log baseline trajectory to Rerun (yellow, timestep N) — in GPS frame
        # if we have an alignment, otherwise in model frame as a fallback.
        if self.acc_cam_positions:
            rr.set_time("stable_time", sequence=self._rr_time)
            baseline_traj = np.concatenate(self.acc_cam_positions, axis=0).astype(np.float64)
            if _alignment is not None:
                s_g, R_g, t_g = _alignment[:3]
                baseline_traj = (s_g * (baseline_traj @ R_g.T) + t_g)
            baseline_traj = baseline_traj.astype(np.float32)
            rr.log("trajectories/baseline", rr.Points3D(
                positions=baseline_traj,
                colors=np.full((len(baseline_traj), 3), [255, 220, 0], dtype=np.uint8),
            ))
            if len(baseline_traj) >= 2:
                rr.log("trajectories/baseline_line", rr.LineStrips3D(
                    [baseline_traj], colors=[[255, 220, 0]],
                ))

        # GT trajectory + optional GPS-PGO
        if self.poses_path is not None:
            # GT in GPS frame (raw KITTI poses, no projection).
            self._log_gt_rerun_gps_frame(_poses)

            if pgo_cfg.get("enabled", False):
                method = pgo_cfg.get("method", "anchor_warp")
                if method == "pgo":
                    self._run_gps_pgo(_alignment, poses=_poses)
                else:
                    self._run_gps_anchor_warp(self.poses_path)
                pgo_pose_path = os.path.join(self.output_dir, "poses_pred_pgo.txt")
                self._save_poses(pgo_pose_path)

                # Log PGO trajectory to Rerun (cyan, timestep N+1 → scrub to see correction)
                self._rr_time += 1
                rr.set_time("stable_time", sequence=self._rr_time)
                pgo_traj = np.concatenate(self.acc_cam_positions, axis=0).astype(np.float32)
                rr.log("trajectories/pgo", rr.Points3D(
                    positions=pgo_traj,
                    colors=np.full((len(pgo_traj), 3), [0, 220, 255], dtype=np.uint8),
                ))
                if len(pgo_traj) >= 2:
                    rr.log("trajectories/pgo_line", rr.LineStrips3D(
                        [pgo_traj], colors=[[0, 220, 255]],
                    ))

                # Re-log retransformed pointcloud at PGO timestep
                if self.acc_pts:
                    all_pts  = np.concatenate(self.acc_pts,  axis=0)
                    all_cols = np.concatenate(self.acc_cols, axis=0)
                    max_pts = 3_000_000
                    if len(all_pts) > max_pts:
                        idx = np.random.choice(len(all_pts), size=max_pts, replace=False)
                        all_pts  = all_pts[idx]
                        all_cols = all_cols[idx]
                    rr.log("map/pointcloud", rr.Points3D(positions=all_pts, colors=all_cols))

        elif self.gps_interp is not None:
            # _alignment was already fit above (so we could project the baseline).
            if _alignment is not None:
                self._log_gps_traj_rerun_gps_frame()

            if pgo_cfg.get("enabled", False):
                if pgo_cfg.get("method", "anchor_warp") == "pgo":
                    if _alignment is None:
                        print("  [GPS-CSV] Alignment failed or sim3_list empty. Skipping PGO.")
                    else:
                        self._run_gps_pgo(_alignment)
                        pgo_pose_path = os.path.join(self.output_dir, "poses_pred_pgo.txt")
                        self._save_poses(pgo_pose_path)

                        self._rr_time += 1
                        rr.set_time("stable_time", sequence=self._rr_time)
                        pgo_traj = np.concatenate(self.acc_cam_positions, axis=0).astype(np.float32)
                        rr.log("trajectories/pgo", rr.Points3D(
                            positions=pgo_traj,
                            colors=np.full((len(pgo_traj), 3), [0, 220, 255], dtype=np.uint8),
                        ))
                        if len(pgo_traj) >= 2:
                            rr.log("trajectories/pgo_line", rr.LineStrips3D(
                                [pgo_traj], colors=[[0, 220, 255]],
                            ))
                        if self.acc_pts:
                            all_pts  = np.concatenate(self.acc_pts,  axis=0)
                            all_cols = np.concatenate(self.acc_cols, axis=0)
                            max_pts = 3_000_000
                            if len(all_pts) > max_pts:
                                idx = np.random.choice(len(all_pts), size=max_pts, replace=False)
                                all_pts  = all_pts[idx]
                                all_cols = all_cols[idx]
                            # After PGO, acc_pts lives in GPS frame — log there
                            rr.log("map/global_pointcloud",
                                   rr.Points3D(positions=all_pts, colors=all_cols))
                else:
                    print("  [GPS-CSV] anchor_warp from GPS CSV not yet supported.")

        # Save final poses (post-PGO if run, otherwise same as baseline)
        final_pose_path = os.path.join(self.output_dir, "poses_pred.txt")
        self._save_poses(final_pose_path)

        # When stride>1, also dump the strided GT alongside pred so eval can use a matched file.
        if self.frame_stride > 1 and self.poses_path is not None:
            gt_full = load_poses(self.poses_path)
            gt_strided = gt_full[::self.frame_stride]
            gt_out = os.path.join(self.output_dir, "gt_strided.txt")
            save_poses(gt_strided, gt_out)
            print(f"Saved strided GT ({len(gt_strided)} of {len(gt_full)}) → {gt_out}")

        if self.acc_pts:
            all_pts  = np.concatenate(self.acc_pts,  axis=0)
            all_cols = np.concatenate(self.acc_cols, axis=0)
            ply_path = os.path.join(self.output_dir, "final_map.ply")
            # Points are already filtered — pass ones mask, no further filtering
            save_confident_pointcloud_batch(
                points=all_pts,
                colors=all_cols,
                confs=np.ones(len(all_pts), dtype=np.float32),
                output_path=ply_path,
                conf_threshold=0.0,
                sample_ratio=1.0,
            ) #TODO: Check if worth adding unfiltered map
            print(f"Saved final map → {ply_path}  ({len(all_pts):,} pts)")

        # Save GPS-aligned outputs
        gps_result = self._compute_gps_alignment()
        if gps_result is not None:
            s_gps, R_gps, t_gps, local_matched, gps_matched = gps_result
            global_pts, global_cols = self._apply_gps_transform(s_gps, R_gps, t_gps)

            ply_global = os.path.join(self.output_dir, "final_map_global.ply")
            save_confident_pointcloud_batch(
                points=global_pts, colors=global_cols,
                confs=np.ones(len(global_pts), dtype=np.float32),
                output_path=ply_global, conf_threshold=0.0, sample_ratio=1.0,
            )
            print(f"Saved GPS-aligned map → {ply_global}  ({len(global_pts):,} pts)")

            # Save per-frame poses: predicted (GPS frame) vs GPS ground truth
            poses_path = os.path.join(self.output_dir, "poses_global.txt")
            all_cam = np.concatenate(self.acc_cam_positions, axis=0)
            cam_global = s_gps * (all_cam.astype(np.float64) @ R_gps.T) + t_gps
            with open(poses_path, "w") as f:
                f.write("# image  pred_e pred_n pred_u  gps_e gps_n gps_u\n")
                for i, path in enumerate(self.frame_image_paths):
                    ts = extract_ts_ns(path)
                    if ts is not None and self.gps_interp is not None:
                        e, n, u = self.gps_interp(ts)
                    else:
                        e, n, u = np.nan, np.nan, np.nan
                    f.write(f"{os.path.basename(path)}  "
                            f"{cam_global[i,0]:.4f} {cam_global[i,1]:.4f} {cam_global[i,2]:.4f}  "
                            f"{float(e):.4f} {float(n):.4f} {float(u):.4f}\n")
            print(f"Saved GPS-aligned poses → {poses_path}")

            # Save alignment transform
            transform_path = os.path.join(self.output_dir, "gps_alignment.txt")
            aligned = s_gps * (local_matched @ R_gps.T) + t_gps
            rmse = np.sqrt(np.mean(np.sum((aligned - gps_matched) ** 2, axis=1)))
            with open(transform_path, "w") as f:
                f.write(f"scale: {s_gps}\n")
                f.write(f"rotation:\n{R_gps}\n")
                f.write(f"translation: {t_gps}\n")
                f.write(f"rmse: {rmse:.4f}\n")
                f.write(f"n_matches: {len(local_matched)}\n")
            print(f"Saved GPS alignment info → {transform_path}")

        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DA3/MapAnything Real-time Streaming (RAM) with Rerun")
    parser.add_argument("--image_dir",  type=str, required=True)
    parser.add_argument("--config",     type=str, default="./configs/base_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--gps_csv",    type=str, default=None,
                        help="Path to GPS CSV file for global alignment (optional)")
    parser.add_argument("--poses", type=str, default=None,
                        help="Path to GT poses file for GPS-PGO. Accepts KITTI 12-float [R|t] "
                             "(e.g. poses/07.txt) or TartanAir 7-float xyz+quat (pose_lcam_front.txt).")
    parser.add_argument("--gps_every_k", type=int, default=None,
                        help="Override GPS_PGO.gps_every_k from config (anchor every k-th chunk)")
    parser.add_argument("--gps_anchor_freq", type=int, default=None,
                        help="Override GPS_PGO.gps_anchor_freq from config (one 5-DOF factor every N frames)")
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Subsample input images (and aligned GT) by taking every Nth frame. "
                             "Default 1 = no downsampling. Example: stride=30 on a 30 Hz sequence → 1 Hz.")
    rr.script_add_args(parser)  # adds --rr-addr, --save etc; must be called before parse_args
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gps_every_k is not None:
        config.setdefault("GPS_PGO", {})["gps_every_k"] = args.gps_every_k
    if args.gps_anchor_freq is not None:
        config.setdefault("GPS_PGO", {})["gps_anchor_freq"] = args.gps_anchor_freq

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = os.path.join(
            "./exps", os.path.basename(args.image_dir.rstrip("/")), ts
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Snapshot the config used for this run
    try:
        cfg_dst = os.path.join(args.output_dir, os.path.basename(args.config))
        shutil.copy2(args.config, cfg_dst)
        print(f"Copied config → {cfg_dst}")
    except Exception as e:
        print(f"  [warn] failed to copy config to output dir: {e}")

    # Rerun — mirrors demo_streaming_inference.py lines 441-443
    rr.script_setup(args, "da3_streaming_rt")
    rr.log("map", rr.ViewCoordinates.RDF, static=True)
    rr.set_time("stable_time", sequence=0)

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    streamer = Any_StreamingRT(args.image_dir, args.output_dir, config,
                               gps_csv=args.gps_csv, poses=args.poses,
                               frame_stride=args.frame_stride)
    streamer.run()

    del streamer
    torch.cuda.empty_cache()
    gc.collect()
    sys.exit()