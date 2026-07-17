#!/usr/bin/env python3
"""Step-by-step chunk-alignment debug visualizer.

Two modes:
  - default: for each chunk, log it in its own LOCAL frame, then run sim3
    alignment against the previous chunk, log the (s, R, t) and residual,
    apply the cumulative sim3, and log the chunk again in the WORLD frame.
    The accumulated world map and trajectory grow chunk by chunk.
  - --no_align: skip alignment entirely; only log each chunk in its own
    local frame (each chunk effectively starts at the origin).

Rerun timeline:
  Each chunk uses two timesteps (or one in --no_align mode):
    t=2k     : chunk k logged at chunks_local/chunk_{k}/...
    t=2k+1   : alignment metrics + chunk k transformed at chunks_world/chunk_{k}/...
               + cumulative world/pointcloud, world/traj re-logged.

Designed for visual debugging only — no PGO, no GPS, no pose eval.
"""

import argparse
import gc
import glob
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import rerun as rr
import torch

from any_streaming_rt import (
    Any_StreamingRT,
    R_Z_UP,
    depth_to_point_cloud_vectorized,
)
from loop_utils.config_utils import load_config
from loop_utils.sim3utils import accumulate_sim3_transforms, warmup_numba


def alignment_residual(point_map1, conf1, point_map2, conf2,
                       s, R, t, conf_threshold):
    """Mean / median / max distance between chunk1's overlap and the sim3-transformed
    chunk2 overlap, over points that pass the same confidence threshold the aligner used.
    Returns (mean, median, max, n_used)."""
    pm1_flat = point_map1.reshape(-1, 3)
    pm2_flat = point_map2.reshape(-1, 3)
    c1_flat = conf1.reshape(-1)
    c2_flat = conf2.reshape(-1)
    valid = (c1_flat > conf_threshold) & (c2_flat > conf_threshold)
    if not valid.any():
        return float("nan"), float("nan"), float("nan"), 0
    p1 = pm1_flat[valid]
    p2 = pm2_flat[valid]
    p2_aligned = s * (p2 @ R.T) + t
    err = np.linalg.norm(p1 - p2_aligned, axis=1)
    return float(err.mean()), float(np.median(err)), float(err.max()), int(valid.sum())


class Any_StreamingDebug(Any_StreamingRT):
    """Step-through visualization of chunk-by-chunk inference + sim3 alignment."""

    def __init__(self, image_dir, save_dir, config, frame_stride=1, no_align=False):
        super().__init__(image_dir, save_dir, config,
                         gps_csv=None, poses=None, frame_stride=frame_stride)
        self.no_align = no_align

    # ── per-chunk visualization helpers ──────────────────────────────────────

    def _filter_and_subsample(self, predictions, sl):
        """Reuse the parent's filter mask + subsample to keep viz consistent with prod."""
        depth = predictions.depth[sl]
        images = predictions.processed_images[sl]
        if getattr(predictions, 'world_points', None) is not None:
            pts = predictions.world_points[sl]
        else:
            pts = depth_to_point_cloud_vectorized(
                depth, predictions.intrinsics[sl], predictions.extrinsics[sl]
            )
        filter_mask = self._build_filter_mask(predictions, sl)
        pts_v = pts.reshape(-1, 3)[filter_mask.reshape(-1)]
        cols_v = images.reshape(-1, 3).astype(np.uint8)[filter_mask.reshape(-1)]
        sample_ratio = self.config["Model"]["Pointcloud_Save"]["sample_ratio"]
        if len(pts_v) > 0 and sample_ratio < 1.0:
            n_keep = max(1, int(len(pts_v) * sample_ratio))
            idx = np.random.choice(len(pts_v), size=n_keep, replace=False)
            pts_v, cols_v = pts_v[idx], cols_v[idx]
        return pts_v, cols_v

    def _log_chunk_local(self, predictions, chunk_idx, t_seq):
        """Log the WHOLE chunk (all frames) in its own local frame."""
        n = len(predictions.depth)
        pts, cols = self._filter_and_subsample(predictions, slice(0, n))
        cam_pos = self._w2c_to_camera_positions(predictions.extrinsics)

        rr.set_time("stable_time", sequence=t_seq)
        ns = f"chunks_local/chunk_{chunk_idx:04d}"
        if len(pts):
            rr.log(f"{ns}/points", rr.Points3D((R_Z_UP @ pts.T).T, colors=cols))
        traj_viz = (R_Z_UP @ cam_pos.T).T
        col_grey = np.full((len(traj_viz), 3), 200, dtype=np.uint8)
        rr.log(f"{ns}/traj", rr.Points3D(traj_viz, colors=col_grey))
        if len(traj_viz) >= 2:
            rr.log(f"{ns}/traj_line", rr.LineStrips3D([traj_viz], colors=[[200, 200, 200]]))
        print(f"  [local] chunk {chunk_idx}: {len(pts):,} pts, {len(cam_pos)} cams")

    def _log_chunk_world(self, predictions, chunk_idx, s_abs, R_abs, t_abs, t_seq):
        """Log this chunk's NON-OVERLAP portion in the world frame, after cumulative sim3.
        Also append to the growing world pointcloud + trajectory and re-log them.
        """
        pts, cols, cam_pos, _ = self._extract_new_points(
            predictions, chunk_idx, s_abs, R_abs, t_abs, chunk_paths=None
        )

        rr.set_time("stable_time", sequence=t_seq)
        ns = f"chunks_world/chunk_{chunk_idx:04d}"
        if len(pts):
            rr.log(f"{ns}/points", rr.Points3D((R_Z_UP @ pts.T).T, colors=cols))
        traj_viz = (R_Z_UP @ cam_pos.T).T
        col_white = np.full((len(traj_viz), 3), 255, dtype=np.uint8)
        rr.log(f"{ns}/traj", rr.Points3D(traj_viz, colors=col_white))
        if len(traj_viz) >= 2:
            rr.log(f"{ns}/traj_line", rr.LineStrips3D([traj_viz], colors=[[255, 255, 255]]))

        # Accumulate and re-log the world map + trajectory
        if len(pts):
            self.acc_pts.append(pts.astype(np.float32))
            self.acc_cols.append(cols)
        self.acc_cam_positions.append(cam_pos)

        all_pts = np.concatenate(self.acc_pts, axis=0) if self.acc_pts else np.zeros((0, 3))
        all_cols = np.concatenate(self.acc_cols, axis=0) if self.acc_cols else np.zeros((0, 3), np.uint8)
        all_traj = np.concatenate(self.acc_cam_positions, axis=0).astype(np.float32)

        if len(all_pts):
            rr.log("world/pointcloud", rr.Points3D((R_Z_UP @ all_pts.T).T, colors=all_cols))
        all_traj_viz = (R_Z_UP @ all_traj.T).T
        rr.log("world/traj",
               rr.Points3D(all_traj_viz,
                           colors=np.full((len(all_traj_viz), 3), 255, dtype=np.uint8)))
        if len(all_traj_viz) >= 2:
            rr.log("world/traj_line",
                   rr.LineStrips3D([all_traj_viz], colors=[[255, 255, 255]]))
        print(f"  [world] chunk {chunk_idx}: +{len(pts):,} pts → {len(all_pts):,} total")

    def _log_alignment(self, prev_pred, cur_pred, chunk_idx,
                       s_rel, R_rel, t_rel, t_seq):
        """Compute and log alignment params + post-alignment residual."""
        pm1 = depth_to_point_cloud_vectorized(
            prev_pred.depth, prev_pred.intrinsics, prev_pred.extrinsics
        )
        pm2 = depth_to_point_cloud_vectorized(
            cur_pred.depth, cur_pred.intrinsics, cur_pred.extrinsics
        )
        ov = self.overlap
        conf1 = prev_pred.conf[-ov:]
        conf2 = cur_pred.conf[:ov]
        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1
        err_mean, err_med, err_max, n_used = alignment_residual(
            pm1[-ov:], conf1, pm2[:ov], conf2, s_rel, R_rel, t_rel, conf_threshold
        )

        rr.set_time("stable_time", sequence=t_seq)
        rr.log("align/s_rel",            rr.Scalars(float(s_rel)))
        rr.log("align/t_rel_norm",       rr.Scalars(float(np.linalg.norm(t_rel))))
        rr.log("align/residual_mean",    rr.Scalars(err_mean))
        rr.log("align/residual_median",  rr.Scalars(err_med))
        rr.log("align/residual_max",     rr.Scalars(err_max))
        rr.log("align/n_used",           rr.Scalars(float(n_used)))
        rr.log("align/info", rr.TextLog(
            f"chunk {chunk_idx-1}→{chunk_idx}: "
            f"s={s_rel:.4f}  |t|={np.linalg.norm(t_rel):.3f}  "
            f"res mean={err_mean:.4f} med={err_med:.4f} max={err_max:.4f}  n={n_used}"
        ))
        print(f"  [residual] mean={err_mean:.4f} med={err_med:.4f} max={err_max:.4f} "
              f"(n={n_used}, conf_thresh={conf_threshold:.3f})")

    # ── main debug loop ──────────────────────────────────────────────────────

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
            print(f"Found {n_total}; using {len(img_list)} after stride={self.frame_stride}")
        else:
            print(f"Found {n_total} images")

        if len(img_list) <= self.chunk_size:
            self.chunk_indices = [(0, len(img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(img_list) - self.overlap + step - 1) // step
            self.chunk_indices = [
                (i * step, min(i * step + self.chunk_size, len(img_list)))
                for i in range(num_chunks)
            ]
        n_chunks = len(self.chunk_indices)
        print(f"[debug] {len(img_list)} frames → {n_chunks} chunks "
              f"(size={self.chunk_size}, overlap={self.overlap}, "
              f"step={self.chunk_size - self.overlap}, "
              f"mode={'no_align' if self.no_align else 'step_align'})")

        s_abs, R_abs, t_abs = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        t_seq = 0

        for chunk_idx, (start, end) in enumerate(self.chunk_indices):
            print(f"\n[Chunk {chunk_idx}/{n_chunks - 1}]  frames {start}–{end - 1}")
            chunk_paths = img_list[start:end]
            cur_pred = self._infer_chunk(chunk_paths, chunk_idx)

            # 1) chunk in its own local frame — always
            self._log_chunk_local(cur_pred, chunk_idx, t_seq)
            t_seq += 1

            if self.no_align:
                self.prev_predictions = cur_pred
                continue

            # 2) compute alignment + log params/residual (chunk_idx > 0)
            if chunk_idx > 0:
                s_rel, R_rel, t_rel = self._align_chunks(
                    self.prev_predictions, cur_pred, chunk_idx
                )
                self.sim3_list.append((s_rel, R_rel, t_rel))
                cumulative = accumulate_sim3_transforms(self.sim3_list)
                s_abs, R_abs, t_abs = cumulative[-1]
                self._log_alignment(self.prev_predictions, cur_pred, chunk_idx,
                                    s_rel, R_rel, t_rel, t_seq)

            # 3) chunk in world frame after cumulative sim3
            self._log_chunk_world(cur_pred, chunk_idx, s_abs, R_abs, t_abs, t_seq)
            t_seq += 1

            self.prev_predictions = cur_pred

        print("\n[debug] complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-by-step chunk-alignment debugger")
    parser.add_argument("--image_dir",  type=str, required=True)
    parser.add_argument("--config",     type=str, default="./configs/base_config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--frame_stride", type=int, default=1,
                        help="Take every Nth image. Same semantics as any_streaming_rt.py.")
    parser.add_argument("--no_align", action="store_true",
                        help="Skip sim3 alignment; only log each chunk in its own local frame.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.output_dir = os.path.join(
            "./exps_debug", os.path.basename(args.image_dir.rstrip("/")), ts
        )
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        cfg_dst = os.path.join(args.output_dir, os.path.basename(args.config))
        shutil.copy2(args.config, cfg_dst)
        print(f"Copied config → {cfg_dst}")
    except Exception as e:
        print(f"  [warn] failed to copy config: {e}")

    rr.script_setup(args, "anystream_rt_debug")
    rr.log("map", rr.ViewCoordinates.RDF, static=True)
    rr.set_time("stable_time", sequence=0)

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    streamer = Any_StreamingDebug(
        args.image_dir, args.output_dir, config,
        frame_stride=args.frame_stride, no_align=args.no_align,
    )
    streamer.run()

    del streamer
    torch.cuda.empty_cache()
    gc.collect()
    sys.exit()
