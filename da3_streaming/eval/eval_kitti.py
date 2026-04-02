#!/usr/bin/env python3
"""Evaluate predicted poses against KITTI ground truth using evo metrics."""

import argparse
import numpy as np
from eval.kitti_utils import load_kitti_poses, load_kitti_timestamps, poses_to_evo

from evo.core import metrics, sync
from evo.core.metrics import PoseRelation, Unit
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe


def evaluate(gt_path: str, pred_path: str, times_path: str = None, correct_scale: bool = True):
    gt_poses = load_kitti_poses(gt_path)
    pred_poses = load_kitti_poses(pred_path)

    N_gt, N_pred = len(gt_poses), len(pred_poses)
    N = min(N_gt, N_pred)
    if N_gt != N_pred:
        print(f"Warning: GT has {N_gt} poses, pred has {N_pred}. Using first {N}.")
        gt_poses = gt_poses[:N]
        pred_poses = pred_poses[:N]

    if times_path is not None:
        ts = load_kitti_timestamps(times_path)[:N]
    else:
        ts = np.arange(N, dtype=np.float64)

    traj_gt = poses_to_evo(gt_poses, ts)
    traj_pred = poses_to_evo(pred_poses, ts)

    # ATE (with Umeyama alignment)
    ate_result = main_ape.ape(
        traj_gt, traj_pred,
        pose_relation=PoseRelation.translation_part,
        align=True, correct_scale=correct_scale,
    )

    # RTE (relative translation error, delta=1 frame)
    rte_result = main_rpe.rpe(
        traj_gt, traj_pred,
        pose_relation=PoseRelation.translation_part,
        delta=1, delta_unit=Unit.frames,
        align=True, correct_scale=correct_scale,
    )

    # ROE (relative orientation error, delta=1 frame)
    roe_result = main_rpe.rpe(
        traj_gt, traj_pred,
        pose_relation=PoseRelation.rotation_angle_deg,
        delta=1, delta_unit=Unit.frames,
        align=True, correct_scale=correct_scale,
    )

    print("\n" + "=" * 50)
    print(f"  KITTI Evaluation  ({N} frames)")
    print("=" * 50)
    print(f"  ATE (m)   RMSE: {ate_result.stats['rmse']:.4f}  "
          f"mean: {ate_result.stats['mean']:.4f}  "
          f"std: {ate_result.stats['std']:.4f}")
    print(f"  RTE (m)   RMSE: {rte_result.stats['rmse']:.4f}  "
          f"mean: {rte_result.stats['mean']:.4f}  "
          f"std: {rte_result.stats['std']:.4f}")
    print(f"  ROE (deg)  RMSE: {roe_result.stats['rmse']:.4f}  "
          f"mean: {roe_result.stats['mean']:.4f}  "
          f"std: {roe_result.stats['std']:.4f}")
    print("=" * 50)

    return {
        "ate_rmse": ate_result.stats["rmse"],
        "ate_mean": ate_result.stats["mean"],
        "rte_rmse": rte_result.stats["rmse"],
        "rte_mean": rte_result.stats["mean"],
        "roe_rmse": roe_result.stats["rmse"],
        "roe_mean": roe_result.stats["mean"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, required=True, help="Path to KITTI GT poses (e.g. poses/07.txt)")
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted poses (KITTI format)")
    parser.add_argument("--times", type=str, default=None, help="Path to times.txt (optional)")
    parser.add_argument("--no_scale", action="store_true", help="Disable scale correction in alignment")
    args = parser.parse_args()

    evaluate(args.gt, args.pred, args.times, correct_scale=not args.no_scale)
