#!/usr/bin/env bash
# TartanAir-hard sweep: baseline (no PGO) + GPS-PGO at k=1,5,10 for P000-P007.
# "GPS" here is x,y,z extracted from TartanAir GT (pose_lcam_front.txt). The
# pose loader auto-detects KITTI 12-float [R|t] vs TartanAir 7-float xyz+quat,
# so the raw pose file is fed directly to --poses.
# Usage: bash scripts/run_tartanair_hard_sweep.sh [seq_ids...]
# Example: bash scripts/run_tartanair_hard_sweep.sh P000        # single seq
#          bash scripts/run_tartanair_hard_sweep.sh             # all seqs P000-P007

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_ROOT="/media/airlab-storage/datasets/TartanAir/tartanair_v2_envs_test/Data_hard"
EXP_ROOT="/mnt/data/slam-proj/exps/gtsam/custom5dof/tartanair_hard_sweep_10fps_v2"
CONFIG="$DA3_DIR/configs/tartanair_ma_rt.yaml"
PY="$DA3_DIR/any_streaming_rt.py"

K_VALUES=(1)

# Stride 3 on TartanAir (30 fps source) → 10 fps effective.
# Set to 1 to disable subsampling.
FRAME_STRIDE=3

ALL_SEQS=(P000 P001 P002 P003 P004 P005 P006 P007)
if [ $# -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=("${ALL_SEQS[@]}")
fi

cd "$DA3_DIR"

# Materialize a stride-matched GT in KITTI 12-float format for eval.
# At stride=1 this is a format-normalized copy of the raw pose file;
# at stride>1 it slices to align with the strided pred trajectory.
# Echoes the path. Cached per seq.
prepare_eval_gt() {
    local gt_src="$1"
    local gt_dst="$2"
    if [ ! -f "$gt_dst" ]; then
        mkdir -p "$(dirname "$gt_dst")"
        python -c "
import sys; sys.path.insert(0, '.')
from eval.pose_utils import load_poses, save_poses
gt = load_poses('$gt_src')
if $FRAME_STRIDE > 1:
    gt = gt[::$FRAME_STRIDE]
save_poses(gt, '$gt_dst')
print(f'[prepare_eval_gt] {len(gt)} poses -> $gt_dst')
" 1>&2
    fi
    echo "$gt_dst"
}

run_one() {
    local seq="$1"
    local tag="$2"
    local extra_args="${@:3}"

    local out_dir="$EXP_ROOT/seq${seq}_${tag}"
    local rrd_path="$out_dir/seq${seq}_${tag}.rrd"
    local done_flag="$out_dir/.done"

    if [ -f "$done_flag" ]; then
        echo "[SKIP] seq${seq} ${tag} already done"
        return
    fi

    echo ""
    echo "========================================"
    echo "  seq=${seq}  tag=${tag}"
    echo "========================================"

    mkdir -p "$out_dir"

    PYTHONUNBUFFERED=1 python "$PY" \
        --image_dir "$IMAGE_ROOT/$seq/image_lcam_front" \
        --config "$CONFIG" \
        --output_dir "$out_dir" \
        --save "$rrd_path" \
        --frame_stride "$FRAME_STRIDE" \
        $extra_args \
        2>&1 | tee "$out_dir/run.log"

    touch "$done_flag"
    echo "[DONE] seq${seq} ${tag}"
}

eval_one() {
    local seq="$1"
    local tag="$2"
    local pred_file="$3"
    local gt_path="$4"
    local out_dir="$EXP_ROOT/seq${seq}_${tag}"

    if [ ! -f "$gt_path" ]; then
        echo "[SKIP eval] No GT for seq ${seq} ($gt_path)"
        return
    fi
    if [ ! -f "$pred_file" ]; then
        echo "[SKIP eval] pred not found: $pred_file"
        return
    fi

    python -m eval.eval_traj \
        --gt "$gt_path" \
        --pred "$pred_file" \
        2>&1 | tee "$out_dir/eval_$(basename $pred_file .txt).log"
}

for seq in "${SEQS[@]}"; do
    image_dir="$IMAGE_ROOT/$seq/image_lcam_front"
    gt_path="$IMAGE_ROOT/$seq/pose_lcam_front.txt"

    if [ ! -d "$image_dir" ]; then
        echo "[SKIP] No image dir for seq $seq ($image_dir)"
        continue
    fi
    if [ ! -f "$gt_path" ]; then
        echo "[SKIP] No GT for seq $seq ($gt_path)"
        continue
    fi

    # Pre-materialize stride-matched GT for eval (cached per seq)
    eval_gt="$(prepare_eval_gt "$gt_path" "$EXP_ROOT/${seq}_eval_gt.txt")"

    # ── 1) Baseline (no PGO) ─────────────────────────────────────────────────
    run_one "$seq" "baseline"
    eval_one "$seq" "baseline" "$EXP_ROOT/seq${seq}_baseline/poses_pred.txt" "$eval_gt"

    # ── 2) GPS-PGO at each k value ───────────────────────────────────────────
    for k in "${K_VALUES[@]}"; do
        run_one "$seq" "pgo_k${k}" \
            --poses "$gt_path" \
            --gps_every_k "$k"
        eval_one "$seq" "pgo_k${k}" "$EXP_ROOT/seq${seq}_pgo_k${k}/poses_pred_baseline.txt" "$eval_gt"
        eval_one "$seq" "pgo_k${k}" "$EXP_ROOT/seq${seq}_pgo_k${k}/poses_pred_pgo.txt" "$eval_gt"
    done
done

echo ""
echo "========================================"
echo "  Sweep complete. Results in $EXP_ROOT"
echo "========================================"
