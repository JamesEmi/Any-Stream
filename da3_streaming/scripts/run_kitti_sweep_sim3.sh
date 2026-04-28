#!/usr/bin/env bash
# KITTI sweep for the Sim3 5-DOF custom GPS factor PGO.
# Runs baseline (no PGO) + PGO at gps_anchor_freq=1,5,10 for each sequence.
#
# Usage: bash scripts/run_kitti_sweep_sim3.sh [seq_ids...]
# Example: bash scripts/run_kitti_sweep_sim3.sh 07          # single seq
#          bash scripts/run_kitti_sweep_sim3.sh             # all seqs 00-10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_ROOT="/mnt/data/anyslam/KITTI_odometry/dataset/sequences"
POSES_ROOT="/mnt/data/anyslam/KITTI_odometry/dataset/poses"
EXP_ROOT="/mnt/data/slam-proj/exps/gtsam/custom5dof/kitti_sweep_sim3_optim"
CONFIG="$DA3_DIR/configs/kitti_rt.yaml"
PY="$DA3_DIR/any_streaming_rt.py"

ANCHOR_FREQS=(1 5 10)

ALL_SEQS=(00 01 02 03 04 05 06 07 08 09 10)
if [ $# -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=("${ALL_SEQS[@]}")
fi

cd "$DA3_DIR"

run_one() {
    local seq="$1"
    local tag="$2"
    shift 2
    local extra_args=("$@")

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

    python "$PY" \
        --image_dir "$IMAGE_ROOT/$seq/image_2" \
        --config "$CONFIG" \
        --output_dir "$out_dir" \
        --save "$rrd_path" \
        "${extra_args[@]}" \
        2>&1 | tee "$out_dir/run.log"

    touch "$done_flag"
    echo "[DONE] seq${seq} ${tag}"
}

eval_one() {
    local seq="$1"
    local tag="$2"
    local pred_file="$3"
    local gt_path="$POSES_ROOT/${seq}.txt"
    local out_dir="$EXP_ROOT/seq${seq}_${tag}"

    if [ ! -f "$gt_path" ]; then
        echo "[SKIP eval] No GT for seq ${seq}"
        return
    fi
    if [ ! -f "$pred_file" ]; then
        echo "[SKIP eval] pred not found: $pred_file"
        return
    fi

    python -m eval.eval_traj \
        --gt "$gt_path" \
        --pred "$pred_file" \
        2>&1 | tee "$out_dir/eval_$(basename "$pred_file" .txt).log"
}

for seq in "${SEQS[@]}"; do
    image_dir="$IMAGE_ROOT/$seq/image_2"
    gt_path="$POSES_ROOT/${seq}.txt"

    if [ ! -d "$image_dir" ]; then
        echo "[SKIP] No image dir for seq $seq"
        continue
    fi

    # 1) Baseline (no PGO)
    run_one "$seq" "baseline"
    eval_one "$seq" "baseline" "$EXP_ROOT/seq${seq}_baseline/poses_pred.txt"

    # 2) Sim3 5-DOF PGO at each anchor freq
    if [ ! -f "$gt_path" ]; then
        echo "[SKIP PGO] No GT poses for seq $seq"
        continue
    fi

    for f in "${ANCHOR_FREQS[@]}"; do
        run_one "$seq" "pgo_anchfreq${f}" \
            --poses "$gt_path" \
            --gps_anchor_freq "$f"
        eval_one "$seq" "pgo_anchfreq${f}" \
            "$EXP_ROOT/seq${seq}_pgo_anchfreq${f}/poses_pred_baseline.txt"
        eval_one "$seq" "pgo_anchfreq${f}" \
            "$EXP_ROOT/seq${seq}_pgo_anchfreq${f}/poses_pred_pgo.txt"
    done
done

echo ""
echo "========================================"
echo "  Sweep complete. Results in $EXP_ROOT"
echo "========================================"
