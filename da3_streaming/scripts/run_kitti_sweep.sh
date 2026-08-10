#!/usr/bin/env bash
# KITTI sweep: baseline (no PGO) + GPS-PGO at k=1,5,10 for sequences 00-10 (no T0 start prior)
# Usage: bash scripts/run_kitti_sweep.sh [seq_ids...]
# Example: bash scripts/run_kitti_sweep.sh 07        # single seq
#          bash scripts/run_kitti_sweep.sh           # all seqs 00-10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_ROOT="/mnt/data/anyslam/KITTI_odometry/dataset/sequences"
POSES_ROOT="/mnt/data/anyslam/KITTI_odometry/dataset/poses"
EXP_ROOT="/mnt/data/slam-proj/exps/kitti_sweep_gps_no_prior"
CONFIG="$DA3_DIR/configs/kitti_rt.yaml"
PY="$DA3_DIR/any_streaming_rt.py"

K_VALUES=(1 5 10)

# Sequences with GT poses available
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

    python "$PY" \
        --image_dir "$IMAGE_ROOT/$seq/image_2" \
        --config "$CONFIG" \
        --output_dir "$out_dir" \
        --save "$rrd_path" \
        $extra_args \
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

    python -m evaluation.eval_traj \
        --gt "$gt_path" \
        --pred "$pred_file" \
        2>&1 | tee "$out_dir/eval_$(basename $pred_file .txt).log"
}

for seq in "${SEQS[@]}"; do
    image_dir="$IMAGE_ROOT/$seq/image_2"
    gt_path="$POSES_ROOT/${seq}.txt"

    if [ ! -d "$image_dir" ]; then
        echo "[SKIP] No image dir for seq $seq"
        continue
    fi

    # ── 1) Baseline (no PGO) ─────────────────────────────────────────────────
    run_one "$seq" "baseline"
    eval_one "$seq" "baseline" "$EXP_ROOT/seq${seq}_baseline/poses_pred.txt"

    # ── 2) GPS-PGO at each k value ────────────────────────────────────────────
    if [ ! -f "$gt_path" ]; then
        echo "[SKIP PGO] No GT poses for seq $seq"
        continue
    fi

    for k in "${K_VALUES[@]}"; do
        run_one "$seq" "pgo_k${k}" \
            --poses "$gt_path" \
            --gps_every_k "$k"
        eval_one "$seq" "pgo_k${k}" "$EXP_ROOT/seq${seq}_pgo_k${k}/poses_pred_baseline.txt"
        eval_one "$seq" "pgo_k${k}" "$EXP_ROOT/seq${seq}_pgo_k${k}/poses_pred_pgo.txt"
    done
done

echo ""
echo "========================================"
echo "  Sweep complete. Results in $EXP_ROOT"
echo "========================================"
