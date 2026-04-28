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
EXP_ROOT="/mnt/data/slam-proj/exps/tartanair_hard_sweep_gps"
CONFIG="$DA3_DIR/configs/tartanair_rt.yaml"
PY="$DA3_DIR/any_streaming_rt.py"

K_VALUES=(1 5 10)

ALL_SEQS=(P000 P001 P002 P003 P004 P005 P006 P007)
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
        --image_dir "$IMAGE_ROOT/$seq/image_lcam_front" \
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

    # ── 1) Baseline (no PGO) ─────────────────────────────────────────────────
    run_one "$seq" "baseline"
    eval_one "$seq" "baseline" "$EXP_ROOT/seq${seq}_baseline/poses_pred.txt" "$gt_path"

    # ── 2) GPS-PGO at each k value ───────────────────────────────────────────
    for k in "${K_VALUES[@]}"; do
        run_one "$seq" "pgo_k${k}" \
            --poses "$gt_path" \
            --gps_every_k "$k"
        eval_one "$seq" "pgo_k${k}" "$EXP_ROOT/seq${seq}_pgo_k${k}/poses_pred_baseline.txt" "$gt_path"
        eval_one "$seq" "pgo_k${k}" "$EXP_ROOT/seq${seq}_pgo_k${k}/poses_pred_pgo.txt" "$gt_path"
    done
done

echo ""
echo "========================================"
echo "  Sweep complete. Results in $EXP_ROOT"
echo "========================================"
