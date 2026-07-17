#!/usr/bin/env bash
# TartanAir sweep over BOTH Data_easy and Data_hard:
# baseline (no PGO) + GPS-PGO at k=1,5,10 for P000-P007 in each difficulty.
# "GPS" here is x,y,z extracted from TartanAir GT (pose_lcam_front.txt). The
# pose loader auto-detects KITTI 12-float [R|t] vs TartanAir 7-float xyz+quat,
# so the raw pose file is fed directly to --poses.
#
# Usage:
#   bash scripts/run_tartanair_sweep.sh                       # all difficulties, all seqs
#   bash scripts/run_tartanair_sweep.sh P000 P003             # both difficulties, only listed seqs
#   DIFFICULTIES="easy" bash scripts/run_tartanair_sweep.sh   # easy only
#   DIFFICULTIES="hard" bash scripts/run_tartanair_sweep.sh P002

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="$(dirname "$SCRIPT_DIR")"

DATASET_ROOT="/media/airlab-storage/datasets/TartanAir/tartanair_v2_envs_test"
# Override via env: EXP_ROOT="..." bash scripts/run_tartanair_sweep.sh
EXP_ROOT="${EXP_ROOT:-/mnt/data/slam-proj/exps/gtsam/custom5dof/tartanair_sweep_10fps_v2}"
CONFIG="$DA3_DIR/configs/tartanair_ma_rt.yaml"
PY="$DA3_DIR/any_streaming_rt.py"

K_VALUES=(1)

# Stride 3 on TartanAir (30 fps source) → 10 fps effective.
# Set to 1 to disable subsampling.
FRAME_STRIDE=3

# Override via env: DIFFICULTIES="easy" or DIFFICULTIES="hard"
DIFFICULTIES="${DIFFICULTIES:-easy hard}"

ALL_SEQS=(P000 P001 P002 P003 P004 P005 P006 P007)
if [ $# -gt 0 ]; then
    SEQS=("$@")
else
    SEQS=("${ALL_SEQS[@]}")
fi

cd "$DA3_DIR"

# Materialize a stride-matched GT in KITTI 12-float format for eval.
# At stride=1 this is just a format-normalized copy of the raw pose file;
# at stride>1 it slices to align with the strided pred trajectory.
# Echoes the path of the materialized file. Cached per seq.
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
    local diff="$1"
    local seq="$2"
    local tag="$3"
    local image_dir="$4"
    local extra_args="${@:5}"

    local out_dir="$EXP_ROOT/$diff/seq${seq}_${tag}"
    local rrd_path="$out_dir/${diff}_seq${seq}_${tag}.rrd"
    local done_flag="$out_dir/.done"

    if [ -f "$done_flag" ]; then
        echo "[SKIP] ${diff}/seq${seq} ${tag} already done"
        return
    fi

    echo ""
    echo "========================================"
    echo "  difficulty=${diff}  seq=${seq}  tag=${tag}"
    echo "========================================"

    mkdir -p "$out_dir"

    PYTHONUNBUFFERED=1 python "$PY" \
        --image_dir "$image_dir" \
        --config "$CONFIG" \
        --output_dir "$out_dir" \
        --save "$rrd_path" \
        --frame_stride "$FRAME_STRIDE" \
        $extra_args \
        2>&1 | tee "$out_dir/run.log"

    touch "$done_flag"
    echo "[DONE] ${diff}/seq${seq} ${tag}"
}

eval_one() {
    local diff="$1"
    local seq="$2"
    local tag="$3"
    local pred_file="$4"
    local gt_path="$5"
    local out_dir="$EXP_ROOT/$diff/seq${seq}_${tag}"

    if [ ! -f "$gt_path" ]; then
        echo "[SKIP eval] No GT for ${diff}/seq${seq} ($gt_path)"
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

for diff in $DIFFICULTIES; do
    case "$diff" in
        easy|hard) ;;
        *) echo "[SKIP] unknown difficulty '$diff' (expected 'easy' or 'hard')"; continue ;;
    esac
    diff_root="$DATASET_ROOT/Data_${diff}"
    if [ ! -d "$diff_root" ]; then
        echo "[SKIP] $diff_root does not exist"
        continue
    fi

    for seq in "${SEQS[@]}"; do
        image_dir="$diff_root/$seq/image_lcam_front"
        gt_path="$diff_root/$seq/pose_lcam_front.txt"

        if [ ! -d "$image_dir" ]; then
            echo "[SKIP] No image dir for ${diff}/$seq ($image_dir)"
            continue
        fi
        if [ ! -f "$gt_path" ]; then
            echo "[SKIP] No GT for ${diff}/$seq ($gt_path)"
            continue
        fi

        # Pre-materialize stride-matched GT for eval (cached per seq)
        eval_gt="$(prepare_eval_gt "$gt_path" "$EXP_ROOT/$diff/${seq}_eval_gt.txt")"

        # ── 1) Baseline (no PGO) ─────────────────────────────────────────────
        run_one "$diff" "$seq" "baseline" "$image_dir"
        eval_one "$diff" "$seq" "baseline" \
            "$EXP_ROOT/$diff/seq${seq}_baseline/poses_pred.txt" "$eval_gt"

        # ── 2) GPS-PGO at each k value ───────────────────────────────────────
        for k in "${K_VALUES[@]}"; do
            run_one "$diff" "$seq" "pgo_k${k}" "$image_dir" \
                --poses "$gt_path" \
                --gps_every_k "$k"
            eval_one "$diff" "$seq" "pgo_k${k}" \
                "$EXP_ROOT/$diff/seq${seq}_pgo_k${k}/poses_pred_baseline.txt" "$eval_gt"
            eval_one "$diff" "$seq" "pgo_k${k}" \
                "$EXP_ROOT/$diff/seq${seq}_pgo_k${k}/poses_pred_pgo.txt" "$eval_gt"
        done
    done
done

echo ""
echo "========================================"
echo "  Sweep complete. Results in $EXP_ROOT"
echo "========================================"
