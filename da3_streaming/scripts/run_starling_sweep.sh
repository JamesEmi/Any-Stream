#!/usr/bin/env bash
# Run any_streaming_rt over every extracted starling clip under
# /mnt/data/airlab_ws/triage_extract/<date>/<starling>/*_*fps.
# Uses darpa_ma_rt.yaml + GPS-PGO (gps_every_k from config) + lite rrd logging.
# Resumable via per-output `.done` flag. Run from `conda activate anystream`.
#
# Usage:
#   bash scripts/run_starling_sweep.sh                            # default: all dates
#   DATES="mar-18" bash scripts/run_starling_sweep.sh
#   DATES="mar-18" STARLINGS="starling1" bash scripts/run_starling_sweep.sh
#   RR_LOG_EVERY=50 DATES="mar-18" bash scripts/run_starling_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA3_DIR="$(dirname "$SCRIPT_DIR")"

EXTRACT_ROOT="${EXTRACT_ROOT:-/mnt/data/airlab_ws/triage_extract}"
EXP_ROOT="${EXP_ROOT:-/mnt/data/slam-proj/exps/gtsam/darpa}"
CONFIG="${CONFIG:-$DA3_DIR/configs/darpa_ma_rt.yaml}"
PY="$DA3_DIR/any_streaming_rt.py"
RR_LOG_EVERY="${RR_LOG_EVERY:-30}"
TAG="${TAG:-pgo_k1_cs16_ol4_ma}"
DATES="${DATES:-mar-17 mar-18 mar-19}"
STARLINGS="${STARLINGS:-starling1 starling2}"

cd "$DA3_DIR"

for date in $DATES; do
    for starling in $STARLINGS; do
        base="$EXTRACT_ROOT/$date/$starling"
        if [ ! -d "$base" ]; then
            echo "[SKIP] no extract dir: $base"
            continue
        fi
        shopt -s nullglob
        ext_dirs=("$base"/*fps)
        shopt -u nullglob
        if [ ${#ext_dirs[@]} -eq 0 ]; then
            echo "[SKIP] no *_<N>fps dirs in $base"
            continue
        fi
        for ext_dir in "${ext_dirs[@]}"; do
            if [ ! -d "$ext_dir/images" ]; then
                echo "[SKIP] no images: $ext_dir"
                continue
            fi
            stem_full="$(basename "$ext_dir")"        # e.g. 2NC2CT~2_10fps
            stem="${stem_full%_*fps}"                  # strip _<N>fps suffix
            out_dir="$EXP_ROOT/$date/$starling/${stem_full}_${TAG}"
            done_flag="$out_dir/.done"
            if [ -f "$done_flag" ]; then
                echo "[SKIP slam] $date/$starling/$stem_full (already done)"
                continue
            fi
            mkdir -p "$out_dir"

            # Use --gps_csv only if gps.csv exists and has data rows beyond the header.
            gps_args=()
            if [ -f "$ext_dir/gps.csv" ] && [ "$(wc -l < "$ext_dir/gps.csv")" -gt 1 ]; then
                gps_args=(--gps_csv "$ext_dir/gps.csv")
            else
                echo "[INFO] no GPS rows for $date/$starling/$stem_full — running baseline-only"
            fi

            echo ""
            echo "============================================================"
            echo "  SLAM  $date / $starling / $stem_full"
            echo "============================================================"
            PYTHONUNBUFFERED=1 python "$PY" \
                --image_dir "$ext_dir/images" \
                "${gps_args[@]}" \
                --config "$CONFIG" \
                --output_dir "$out_dir" \
                --save "$out_dir/run.rrd" \
                --rr_log_every "$RR_LOG_EVERY" \
                2>&1 | tee "$out_dir/run.log"
            touch "$done_flag"
            echo "[DONE slam] $date/$starling/$stem_full"
        done
    done
done

echo ""
echo "============================================================"
echo "  Sweep complete. Results under $EXP_ROOT"
echo "============================================================"
