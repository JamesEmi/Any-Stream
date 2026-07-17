#!/usr/bin/env python3
"""Collect eval logs from a TartanAir sweep and print a summary table.

Auto-detects layout:
  - Flat:   $EXP_ROOT/seqP000_baseline/eval_*.log              (e.g. tartanair_hard_sweep_10fps)
  - Nested: $EXP_ROOT/{easy,hard}/seqP000_baseline/eval_*.log  (e.g. tartanair_sweep_10fps)
"""
import glob
import os
import re
import sys

DEFAULT_EXP_ROOT = "/mnt/data/slam-proj/exps/gtsam/custom5dof/tartanair_hard_sweep_10fps"
EXP_ROOT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXP_ROOT


def parse_log(path):
    text = open(path).read()
    m_ate = re.search(r"ATE \(m\)\s+RMSE:\s+([\d.]+)", text)
    m_rte = re.search(r"RTE \(m\)\s+RMSE:\s+([\d.]+)", text)
    m_roe = re.search(r"ROE \(deg\)\s+RMSE:\s+([\d.]+)", text)
    m_n = re.search(r"(?:Trajectory|KITTI) Evaluation\s+\((\d+) frames\)", text)
    if not (m_ate and m_rte and m_roe):
        return None
    return {
        "ate": float(m_ate.group(1)),
        "rte": float(m_rte.group(1)),
        "roe": float(m_roe.group(1)),
        "n": int(m_n.group(1)) if m_n else -1,
    }


def collect(base_dir, diff_label=""):
    rows = []
    for exp_dir in sorted(glob.glob(os.path.join(base_dir, "seqP*"))):
        dirname = os.path.basename(exp_dir)
        m = re.match(r"seq(P\d+)_(.+)", dirname)
        if not m:
            continue
        seq, tag = m.group(1), m.group(2)
        for log_file in sorted(glob.glob(os.path.join(exp_dir, "eval_*.log"))):
            pred_tag = re.sub(r"eval_poses_pred_?(.*)\.log", r"\1", os.path.basename(log_file))
            pred_tag = pred_tag or "final"
            result = parse_log(log_file)
            if result:
                rows.append((diff_label, seq, tag, pred_tag, result))
    return rows


# Auto-detect layout
rows = []
nested_dirs = [d for d in ("easy", "hard") if os.path.isdir(os.path.join(EXP_ROOT, d))]
if nested_dirs:
    for diff in nested_dirs:
        rows.extend(collect(os.path.join(EXP_ROOT, diff), diff_label=diff))
else:
    rows.extend(collect(EXP_ROOT, diff_label=""))

if not rows:
    print(f"No eval logs found in {EXP_ROOT}")
    sys.exit(0)

show_diff = any(r[0] for r in rows)

# Build table
if show_diff:
    hdr = (f"{'diff':<5}  {'seq':<5}  {'run tag':<14}  {'pred':<10}  {'N':>5}  "
           f"{'ATE RMSE':>10}  {'RTE RMSE':>10}  {'ROE RMSE':>10}")
else:
    hdr = (f"{'seq':<5}  {'run tag':<14}  {'pred':<10}  {'N':>5}  "
           f"{'ATE RMSE':>10}  {'RTE RMSE':>10}  {'ROE RMSE':>10}")
sep = "-" * len(hdr)

lines = [f"# Sweep summary for {EXP_ROOT}", sep, hdr, sep]
for diff, seq, tag, pred_tag, r in rows:
    if show_diff:
        lines.append(f"{diff:<5}  {seq:<5}  {tag:<14}  {pred_tag:<10}  {r['n']:>5}  "
                     f"{r['ate']:>10.4f}  {r['rte']:>10.4f}  {r['roe']:>10.4f}")
    else:
        lines.append(f"{seq:<5}  {tag:<14}  {pred_tag:<10}  {r['n']:>5}  "
                     f"{r['ate']:>10.4f}  {r['rte']:>10.4f}  {r['roe']:>10.4f}")

table = "\n".join(lines)
print(table)

out_path = os.path.join(EXP_ROOT, "summary.txt")
with open(out_path, "w") as f:
    f.write(table + "\n")
print(f"\nSaved → {out_path}")
