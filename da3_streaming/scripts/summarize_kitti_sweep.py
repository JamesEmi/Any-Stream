#!/usr/bin/env python3
"""Collect eval logs from kitti_sweep and print a summary table."""
import glob
import re
import os
import sys

EXP_ROOT = "/mnt/data/slam-proj/exps/kitti_sweep"
if len(sys.argv) > 1:
    EXP_ROOT = sys.argv[1]

# Parse one eval log, return dict with ate/rte/roe or None
def parse_log(path):
    text = open(path).read()
    m_ate = re.search(r"ATE \(m\)\s+RMSE:\s+([\d.]+)", text)
    m_rte = re.search(r"RTE \(m\)\s+RMSE:\s+([\d.]+)", text)
    m_roe = re.search(r"ROE \(deg\)\s+RMSE:\s+([\d.]+)", text)
    m_n   = re.search(r"KITTI Evaluation\s+\((\d+) frames\)", text)
    if not (m_ate and m_rte and m_roe):
        return None
    return {
        "ate": float(m_ate.group(1)),
        "rte": float(m_rte.group(1)),
        "roe": float(m_roe.group(1)),
        "n":   int(m_n.group(1)) if m_n else -1,
    }

rows = []
for exp_dir in sorted(glob.glob(os.path.join(EXP_ROOT, "seq*"))):
    dirname = os.path.basename(exp_dir)
    # e.g. seq07_baseline, seq07_pgo_k5
    m = re.match(r"seq(\d+)_(.*)", dirname)
    if not m:
        continue
    seq, tag = m.group(1), m.group(2)

    for log_file in sorted(glob.glob(os.path.join(exp_dir, "eval_*.log"))):
        pred_tag = re.sub(r"eval_poses_pred_?(.*)\.log", r"\1", os.path.basename(log_file))
        pred_tag = pred_tag or "final"
        result = parse_log(log_file)
        if result:
            rows.append((seq, tag, pred_tag, result))

if not rows:
    print(f"No eval logs found in {EXP_ROOT}")
    sys.exit(0)

# Build table lines
hdr = f"{'seq':>4}  {'run tag':<16}  {'pred':<10}  {'N':>5}  {'ATE RMSE':>10}  {'RTE RMSE':>10}  {'ROE RMSE':>10}"
sep = "-" * len(hdr)
lines = [hdr, sep]
for seq, tag, pred_tag, r in rows:
    lines.append(f"{seq:>4}  {tag:<16}  {pred_tag:<10}  {r['n']:>5}  "
                 f"{r['ate']:>10.4f}  {r['rte']:>10.4f}  {r['roe']:>10.4f}")

table = "\n".join(lines)
print(table)

# Save to file
out_path = os.path.join(EXP_ROOT, "summary.txt")
with open(out_path, "w") as f:
    f.write(table + "\n")
print(f"\nSaved → {out_path}")
