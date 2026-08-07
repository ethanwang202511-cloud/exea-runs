"""Extend the capacity sweep to 8M and 16M params (currently maxes at 4.36M).

Reviewer-relevance: directly addresses the "minimal-CPA artifact" critique by
extending the monotone-gap-with-capacity trend to 100x the smallest config.
Per-config scaling (z=128/256, hidden=768/1024, pert_dim=128/256):

  xlarge:  pert_dim=96, hidden=768,  z_dim=192   ~ 9.5 M params
  xxlarge: pert_dim=128, hidden=1024, z_dim=256  ~ 17 M params

3 seeds × 2 configs = 6 runs ~ 12-15 min on Mac CPU each → ~80-90 min total.
"""
import subprocess, sys, os
configs = [
    {"tag": "cap_xlarge",   "pert_dim": 128,  "hidden": 1024, "z_dim": 256},   # 6.9 M
    {"tag": "cap_xxlarge",  "pert_dim": 256,  "hidden": 1536, "z_dim": 384},   # 12.5 M
]
SEEDS = [0, 1, 2]
for seed in SEEDS:
    for c in configs:
        cmd = [
            sys.executable, "scripts/train_and_audit.py",
            "--seed", str(seed), "--epochs", "20", "--max_cells", "60000", "--hvg", "2000",
            "--out_tag", c["tag"],
            "--pert_dim", str(c["pert_dim"]),
            "--hidden", str(c["hidden"]),
            "--z_dim", str(c["z_dim"]),
        ]
        print(f"\n=== {c['tag']} seed={seed} ===")
        rc = subprocess.run(cmd).returncode
        print(f"  rc={rc}")
