"""Re-run capacity sweep with seeds 1 and 2 (seed 0 is already done)."""
import subprocess, sys
configs = [
    {"tag": "cap_tiny",    "pert_dim": 8,   "hidden": 64,  "z_dim": 16},
    {"tag": "cap_small",   "pert_dim": 16,  "hidden": 128, "z_dim": 32},
    {"tag": "cap_default", "pert_dim": 32,  "hidden": 256, "z_dim": 64},
    {"tag": "cap_large",   "pert_dim": 64,  "hidden": 512, "z_dim": 128},
]
for seed in [1, 2]:
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
