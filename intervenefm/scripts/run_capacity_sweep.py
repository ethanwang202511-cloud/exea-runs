"""
Capacity sweep: vary model size and check whether the intervention gap is
architecture-invariant or capacity-dependent.

Configurations:
  tiny    : pert_dim=8,  hidden=64,  z_dim=16
  small   : pert_dim=16, hidden=128, z_dim=32
  default : pert_dim=32, hidden=256, z_dim=64    (paper's reference config)
  large   : pert_dim=64, hidden=512, z_dim=128

Each is one seed (=0) for speed. We're testing: does the gap shrink at higher
capacity (suggesting under-fit) or stay flat (architectural property)?
"""
import subprocess, sys
configs = [
    {"tag": "cap_tiny",    "pert_dim": 8,   "hidden": 64,  "z_dim": 16},
    {"tag": "cap_small",   "pert_dim": 16,  "hidden": 128, "z_dim": 32},
    {"tag": "cap_default", "pert_dim": 32,  "hidden": 256, "z_dim": 64},
    {"tag": "cap_large",   "pert_dim": 64,  "hidden": 512, "z_dim": 128},
]

for c in configs:
    cmd = [
        sys.executable, "scripts/train_and_audit.py",
        "--seed", "0", "--epochs", "20", "--max_cells", "60000", "--hvg", "2000",
        "--out_tag", c["tag"],
        "--pert_dim", str(c["pert_dim"]),
        "--hidden", str(c["hidden"]),
        "--z_dim", str(c["z_dim"]),
    ]
    print(f"\n=== {c['tag']} ===  cmd: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    print(f"  rc={rc}")
