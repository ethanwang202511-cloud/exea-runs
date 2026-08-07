"""Replogle K562 3-seed audit at 40 epochs (extends 1-seed long-train run).

Headlines the Replogle inversion (pop_mean > learned, gap widens with training).
With 3 seeds × 80 test genes = 240 (gene × seed) observations, the negative
gap becomes a robust cross-dataset replication.
"""
import subprocess, sys
SEEDS = [0, 1, 2]
EPOCHS = 40
for seed in SEEDS:
    cmd = [
        sys.executable, "scripts/train_audit_replogle.py",
        "--seed", str(seed), "--epochs", str(EPOCHS),
        "--max_cells", "80000",
        "--out_tag", f"replogle_3seed",
    ]
    print(f"\n=== Replogle 3seed seed={seed} ===")
    rc = subprocess.run(cmd).returncode
    print(f"  rc={rc}")
