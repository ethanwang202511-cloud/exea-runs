"""Run training+audit across multiple seeds for robustness."""
import subprocess, sys

SEEDS = [0, 1, 2]
EPOCHS = 20
MAX_CELLS = 60000
HVG = 2000

results = []
for seed in SEEDS:
    cmd = [
        sys.executable, "scripts/train_and_audit.py",
        "--seed", str(seed), "--epochs", str(EPOCHS),
        "--max_cells", str(MAX_CELLS), "--hvg", str(HVG),
        "--out_tag", "multiseed",
    ]
    print(f"\n=== seed={seed} ===")
    res = subprocess.run(cmd, check=False)
    results.append((seed, res.returncode))

print("\n\n=== Multi-seed summary ===")
for seed, rc in results:
    print(f"  seed={seed}  rc={rc}")
