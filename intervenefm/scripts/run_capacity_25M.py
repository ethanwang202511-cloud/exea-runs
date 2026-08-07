"""Add one larger capacity config (~25 M params) to test the saturation
hypothesis directly (reviewer Issue 2).

The existing capacity sweep on Norman 0/2 ends at xxlarge (12.5 M params)
with gap = +0.162. The xlarge → xxlarge increment is +0.000 (gap is
identical to three decimal places between 6.89 M and 12.51 M), which is
the canonical signature of saturation onset. Adding one larger config
discriminates between:

  (a) Continued power-law growth: gap > +0.16 at 25 M → power law preferred
  (b) Flat saturation: gap ≈ +0.16 at 25 M → saturating-exp preferred
  (c) Non-monotone: gap < +0.16 at 25 M (overfit-on-cell-budget regime)

We use the same Norman 0/2 split, 3 seeds, 20 epochs (matches the rest of
the sweep). Mac CPU takes ~10 min per seed at this size.

Config xxxlarge: pert_dim=384, hidden=2048, z_dim=384 → ~25 M params.
(For comparison: xxlarge is pert_dim=256, hidden=1536, z_dim=384, ~12.5 M.)
"""
import subprocess, sys

CONFIG = {"tag": "cap_xxxlarge", "pert_dim": 384, "hidden": 2048, "z_dim": 384}
SEEDS = [0, 1, 2]

for seed in SEEDS:
    cmd = [
        sys.executable, "scripts/train_and_audit.py",
        "--seed", str(seed), "--epochs", "20", "--max_cells", "60000", "--hvg", "2000",
        "--out_tag", CONFIG["tag"],
        "--pert_dim", str(CONFIG["pert_dim"]),
        "--hidden", str(CONFIG["hidden"]),
        "--z_dim", str(CONFIG["z_dim"]),
    ]
    print(f"\n=== {CONFIG['tag']} seed={seed} ===")
    rc = subprocess.run(cmd).returncode
    print(f"  rc={rc}")
