"""Aggregate capacity sweep across seeds into a single table with mean/SEM/CI."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

CONFIGS = [
    ("cap_tiny", "tiny", "8/64/16", 0.16),
    ("cap_small", "small", "16/128/32", 0.42),
    ("cap_default", "default", "32/256/64", 1.20),
    ("cap_large", "large", "64/512/128", 4.36),
    ("cap_xlarge", "xlarge", "128/1024/256", 6.89),
    ("cap_xxlarge", "xxlarge", "256/1536/384", 12.51),
]

# Load per-pair files for each (config, seed) and aggregate.
all_rows = []
for tag, name, dims, params_M in CONFIGS:
    for seed in [0, 1, 2]:
        f = RES / f"audit_{tag}_norman02_seed{seed}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d['config'] = name
        d['params_M'] = params_M
        d['seed'] = seed
        all_rows.append(d)

if not all_rows:
    print("No capacity files found.")
    sys.exit(0)
big = pd.concat(all_rows, ignore_index=True)
big.to_csv(RES / "capacity_sweep_all_pairs.csv", index=False)

# Per (config, mode): mean ± SEM across pairs and seeds, plus 1000-iter bootstrap CI
rng = np.random.default_rng(0)
summary_rows = []
for (config, mode), grp in big.groupby(['config', 'mode']):
    vals = grp['DE_Spearman'].dropna().values
    if len(vals) == 0: continue
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(1000)])
    summary_rows.append({
        "config": config,
        "mode": mode,
        "n": len(vals),
        "n_seeds": grp['seed'].nunique(),
        "mean": float(vals.mean()),
        "sem": float(vals.std(ddof=1) / np.sqrt(len(vals))),
        "ci95_lower": float(np.percentile(boot, 2.5)),
        "ci95_upper": float(np.percentile(boot, 97.5)),
    })
sdf = pd.DataFrame(summary_rows).round(4)
sdf.to_csv(RES / "capacity_sweep_summary_3seed.csv", index=False)

# Pivoted view: mode × config (mean only)
piv = sdf.pivot(index='mode', columns='config', values='mean').reindex(
    ['learned', 'pop_mean', 'random', 'mean', 'zero', 'identity']
)
config_order = ['tiny', 'small', 'default', 'large', 'xlarge', 'xxlarge']
piv = piv[[c for c in config_order if c in piv.columns]]
print("=== Capacity sweep — DE-Spearman ρ mean per (mode, config), across all available seeds ===\n")
print(piv.round(3))
piv.to_csv(RES / "capacity_sweep_summary.csv")

# Gap learned vs pop_mean per config (with CI on the gap)
gap_rows = []
for config in piv.columns:
    learned_vals = big[(big['config'] == config) & (big['mode'] == 'learned')]['DE_Spearman'].dropna().values
    pm_vals = big[(big['config'] == config) & (big['mode'] == 'pop_mean')]['DE_Spearman'].dropna().values
    n = min(len(learned_vals), len(pm_vals))
    if n == 0: continue
    diff = learned_vals[:n] - pm_vals[:n]   # paired by index (same pair, same seed)
    boot = np.array([rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(1000)])
    gap_rows.append({
        "config": config,
        "n_pairs": n,
        "n_seeds": big[big['config'] == config]['seed'].nunique(),
        "gap_mean": float(diff.mean()),
        "gap_ci95_lower": float(np.percentile(boot, 2.5)),
        "gap_ci95_upper": float(np.percentile(boot, 97.5)),
    })
gdf = pd.DataFrame(gap_rows).round(4)
gdf.to_csv(RES / "capacity_sweep_gaps.csv", index=False)
print("\n=== Gap (learned − pop_mean) per config ===")
print(gdf.to_string(index=False))
