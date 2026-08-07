"""Aggregate Replogle 3-seed audit results."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

dfs = []
for seed in [0, 1, 2]:
    f = RES / f"audit_replogle_3seed_seed{seed}.csv"
    if not f.exists():
        print(f"[skip] {f} missing"); continue
    d = pd.read_csv(f)
    d['seed'] = seed
    dfs.append(d)
big = pd.concat(dfs, ignore_index=True)
big.to_csv(RES / "audit_replogle_3seed_all.csv", index=False)

# Per-mode aggregation across genes × seeds
rng = np.random.default_rng(0)
rows = []
for mode in big['mode'].unique():
    sub = big[big['mode'] == mode]
    vals = sub['DE_Spearman'].dropna().values
    pearson_vals = sub['Pearson_delta_full'].dropna().values
    if len(vals) == 0: continue
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)])
    rows.append({
        "mode": mode,
        "n": len(vals),
        "n_seeds": sub['seed'].nunique(),
        "DE_Spearman_mean": float(vals.mean()),
        "DE_Spearman_sem": float(vals.std(ddof=1)/np.sqrt(len(vals))),
        "DE_Spearman_ci_lower": float(np.percentile(boot, 2.5)),
        "DE_Spearman_ci_upper": float(np.percentile(boot, 97.5)),
        "Pearson_full_mean": float(pearson_vals.mean()),
    })
sdf = pd.DataFrame(rows).sort_values("DE_Spearman_mean", ascending=False).round(4)
sdf.to_csv(RES / "audit_replogle_3seed_summary.csv", index=False)
print(sdf.to_string(index=False))

# Gap learned − pop_mean per (gene × seed) — paired
print("\n=== Replogle gap (learned − pop_mean), per (gene × seed) paired bootstrap ===")
piv = big.pivot_table(index=['test_gene', 'seed'], columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean'])
gap = piv['learned'] - piv['pop_mean']
boot = np.array([rng.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
print(f"  n_paired_obs = {len(gap)}, mean = {gap.mean():.4f}, 95% CI [{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
print(f"  median per-pair gap = {gap.median():.4f}; quartiles = ({gap.quantile(0.25):.4f}, {gap.quantile(0.75):.4f})")

# Cluster bootstrap by (test_gene)
genes = sorted(piv.index.get_level_values('test_gene').unique())
gap_per_gene = piv.groupby(level='test_gene').apply(lambda g: (g['learned'] - g['pop_mean']).mean())
print(f"  n_unique_genes = {len(gap_per_gene)}")
boot_g = np.array([rng.choice(gap_per_gene.values, size=len(gap_per_gene), replace=True).mean() for _ in range(2000)])
print(f"  cluster bootstrap by gene: mean={gap_per_gene.mean():.4f}, 95% CI [{np.percentile(boot_g, 2.5):.4f}, {np.percentile(boot_g, 97.5):.4f}]")
