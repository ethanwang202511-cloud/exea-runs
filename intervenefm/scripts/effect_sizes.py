"""Compute effect sizes (Cohen's d, paired) for all key gaps.

Addresses reviewer critique: bootstrap CIs are nice but effect sizes are
the standard way to communicate "how big is this gap".
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

def cohens_d_paired(x, y):
    """Paired Cohen's d (within-subject). x, y are aligned arrays."""
    diff = x - y
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float('inf')

def cohens_d_independent(x, y):
    """Independent Cohen's d with pooled SD."""
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx-1)*x.std(ddof=1)**2 + (ny-1)*y.std(ddof=1)**2) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled if pooled > 0 else float('inf')

print("=" * 60)
print("EFFECT SIZES (Cohen's d, paired) — learned vs ablation")
print("=" * 60)

# === Norman 3-seed default ===
norman = pd.read_csv(RES / "audit_multiseed_norman02_all_seeds.csv")
print("\n=== Norman 0/2 default config (n=117 = 39 pairs × 3 seeds) ===")
piv = norman.pivot_table(index=['pair', 'seed'], columns='mode', values='DE_Spearman')
piv = piv.dropna(subset=['learned'])
for mode in ['pop_mean', 'random', 'mean', 'zero', 'identity']:
    if mode not in piv.columns: continue
    sub = piv[['learned', mode]].dropna()
    d = cohens_d_paired(sub['learned'].values, sub[mode].values)
    diff = sub['learned'].values - sub[mode].values
    print(f"  learned vs {mode:10s}: Cohen's d = {d:+.3f}, n={len(sub)}, gap={diff.mean():+.4f}, p25={np.percentile(diff,25):+.3f}, p50={np.percentile(diff,50):+.3f}, p75={np.percentile(diff,75):+.3f}")

# Cluster effect size by pair
print("\n  --- Cluster (per-pair averaged) effect sizes ---")
piv_pair = norman.pivot_table(index='pair', columns='mode', values='DE_Spearman', aggfunc='mean')
for mode in ['pop_mean', 'random', 'mean', 'zero', 'identity']:
    if mode not in piv_pair.columns: continue
    sub = piv_pair[['learned', mode]].dropna()
    d = cohens_d_paired(sub['learned'].values, sub[mode].values)
    diff = sub['learned'].values - sub[mode].values
    print(f"  learned vs {mode:10s}: Cohen's d (per-pair) = {d:+.3f}, n_pairs={len(sub)}, gap={diff.mean():+.4f}")

# === Replogle 3-seed ===
print("\n=== Replogle K562 3-seed × 80 test genes × 40 epochs (n=240) ===")
replogle = pd.read_csv(RES / "audit_replogle_3seed_all.csv")
piv = replogle.pivot_table(index=['test_gene', 'seed'], columns='mode', values='DE_Spearman').dropna(subset=['learned'])
for mode in ['pop_mean', 'random', 'mean', 'zero', 'identity']:
    if mode not in piv.columns: continue
    sub = piv[['learned', mode]].dropna()
    d = cohens_d_paired(sub['learned'].values, sub[mode].values)
    diff = sub['learned'].values - sub[mode].values
    print(f"  learned vs {mode:10s}: Cohen's d = {d:+.3f}, n={len(sub)}, gap={diff.mean():+.4f}")

# Pearson δ effect
print("\n  --- Pearson δ on full transcriptome (effect sizes) ---")
piv2 = replogle.pivot_table(index=['test_gene', 'seed'], columns='mode', values='Pearson_delta_full').dropna(subset=['learned'])
for mode in ['pop_mean', 'random', 'mean', 'zero', 'identity']:
    if mode not in piv2.columns: continue
    sub = piv2[['learned', mode]].dropna()
    d = cohens_d_paired(sub['learned'].values, sub[mode].values)
    print(f"  learned vs {mode:10s}: Cohen's d (Pearson δ) = {d:+.3f}, n={len(sub)}")

# === GEARS ===
print("\n=== GEARS Norman, 5 epochs, 1 seed, n=107 test conditions ===")
gears = pd.read_csv(RES / "audit_gears_norman_e5_seed1.csv")
piv = gears.pivot_table(index='condition', columns='mode', values='DE_Spearman').dropna(subset=['learned'])
for mode in ['pop_mean', 'random', 'mean', 'zero']:
    if mode not in piv.columns: continue
    sub = piv[['learned', mode]].dropna()
    d = cohens_d_paired(sub['learned'].values, sub[mode].values)
    diff = sub['learned'].values - sub[mode].values
    print(f"  learned vs {mode:10s}: Cohen's d = {d:+.3f}, n={len(sub)}, gap={diff.mean():+.4f}")

print("\n  --- Pearson δ effect sizes on GEARS ---")
piv2 = gears.pivot_table(index='condition', columns='mode', values='Pearson_delta_full').dropna(subset=['learned'])
for mode in ['pop_mean', 'random', 'mean', 'zero']:
    if mode not in piv2.columns: continue
    sub = piv2[['learned', mode]].dropna()
    d = cohens_d_paired(sub['learned'].values, sub[mode].values)
    diff = sub['learned'].values - sub[mode].values
    print(f"  learned vs {mode:10s}: Cohen's d = {d:+.3f}, n={len(sub)}, gap={diff.mean():+.4f}")
