"""Per-condition / per-gene analysis: WHICH perturbations does the model help on?

Addresses the reviewer's "explain the WHY" critique by characterizing biology
of the gap. For each test condition, compute (learned − pop_mean) and rank.
Identify whether negative-gap conditions cluster (cell-cycle? stress?).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def per_condition_gap(csv_path: Path, condition_col: str, dataset_name: str):
    """Returns DataFrame with per-condition (learned − pop_mean) gap on DE-Spearman."""
    df = pd.read_csv(csv_path)
    if condition_col not in df.columns:
        raise ValueError(f"{condition_col} not in {df.columns.tolist()}")
    piv = df.pivot_table(index=condition_col, columns='mode', values='DE_Spearman')
    if 'pop_mean' not in piv.columns or 'learned' not in piv.columns:
        raise ValueError(f"need pop_mean and learned columns")
    piv['gap'] = piv['learned'] - piv['pop_mean']
    piv = piv.dropna(subset=['gap']).sort_values('gap')
    piv['dataset'] = dataset_name
    return piv


# ---------- Norman (3-seed multiseed default) ----------
norman_path = RES / "audit_multiseed_norman02_all_seeds.csv"
norman = pd.read_csv(norman_path)
norman_pp = norman.pivot_table(index='pair', columns='mode', values='DE_Spearman', aggfunc='mean')
norman_pp['gap'] = norman_pp['learned'] - norman_pp['pop_mean']
norman_pp['dataset'] = 'Norman_minimalCPA_default'
norman_pp = norman_pp.dropna(subset=['gap']).sort_values('gap')
print(f"Norman default: n={len(norman_pp)} pairs")
print(f"  median gap = {norman_pp['gap'].median():+.4f}")
print(f"  fraction gap > 0: {(norman_pp['gap'] > 0).mean():.2f}")
print(f"  worst 5 (most negative): {norman_pp.head(5).index.tolist()}")
print(f"  best 5 (most positive): {norman_pp.tail(5).index.tolist()}")
norman_pp.to_csv(RES / "per_condition_gap_norman.csv")

# ---------- Replogle K562 3-seed ----------
rep_pp = per_condition_gap(RES / "audit_replogle_3seed_all.csv", 'test_gene', 'Replogle_K562_minimalCPA')
rep_pp = rep_pp.groupby(level=0).first()  # average per gene if multi-index
print(f"\nReplogle K562: n={len(rep_pp)} genes")
print(f"  median gap = {rep_pp['gap'].median():+.4f}")
print(f"  fraction gap > 0: {(rep_pp['gap'] > 0).mean():.2f}")
print(f"  worst 5: {rep_pp.head(5).index.tolist()}")
print(f"  best 5: {rep_pp.tail(5).index.tolist()}")
rep_pp.to_csv(RES / "per_condition_gap_replogle_k562.csv")

# ---------- GEARS Norman 4-seed ----------
gears_files = [RES / f"audit_gears_norman_e5_seed{s}.csv" for s in [1, 2, 3, 4]]
gears_dfs = []
for f in gears_files:
    if not f.exists(): continue
    d = pd.read_csv(f); d['seed'] = int(f.stem.split("seed")[-1]); gears_dfs.append(d)
gears_all = pd.concat(gears_dfs, ignore_index=True)
gears_pp = gears_all.pivot_table(index='condition', columns='mode', values='DE_Spearman', aggfunc='mean')
gears_pp['gap'] = gears_pp['learned'] - gears_pp['pop_mean']
gears_pp['dataset'] = 'GEARS_Norman'
gears_pp = gears_pp.dropna(subset=['gap']).sort_values('gap')
print(f"\nGEARS Norman: n={len(gears_pp)} unique conditions")
print(f"  median gap = {gears_pp['gap'].median():+.4f}")
print(f"  fraction gap > 0: {(gears_pp['gap'] > 0).mean():.2f}")
print(f"  worst 5: {gears_pp.head(5).index.tolist()}")
print(f"  best 5: {gears_pp.tail(5).index.tolist()}")
gears_pp.to_csv(RES / "per_condition_gap_gears.csv")

# ---------- Plot: gap distribution per dataset ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, name, df in [
    (axes[0], "Norman 0/2 (minimal CPA, 88 unique pairs × 3 seeds avg)", norman_pp),
    (axes[1], "Replogle K562 (minimal CPA, ~213 unique genes × 3 seeds avg)", rep_pp),
    (axes[2], "GEARS Norman (4 seeds × 233 unique conds avg)", gears_pp),
]:
    gaps = df['gap'].values
    ax.hist(gaps, bins=30, color='#1f77b4', edgecolor='white')
    ax.axvline(0, color='red', linestyle='--', lw=1.5, label='zero')
    ax.axvline(gaps.mean(), color='black', linestyle='-', lw=1.5, label=f'mean={gaps.mean():+.3f}')
    ax.axvline(np.median(gaps), color='gray', linestyle=':', lw=1.5, label=f'median={np.median(gaps):+.3f}')
    ax.set_xlabel('per-condition gap (learned − pop_mean) DE-Spearman')
    ax.set_ylabel('# conditions')
    ax.set_title(name, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
plt.suptitle('Per-condition gap distribution: which perturbations does the embedding help on?', y=1.02)
plt.tight_layout()
plt.savefig(FIG / 'figure_per_condition_gap.png', dpi=200, bbox_inches='tight')
plt.savefig(FIG / 'figure_per_condition_gap.pdf', bbox_inches='tight')
print(f"\n[fig] saved figure_per_condition_gap")
plt.close()
