"""GEARS audit figure: bar plot of all 6 modes with bootstrap CIs."""
import sys
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

MODE_ORDER = ["learned", "pop_mean", "mean", "random", "zero", "identity"]
MODE_LABELS = {
    "learned": "learned (GNN-refined e)",
    "mean": "mean ablation",
    "zero": "zero ablation",
    "random": "random gene's e",
    "identity": "identity (basal)",
    "pop_mean": "pop mean (model-free)",
}
MODE_COLORS = {
    "learned": "#1f77b4", "random": "#ff7f0e", "pop_mean": "#2ca02c",
    "mean": "#d62728", "zero": "#7f7f7f", "identity": "#bcbd22",
}

all_files = list(RES.glob("audit_gears_norman_e5_seed[1-4].csv"))
all_dfs = []
for f in sorted(all_files):
    d = pd.read_csv(f)
    d['seed'] = int(f.stem.split("seed")[-1])
    all_dfs.append(d)
df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.read_csv(RES / "audit_gears_norman_e5_seed1.csv")
print(f"[fig] using {len(all_dfs)} seeds, {len(df)} rows")

# Compute per-mode mean + bootstrap CI on DE_Spearman across 107 conditions
rng = np.random.default_rng(0)
rows = []
for mode in MODE_ORDER:
    sub = df[df['mode'] == mode]
    vals = sub['DE_Spearman'].dropna().values
    pearson_vals = sub['Pearson_delta_full'].dropna().values
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(2000)]) if len(vals) > 0 else np.array([0])
    boot_p = np.array([rng.choice(pearson_vals, size=len(pearson_vals), replace=True).mean() for _ in range(2000)])
    rows.append({
        "mode": mode,
        "DE_Spearman": vals.mean() if len(vals) > 0 else float('nan'),
        "DE_lo": np.percentile(boot, 2.5) if len(vals) > 0 else 0,
        "DE_hi": np.percentile(boot, 97.5) if len(vals) > 0 else 0,
        "Pearson": pearson_vals.mean(),
        "P_lo": np.percentile(boot_p, 2.5),
        "P_hi": np.percentile(boot_p, 97.5),
    })
sdf = pd.DataFrame(rows).set_index("mode").loc[MODE_ORDER]
print(sdf)
sdf.to_csv(RES / "audit_gears_norman_summary_with_ci.csv")

# Two-panel figure: DE-Spearman + Pearson δ
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, metric, lo_col, hi_col, ylabel in [
    (axes[0], "DE_Spearman", "DE_lo", "DE_hi",
     "DE-Spearman ρ (top-200 DEGs of held-out condition)"),
    (axes[1], "Pearson", "P_lo", "P_hi",
     "Pearson δ on full transcriptome"),
]:
    means = sdf[metric].fillna(0).values
    lo = means - sdf[lo_col].values
    hi = sdf[hi_col].values - means
    yerr = np.array([lo, hi])
    bars = ax.bar(np.arange(len(sdf)), means, yerr=yerr, capsize=3,
                  edgecolor='black', linewidth=0.8,
                  color=[MODE_COLORS.get(m, "#888") for m in sdf.index])
    ax.set_xticks(np.arange(len(sdf)))
    ax.set_xticklabels([MODE_LABELS[m] for m in sdf.index], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color='black', lw=0.5)
    upper = sdf[hi_col].values
    for i, v in enumerate(means):
        ax.text(i, max(v, upper[i]) + 0.012, f"{v:.2f}", ha='center', fontsize=9)
    if "pop_mean" in sdf.index:
        pm = sdf.loc["pop_mean", metric]
        ax.axhline(pm, color="#2ca02c", linestyle="--", lw=0.8, alpha=0.6)
    n_total = len(df[df['mode'] == 'learned'])
    n_seeds = df['seed'].nunique() if 'seed' in df.columns else 1
    ax.set_title(f"GEARS Norman audit — {metric.replace('_',' ')}\n(n={n_total} = {n_total // n_seeds} test conds × {n_seeds} seeds)")
fig.suptitle("Published-checkpoint extension: GEARS (Roohani et al. 2024) intervention-gap audit on Norman 0/2",
             y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(FIG / "figure_gears_norman.png", dpi=200, bbox_inches='tight')
plt.savefig(FIG / "figure_gears_norman.pdf", bbox_inches='tight')
print(f"[fig] saved figure_gears_norman")
plt.close()
