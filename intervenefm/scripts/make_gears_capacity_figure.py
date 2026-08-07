"""GEARS capacity sweep figure: gap vs hidden_size on Norman."""
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

CAP = {32: 1.15, 64: 2.31, 128: 4.69}
rows = []
for h, params in CAP.items():
    df = pd.read_csv(RES / f"audit_gears_capacity_norman_h{h}_seed1.csv")
    piv = df.pivot_table(index='condition', columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean'])
    gap_de = (piv['learned'] - piv['pop_mean'])
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(gap_de.values, size=len(gap_de), replace=True).mean() for _ in range(2000)])
    piv2 = df.pivot_table(index='condition', columns='mode', values='Pearson_delta_full').dropna(subset=['learned', 'pop_mean'])
    gap_pe = (piv2['learned'] - piv2['pop_mean'])
    boot_pe = np.array([rng.choice(gap_pe.values, size=len(gap_pe), replace=True).mean() for _ in range(2000)])
    rows.append({
        "hidden": h, "params_M": params,
        "gap_de_mean": gap_de.mean(),
        "gap_de_lo": np.percentile(boot, 2.5), "gap_de_hi": np.percentile(boot, 97.5),
        "gap_pe_mean": gap_pe.mean(),
        "gap_pe_lo": np.percentile(boot_pe, 2.5), "gap_pe_hi": np.percentile(boot_pe, 97.5),
    })
gdf = pd.DataFrame(rows)
print(gdf)
gdf.to_csv(RES / "gears_capacity_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
N = gdf['params_M'].values
ax.errorbar(N, gdf['gap_de_mean'], yerr=[gdf['gap_de_mean']-gdf['gap_de_lo'], gdf['gap_de_hi']-gdf['gap_de_mean']],
            fmt='o-', capsize=4, color='#1f77b4', label='DE-Spearman gap', markersize=10, lw=1.8)
ax.errorbar(N, gdf['gap_pe_mean'], yerr=[gdf['gap_pe_mean']-gdf['gap_pe_lo'], gdf['gap_pe_hi']-gdf['gap_pe_mean']],
            fmt='s-', capsize=4, color='#d62728', label='Pearson δ gap (full transcriptome)', markersize=10, lw=1.8)
ax.axhline(0, color='black', lw=0.8, linestyle=':')
# Annotate
for i, h in enumerate(gdf['hidden'].values):
    ax.annotate(f'h={h}', xy=(N[i], gdf['gap_de_mean'].iloc[i]),
                xytext=(N[i]*1.05, gdf['gap_de_mean'].iloc[i]+0.005), fontsize=9)
ax.set_xscale('log')
ax.set_xlabel('GEARS parameters (M)')
ax.set_ylabel('gap (learned − pop_mean)')
ax.set_title("GEARS capacity sweep on Norman — non-monotonic, peaks at h=64\n"
             "(1 seed × 5 epochs × 107 test conditions)")
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / 'figure_gears_capacity.png', dpi=200, bbox_inches='tight')
plt.savefig(FIG / 'figure_gears_capacity.pdf', bbox_inches='tight')
print("[fig] saved figure_gears_capacity")
plt.close()
