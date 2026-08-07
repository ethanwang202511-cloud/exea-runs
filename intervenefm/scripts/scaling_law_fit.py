"""Power-law / scaling-law fit to the capacity sweep.

Per the reviewer feedback (NeurIPS 2026): "Why does the gap saturate at 6.9M?
Use a scaling-law argument."

We fit two scaling forms to the gap (learned − pop_mean) vs parameters:
1. Power law: gap = a * N^b
2. Saturating power: gap = G_inf * (1 - exp(-N/N_*))^c
3. Compare to the LEARNED-only and POP_MEAN-only scaling.

This separates the "global perturbation-response" signal (pop_mean: flat
across capacity by construction) from the "gene-specific residual"
(gap = learned - pop_mean, which scales).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

gaps = pd.read_csv(RES / "capacity_sweep_gaps.csv")
print(gaps)

CAP_PARAMS = {
    "tiny": 0.16e6, "small": 0.42e6, "default": 1.20e6,
    "large": 4.36e6, "xlarge": 6.89e6, "xxlarge": 12.51e6,
}
gaps['params_M'] = gaps['config'].map(lambda c: CAP_PARAMS[c] / 1e6)
gaps = gaps.sort_values('params_M').reset_index(drop=True)

# Drop the tiny config (gap < 0; not a meaningful scaling baseline)
fit_data = gaps[gaps['gap_mean'] > 0].copy()
N = fit_data['params_M'].values
y = fit_data['gap_mean'].values
y_lo = fit_data['gap_ci95_lower'].values
y_hi = fit_data['gap_ci95_upper'].values
y_err = (y_hi - y_lo) / 2.0  # half-width as a CI proxy weight

# === 1. Pure power law: gap = a * N^b ===
def power(N, a, b): return a * np.power(N, b)
try:
    popt_p, pcov_p = curve_fit(power, N, y, sigma=y_err, p0=[0.1, 0.2], maxfev=10000)
    a_p, b_p = popt_p
    print(f"\nPure power law: gap = {a_p:.4f} * N^{b_p:.4f}")
    perr_p = np.sqrt(np.diag(pcov_p))
    print(f"  exponent b = {b_p:.4f} ± {perr_p[1]:.4f}")
    pred_p = power(N, a_p, b_p)
    rss_p = np.sum((y - pred_p) ** 2)
    print(f"  RSS = {rss_p:.6f}")
except Exception as e:
    print(f"Power-law fit failed: {e}")
    popt_p = None

# === 2. Saturating: gap = G_inf * (1 - exp(-N/N_*)) ===
def saturating(N, G_inf, N_star):
    return G_inf * (1 - np.exp(-N / N_star))
try:
    popt_s, pcov_s = curve_fit(saturating, N, y, sigma=y_err, p0=[0.20, 5.0], maxfev=10000)
    G_inf, N_star = popt_s
    perr_s = np.sqrt(np.diag(pcov_s))
    print(f"\nSaturating model: gap = {G_inf:.4f} * (1 - exp(-N/{N_star:.3f}))")
    print(f"  G_inf (asymptote) = {G_inf:.4f} ± {perr_s[0]:.4f}")
    print(f"  N* (half-saturation) = {N_star:.4f} M params ± {perr_s[1]:.4f}")
    pred_s = saturating(N, *popt_s)
    rss_s = np.sum((y - pred_s) ** 2)
    print(f"  RSS = {rss_s:.6f}")
except Exception as e:
    print(f"Saturating fit failed: {e}")
    popt_s = None

# === 3. Compare to LEARNED-only scaling (without subtracting pop_mean) ===
piv = pd.read_csv(RES / "capacity_sweep_summary.csv", index_col=0)
print("\n=== Learned-only scaling vs config ===")
print(piv.loc['learned'])

learned_vals = np.array([piv.loc['learned'][c] for c in ['tiny','small','default','large','xlarge','xxlarge']])
all_N = np.array([0.16, 0.42, 1.20, 4.36, 6.89, 12.51])

# Fit power law to learned ρ as function of N (above tiny baseline)
# learned ρ = ρ_inf - δ * N^(-α) form is common in scaling laws
def neg_power(N, rho_inf, delta, alpha):
    return rho_inf - delta * np.power(N, -alpha)
try:
    popt_l, pcov_l = curve_fit(neg_power, all_N[1:], learned_vals[1:], p0=[0.85, 0.1, 0.3], maxfev=10000)
    rho_inf, delta_l, alpha_l = popt_l
    print(f"\nLearned ρ scaling: ρ_inf - δ*N^(-α) = {rho_inf:.3f} - {delta_l:.3f}*N^(-{alpha_l:.3f})")
except Exception as e:
    print(f"Learned-only fit failed: {e}")

# === 4. Save and plot ===
out_rows = [
    {"model": "power_law", "param_a_or_Ginf": popt_p[0] if popt_p is not None else None,
     "param_b_or_Nstar": popt_p[1] if popt_p is not None else None, "rss": rss_p if popt_p is not None else None},
    {"model": "saturating", "param_a_or_Ginf": popt_s[0] if popt_s is not None else None,
     "param_b_or_Nstar": popt_s[1] if popt_s is not None else None, "rss": rss_s if popt_s is not None else None},
]
pd.DataFrame(out_rows).to_csv(RES / "scaling_law_fits.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.errorbar(N, y, yerr=y_err, fmt='o', color='#1f77b4', capsize=3, label='measured gap',
            zorder=3, markersize=8)
N_plot = np.logspace(np.log10(0.1), np.log10(20), 200)
if popt_p is not None:
    ax.plot(N_plot, power(N_plot, *popt_p), '--', color='#d62728', lw=1.5,
            label=f'power: gap ∝ N^{b_p:.2f}')
if popt_s is not None:
    ax.plot(N_plot, saturating(N_plot, *popt_s), '-', color='#2ca02c', lw=1.5,
            label=f'saturating: G_∞={G_inf:.3f}, N*={N_star:.1f}M')
ax.set_xscale('log')
ax.set_xlabel('parameters (M)')
ax.set_ylabel('gap (learned − pop_mean) DE-Spearman ρ')
ax.set_title('Capacity-sweep gap fits: saturating + power-law\n'
             '(minimal CPA, Norman 0/2, 3 seeds × 6 configs)')
ax.legend(frameon=False, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / 'figure_scaling_law.png', dpi=200, bbox_inches='tight')
plt.savefig(FIG / 'figure_scaling_law.pdf', bbox_inches='tight')
print(f"\n[fig] saved figure_scaling_law")
plt.close()
