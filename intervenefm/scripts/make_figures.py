"""Generate publication figures from results/.
Each figure: figure_*.png + figure_*.pdf in figures/.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Style
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

MODE_ORDER = ["learned", "random", "pop_mean", "mean", "zero", "identity"]
MODE_LABELS = {
    "learned": "learned (actual e)",
    "mean": "mean ablation",
    "zero": "zero ablation",
    "random": "random gene's e",
    "identity": "identity (basal)",
    "pop_mean": "pop mean (model-free)",
}
MODE_COLORS = {
    "learned": "#1f77b4",
    "random": "#ff7f0e",
    "pop_mean": "#2ca02c",
    "mean": "#d62728",
    "zero": "#7f7f7f",
    "identity": "#bcbd22",
}


def fig_intervention_gap(tag: str = "multiseed"):
    """Bar plot: DE-Spearman ρ per ablation mode, with bootstrap CIs."""
    summary_path = RES / f"audit_{tag}_norman02_summary.csv"
    if not summary_path.exists():
        print(f"[fig] missing {summary_path}; run aggregate_seeds.py first.")
        return
    s = pd.read_csv(summary_path)
    s = s.set_index("mode").loc[[m for m in MODE_ORDER if m in s["mode"].tolist() or m in s.index]].reset_index() if False else s.set_index("mode")
    # Sort by MODE_ORDER
    ordered = [m for m in MODE_ORDER if m in s.index]
    s = s.loc[ordered]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(s))
    means = s["mean"].values
    lo = means - s["ci95_lower"].values
    hi = s["ci95_upper"].values - means
    yerr = np.array([lo, hi])
    bars = ax.bar(
        x, means, yerr=yerr, capsize=3, edgecolor='black', linewidth=0.8,
        color=[MODE_COLORS.get(m, "#888") for m in s.index],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in s.index], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("DE-Spearman ρ on top-200 DEGs\n(Norman 0/2 doubles)")
    ax.set_title("Intervention-gap audit: per-ablation DE-Spearman ρ\n"
                 "(minimal CPA, n=39 held-out pairs × 3 seeds)")
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color='black', lw=0.5)
    # Annotate above the error-bar top (not just above bar height)
    upper = s["ci95_upper"].values
    for i, v in enumerate(means):
        ax.text(i, max(v, upper[i]) + 0.025, f"{v:.2f}", ha='center', fontsize=9)
    # Reference line: pop_mean
    if "pop_mean" in s.index:
        pm = s.loc["pop_mean", "mean"]
        ax.axhline(pm, color="#2ca02c", linestyle="--", lw=0.8, alpha=0.6)
        ax.text(len(s) - 0.5, pm + 0.02, f"pop_mean = {pm:.2f}",
                color="#2ca02c", fontsize=8)
    plt.tight_layout()
    out = FIG / f"figure_intervention_gap_{tag}.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.savefig(FIG / f"figure_intervention_gap_{tag}.pdf", bbox_inches='tight')
    print(f"[fig] saved {out}")
    plt.close()


def fig_per_pair_distribution(tag: str = "multiseed"):
    """Strip plot: per-pair DE-Spearman ρ across modes."""
    all_path = RES / f"audit_{tag}_norman02_all_seeds.csv"
    if not all_path.exists():
        print(f"[fig] missing {all_path}; run aggregate_seeds.py first.")
        return
    df = pd.read_csv(all_path)
    df = df[df['mode'].isin(MODE_ORDER)]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    pos = {m: i for i, m in enumerate(MODE_ORDER)}
    rng = np.random.default_rng(0)
    for m in MODE_ORDER:
        sub = df[df['mode'] == m]
        vals = sub['DE_Spearman'].dropna().values
        if len(vals) == 0:
            continue
        x = np.full(len(vals), pos[m]) + rng.normal(0, 0.05, size=len(vals))
        ax.scatter(x, vals, alpha=0.4, s=12, color=MODE_COLORS.get(m, "#888"), edgecolors='none')
        med = np.median(vals)
        ax.plot([pos[m]-0.25, pos[m]+0.25], [med, med], color='black', lw=2)
    ax.set_xticks(list(pos.values()))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in MODE_ORDER], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("DE-Spearman ρ (per pair × seed)")
    ax.set_title("Per-pair distribution of DE-Spearman across ablation modes\n"
                 "(black bars = median)")
    ax.axhline(0, color='black', lw=0.5)
    plt.tight_layout()
    plt.savefig(FIG / f"figure_per_pair_{tag}.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIG / f"figure_per_pair_{tag}.pdf", bbox_inches='tight')
    print(f"[fig] saved figure_per_pair_{tag}")
    plt.close()


def fig_training_curve(tag: str = "multiseed"):
    """Training MSE curves across seeds."""
    files = sorted(RES.glob(f"training_log_{tag}_seed*.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for f in files:
        seed = int(f.stem.split("seed")[-1])
        d = pd.read_csv(f)
        ax.plot(d['epoch'], d['mse'], label=f"seed {seed}", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train MSE")
    ax.set_title("Training MSE — minimal CPA on Norman 2019")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG / f"figure_training_{tag}.png", dpi=200, bbox_inches='tight')
    print(f"[fig] saved figure_training_{tag}")
    plt.close()


def fig_capacity_sweep():
    """Plot DE-Spearman per mode across capacities, with 3-seed CIs."""
    summary_3seed = RES / "capacity_sweep_summary_3seed.csv"
    piv_path = RES / "capacity_sweep_summary.csv"
    if not summary_3seed.exists() or not piv_path.exists():
        return
    long = pd.read_csv(summary_3seed)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    configs = ["tiny", "small", "default", "large", "xlarge", "xxlarge"]
    params = {"tiny": 0.16, "small": 0.42, "default": 1.20, "large": 4.36,
              "xlarge": 6.89, "xxlarge": 12.51}
    x = np.array([params[c] for c in configs])
    for mode in ["learned", "pop_mean", "random", "mean", "zero", "identity"]:
        sub = long[long["mode"] == mode].set_index("config")
        if len(sub) == 0:
            continue
        y_mean = np.array([sub.loc[c, "mean"] for c in configs if c in sub.index])
        y_lo = np.array([sub.loc[c, "ci95_lower"] for c in configs if c in sub.index])
        y_hi = np.array([sub.loc[c, "ci95_upper"] for c in configs if c in sub.index])
        x_used = np.array([params[c] for c in configs if c in sub.index])
        ax.plot(x_used, y_mean, marker='o', lw=1.8,
                label=MODE_LABELS.get(mode, mode), color=MODE_COLORS.get(mode, "#888"))
        ax.fill_between(x_used, y_lo, y_hi, alpha=0.15, color=MODE_COLORS.get(mode, "#888"))
    ax.set_xscale('log')
    ax.set_xlabel("Model parameters (M)")
    ax.set_ylabel("DE-Spearman ρ on top-200 DEGs (Norman 0/2)")
    ax.set_title("Capacity sweep: each ablation's ρ vs. model parameters\n"
                 "(3 seeds × 6 configs, 0.16M–12.5M; shaded = 95% bootstrap CI)")
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "figure_capacity_sweep.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIG / "figure_capacity_sweep.pdf", bbox_inches='tight')
    print(f"[fig] saved figure_capacity_sweep")
    plt.close()


def fig_orthogonal_scaling():
    """Plot DE-Spearman vs alpha along learned and orthogonal directions (multi-direction)."""
    f1 = RES / "scaling_sweep_summary.csv"
    f2 = RES / "orthogonal_multi_summary.csv"
    if not f1.exists() or not f2.exists():
        # fall back to single-direction
        f2 = RES / "orthogonal_scaling_summary.csv"
    if not f1.exists() or not f2.exists():
        return
    s1 = pd.read_csv(f1)
    s2 = pd.read_csv(f2)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(s1['alpha'], s1['DE_Spearman_mean'], yerr=s1['DE_Spearman_sem'],
                marker='o', lw=1.8, capsize=3, color="#1f77b4",
                label=r"along learned direction: $\alpha \cdot e_{\rm learned}$")
    ax.errorbar(s2['alpha_orth'], s2['DE_Spearman_mean'], yerr=s2['DE_Spearman_sem'],
                marker='s', lw=1.8, capsize=3, color="#ff7f0e",
                label=r"random orthogonal direction (8 draws): $\alpha \cdot ||e|| \cdot \hat u_\perp$")
    ax.axhline(0.672, color="#2ca02c", linestyle=':', lw=0.8, alpha=0.7)
    ax.text(2.7, 0.685, "pop_mean ≈ 0.67", color="#2ca02c", fontsize=8)
    ax.set_xlabel(r"$\alpha$  (multiplier)")
    ax.set_ylabel("DE-Spearman ρ on top-200 DEGs")
    ax.set_title("Embedding scaling: learned direction vs random orthogonal direction\n"
                 "(default config; orthogonal n=312 = 8 draws × 39 pairs)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "figure_scaling_sweep.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIG / "figure_scaling_sweep.pdf", bbox_inches='tight')
    print(f"[fig] saved figure_scaling_sweep (with orthogonal overlay, multi-direction)")
    plt.close()


def fig_replogle_3seed():
    """Bar plot of Replogle 3-seed audit modes."""
    f = RES / "audit_replogle_3seed_summary.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    s = s.set_index("mode")
    ordered = [m for m in MODE_ORDER if m in s.index]
    s = s.loc[ordered]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(s))
    means = s["DE_Spearman_mean"].values
    lo = means - s["DE_Spearman_ci_lower"].values
    hi = s["DE_Spearman_ci_upper"].values - means
    yerr = np.array([lo, hi])
    ax.bar(x, means, yerr=yerr, capsize=3, edgecolor='black', linewidth=0.8,
           color=[MODE_COLORS.get(m, "#888") for m in s.index])
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in s.index], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("DE-Spearman ρ on top-200 DEGs\n(Replogle K562 essential 0/1)")
    ax.set_title("Replogle K562 audit (3 seeds × 80 test genes × 40 epochs, n=240)\n"
                 "model-free pop_mean BEATS the unablated learned model")
    ax.set_ylim(-0.05, 0.6)
    ax.axhline(0, color='black', lw=0.5)
    upper = s["DE_Spearman_ci_upper"].values
    for i, v in enumerate(means):
        ax.text(i, max(v, upper[i]) + 0.012, f"{v:.2f}", ha='center', fontsize=9)
    if "pop_mean" in s.index:
        pm = s.loc["pop_mean", "DE_Spearman_mean"]
        ax.axhline(pm, color="#2ca02c", linestyle="--", lw=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIG / "figure_replogle_3seed.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIG / "figure_replogle_3seed.pdf", bbox_inches='tight')
    print(f"[fig] saved figure_replogle_3seed")
    plt.close()


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "multiseed"
    fig_intervention_gap(tag)
    fig_per_pair_distribution(tag)
    fig_training_curve(tag)
    fig_capacity_sweep()
    fig_orthogonal_scaling()
    fig_replogle_3seed()
