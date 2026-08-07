"""E-distance analysis: control-to-perturbed effect-size on Norman vs Replogle.

Reviewer hypothesis (NeurIPS 2026 critique): 'CRISPRi has smaller perturbation
effects than CRISPRa; the pop_mean baseline is a regularizer that prevents
the model from over-fitting the noise of small shifts.'

We test this directly. For each perturbation in Norman (CRISPRa K562) and
Replogle K562 essential (CRISPRi K562), compute:
- E-distance between perturbed cells and control cells (Peidli et al. 2023)
- Mean Euclidean delta-log expression magnitude per perturbation
- Distribution comparisons

If Replogle perturbations have systematically smaller E-distance than Norman's,
the inversion is consistent with the 'regularizer' hypothesis.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
FIG = ROOT / "figures"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def edistance(X_a: np.ndarray, X_b: np.ndarray, max_n: int = 200) -> float:
    """Energy distance between two samples (Peidli et al. 2023 metric).

    E^2 = 2 * mean(||a - b||) - mean(||a - a'||) - mean(||b - b'||)
    Subsample to max_n cells per group for tractability.
    """
    n_a = min(X_a.shape[0], max_n)
    n_b = min(X_b.shape[0], max_n)
    rng = np.random.default_rng(0)
    A = X_a[rng.choice(X_a.shape[0], n_a, replace=False)]
    B = X_b[rng.choice(X_b.shape[0], n_b, replace=False)]
    # Mean pairwise distances
    delta_ab = np.mean(np.linalg.norm(A[:, None] - B[None, :], axis=-1))
    delta_aa = np.mean(np.linalg.norm(A[:, None] - A[None, :], axis=-1))
    delta_bb = np.mean(np.linalg.norm(B[:, None] - B[None, :], axis=-1))
    e2 = 2 * delta_ab - delta_aa - delta_bb
    return np.sqrt(max(e2, 0))


def analyze_dataset(adata, name: str, n_perts: int = 30, max_cells: int = 200):
    """Analyze E-distance and L2 magnitude per perturbation."""
    print(f"\n=== {name} ===")
    obs = adata.obs
    if 'pert_genes' not in obs.columns:
        from src.data_norman import load_norman
        # we'd need to import the right loader; assume already prepared
        pass
    pg = obs['pert_genes'].values
    n_pert = obs['n_pert'].values
    ctrl_mask = (n_pert == 0)
    X = adata.X
    if hasattr(X, 'toarray'):
        get = lambda rows: np.asarray(X[rows].toarray()).astype(np.float32)
    else:
        get = lambda rows: np.asarray(X[rows]).astype(np.float32)
    ctrl_X = get(np.where(ctrl_mask)[0])
    print(f"  ctrl cells: {ctrl_X.shape[0]}, genes: {ctrl_X.shape[1]}")
    ctrl_mean = ctrl_X.mean(0)

    # Sample n_perts perturbations randomly
    pert_genes_set = set()
    for genes in pg:
        for g in genes:
            pert_genes_set.add(g)
    pert_genes_list = sorted(pert_genes_set)
    rng = np.random.default_rng(0)
    sampled = list(rng.choice(np.array(pert_genes_list, dtype=object),
                              size=min(n_perts, len(pert_genes_list)), replace=False))
    rows = []
    for g in sampled:
        # Select cells where this gene appears in pg
        if name.startswith("Norman"):
            # Use single perturbations only for clean comparison
            mask = np.array([(len(genes) == 1 and genes[0] == g) for genes in pg])
        else:
            mask = np.array([(len(genes) == 1 and genes[0] == g) for genes in pg])
        rows_p = np.where(mask)[0]
        if len(rows_p) < 5: continue
        pert_X = get(rows_p)
        ed = edistance(pert_X, ctrl_X, max_n=max_cells)
        delta_log_l2 = np.linalg.norm(pert_X.mean(0) - ctrl_mean)
        rows.append({
            "dataset": name, "gene": g, "n_cells": len(rows_p),
            "edistance": ed,
            "delta_log_l2": delta_log_l2,
        })
        print(f"  {g}: n={len(rows_p)}, E-dist={ed:.4f}, |Δ|_2={delta_log_l2:.4f}")
    return pd.DataFrame(rows)


def main():
    from src.data_norman import load_norman
    from src.data_replogle import load_replogle
    from src.data_replogle_rpe1 import load_replogle_rpe1

    print("[load_norman] ...")
    nm = load_norman(n_top_hvg=2000, max_cells=60000, seed=0)
    nm_df = analyze_dataset(nm, "Norman_CRISPRa_K562_singles", n_perts=20, max_cells=150)

    print("[load_replogle K562] ...")
    rk = load_replogle(n_top_hvg=2000, max_cells=80000, seed=0)
    rk_df = analyze_dataset(rk, "Replogle_CRISPRi_K562", n_perts=20, max_cells=150)

    print("[load_replogle RPE1] ...")
    try:
        rr = load_replogle_rpe1(n_top_hvg=2000, max_cells=80000, seed=0)
        rr_df = analyze_dataset(rr, "Replogle_CRISPRi_RPE1", n_perts=20, max_cells=150)
        all_df = pd.concat([nm_df, rk_df, rr_df], ignore_index=True)
    except Exception as e:
        print(f"[skip RPE1] {e}")
        all_df = pd.concat([nm_df, rk_df], ignore_index=True)

    all_df.to_csv(RES / "edistance_analysis.csv", index=False)
    print(f"\n=== Per-dataset summary ===")
    summary = all_df.groupby('dataset').agg(
        n=('gene', 'count'),
        edist_mean=('edistance', 'mean'),
        edist_median=('edistance', 'median'),
        delta_l2_mean=('delta_log_l2', 'mean'),
        delta_l2_median=('delta_log_l2', 'median'),
    ).round(4)
    print(summary)
    summary.to_csv(RES / "edistance_summary.csv")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ds_colors = {"Norman_CRISPRa_K562_singles": "#1f77b4",
                 "Replogle_CRISPRi_K562": "#d62728",
                 "Replogle_CRISPRi_RPE1": "#ff7f0e"}
    for ax, col, label in [
        (axes[0], "edistance", "E-distance(perturbed, control) [Peidli 2023]"),
        (axes[1], "delta_log_l2", "‖Δlog mean‖₂ per perturbation"),
    ]:
        for ds, color in ds_colors.items():
            sub = all_df[all_df['dataset'] == ds]
            if len(sub) == 0: continue
            ax.scatter(np.full(len(sub), list(ds_colors).index(ds)) + np.random.normal(0, 0.05, len(sub)),
                       sub[col], alpha=0.6, s=30, color=color, label=ds.replace("_", " "))
        ax.set_xticks(range(len(ds_colors)))
        ax.set_xticklabels([k.replace("_", "\n") for k in ds_colors.keys()], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-perturbation effect-size: CRISPRa vs CRISPRi", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG / "figure_edistance.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIG / "figure_edistance.pdf", bbox_inches='tight')
    print(f"\n[fig] saved figure_edistance")
    plt.close()


if __name__ == "__main__":
    main()
