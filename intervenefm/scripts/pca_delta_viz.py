"""PCA visualization of delta vectors across modes.

Loads delta_vectors_*.npz (saved by save_delta_vectors.py) and produces
a PCA scatter showing where each mode's predicted delta vector lands in
2D PCA space, alongside the observed deltas.

The visual mic-drop (per the reviewer): if 'learned' and 'pop_mean' modes
both project to similar regions of the global perturbation-response axis,
the audit's central claim becomes visually obvious.
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

MODE_COLORS = {
    "learned": "#1f77b4", "pop_mean": "#2ca02c", "mean": "#d62728",
    "random": "#ff7f0e", "zero": "#7f7f7f", "identity": "#bcbd22",
    "observed": "#000000",
}


def load_npz(name: str):
    f = RES / f"delta_vectors_{name}.npz"
    if not f.exists(): return None
    d = np.load(f, allow_pickle=True)
    return {k: d[k] for k in d.files}


def pca_2d(X: np.ndarray) -> tuple:
    """Compute PCA of X (n_samples × n_features), return (PCs, explained_var)."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    PCs = Xc @ Vt[:2].T  # (n, 2)
    var = (S ** 2) / (X.shape[0] - 1)
    return PCs, var[:2] / var.sum()


def plot_pca(name: str, dataset_label: str, ax_de=None, ax_full=None):
    d = load_npz(name)
    if d is None:
        return
    pred_delta = d['pred_delta']  # (n_obs, n_genes)
    obs_delta = d['obs_delta']
    mode = d['mode']
    cond = d['condition']
    n_genes = pred_delta.shape[1]

    # Stack ALL vectors (predicted across modes + observed once per condition)
    # Build a unified set
    learned_vecs = pred_delta[mode == 'learned']
    learned_conds = cond[mode == 'learned']
    pm_vecs = pred_delta[mode == 'pop_mean']
    pm_conds = cond[mode == 'pop_mean']
    obs_unique_idx = np.unique(cond, return_index=True)[1]
    obs_unique_vecs = obs_delta[obs_unique_idx]
    obs_unique_conds = cond[obs_unique_idx]
    print(f"  {name}: {pred_delta.shape[0]} (cond × mode) rows; {len(obs_unique_conds)} unique conditions")

    # Stack and PCA together so coordinates are comparable
    stacked_de = []
    stacked_labels = []  # (mode, cond)
    for m in ('observed', 'learned', 'pop_mean', 'mean', 'random', 'zero'):
        if m == 'observed':
            stacked_de.append(obs_unique_vecs)
            stacked_labels.extend([('observed', c) for c in obs_unique_conds])
        else:
            sub_vecs = pred_delta[mode == m]
            sub_conds = cond[mode == m]
            stacked_de.append(sub_vecs)
            stacked_labels.extend([(m, c) for c in sub_conds])
    stacked_de = np.concatenate(stacked_de, axis=0)
    print(f"    stacked: {stacked_de.shape}")
    PCs, var = pca_2d(stacked_de)
    print(f"    PCA explained var: PC1={var[0]:.3f}, PC2={var[1]:.3f}")

    if ax_de is None:
        fig, ax_de = plt.subplots(figsize=(7, 5))
    for m in ('zero', 'mean', 'random', 'pop_mean', 'learned', 'observed'):
        idx = [i for i, (mm, cc) in enumerate(stacked_labels) if mm == m]
        if not idx: continue
        ax_de.scatter(PCs[idx, 0], PCs[idx, 1],
                      color=MODE_COLORS[m], alpha=0.4 if m != 'observed' else 0.8,
                      s=12 if m != 'observed' else 25,
                      edgecolors='none' if m != 'observed' else 'black',
                      linewidths=0.5,
                      label=m if m != 'observed' else 'observed (target)')
    ax_de.set_xlabel(f"PC1 ({100*var[0]:.1f}% var)")
    ax_de.set_ylabel(f"PC2 ({100*var[1]:.1f}% var)")
    ax_de.set_title(f"{dataset_label}: PCA of delta-log expression vectors\n"
                    f"({len(obs_unique_conds)} unique test conditions × 6 modes)")
    ax_de.legend(loc='best', frameon=False, fontsize=8)
    ax_de.grid(True, alpha=0.3)


# Build a 2-panel figure (RPE1 + K562) and a separate single-panel for each
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
plot_pca("rpe1_seed0", "Replogle RPE1 (CRISPRi, n_test=70)", ax_de=axes[0])
plot_pca("replogle_k562_diag_seed0", "Replogle K562 (CRISPRi, n_test=80)", ax_de=axes[1])
fig.suptitle("PCA of predicted vs observed delta-log vectors across ablation modes (Replogle)\n"
             "If learned ≈ pop_mean ≈ observed in PC space, the embedding contributes nothing on top of the global axis.",
             y=1.03)
plt.tight_layout()
plt.savefig(FIG / "figure_pca_replogle.png", dpi=200, bbox_inches='tight')
plt.savefig(FIG / "figure_pca_replogle.pdf", bbox_inches='tight')
print(f"\n[fig] saved figure_pca_replogle")
plt.close()
