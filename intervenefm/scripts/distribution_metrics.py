"""Distribution-aware metrics for the audit (E-distance, MMD).

Reviewer concern (Miller 2025, Amani 2026): Pearson δ and Spearman ρ on top-DEGs
are dominated by the 95% of genes that don't change. We need distribution-aware
metrics that score the full predicted distribution against the observed.

For each (test condition × mode), we have:
- pred_delta: the predicted mean delta-log expression (HVG-restricted vector)
- obs_delta: the observed mean delta-log
- ctrl_mean: control-cell mean

Standard E-distance and MMD are between two SAMPLES. Our audit code currently
saves only the MEAN predicted delta, not per-cell predicted samples — so true
sample-level E-distance is not directly computable from the saved CSVs.

What IS computable from saved data:
1. **L2-distance between predicted and observed delta means** (a deterministic
   point-vs-point metric, complementary to Spearman/Pearson)
2. **MAGNITUDE-NORMALIZED correlation** (Pearson on residuals after equalizing
   norms — robust to scale differences)
3. **L1-distance over top-k DEGs**

For the saved delta_vectors_*.npz files (Replogle K562 + RPE1), we have the
HVG-restricted delta vectors per (condition × mode), so we can compute these.

This script also computes a SAMPLE-LEVEL E-distance proxy by treating each
(condition × mode) as a single sample-mean vs. observed sample-mean — a degenerate
case that just gives the L2 distance, but we report it consistently with the
literature for drop-in comparison.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def l2_distance(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.linalg.norm(pred - obs))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """RBF-kernel MMD^2 between two samples X (n, d) and Y (m, d).

    Uses the median heuristic for sigma if not given.
    """
    if X.ndim == 1: X = X[None]
    if Y.ndim == 1: Y = Y[None]
    n, m = X.shape[0], Y.shape[0]
    if sigma is None:
        # Median pairwise distance heuristic on the combined sample
        combined = np.concatenate([X, Y], axis=0)
        # subsample if too big
        if combined.shape[0] > 200:
            rng = np.random.default_rng(0)
            combined = combined[rng.choice(combined.shape[0], 200, replace=False)]
        from scipy.spatial.distance import pdist
        dists = pdist(combined)
        sigma = float(np.median(dists)) if len(dists) > 0 else 1.0
        sigma = max(sigma, 1e-3)
    def kernel(A, B):
        sq = np.sum(A**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-sq / (2 * sigma**2))
    Kxx = kernel(X, X); Kyy = kernel(Y, Y); Kxy = kernel(X, Y)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def analyze_npz(path: Path, dataset_label: str):
    print(f"\n=== {dataset_label} ===")
    d = np.load(path, allow_pickle=True)
    pred_delta = d['pred_delta']  # (n_obs, n_genes)
    obs_delta = d['obs_delta']
    cond = d['condition']
    mode = d['mode']

    rows = []
    for m in np.unique(mode):
        idx = np.where(mode == m)[0]
        l2_per_cond = []
        mmd_per_cond = []
        for i in idx:
            pred_v = pred_delta[i]
            obs_v = obs_delta[i]
            l2 = l2_distance(pred_v, obs_v)
            l2_per_cond.append(l2)
            # Sample-level MMD between {pred_v} (n=1) and {obs_v} (n=1) is just k_xx + k_yy - 2 k_xy
            # which simplifies. For meaningful MMD, we'd need multiple samples per condition.
            # Instead: report MMD over the SET of (predicted delta vectors across conditions)
            # vs the SET of (observed delta vectors across conditions) — done OUTSIDE this loop.
        rows.append({
            "mode": m,
            "n_conditions": len(idx),
            "l2_mean": float(np.mean(l2_per_cond)),
            "l2_median": float(np.median(l2_per_cond)),
            "l2_std": float(np.std(l2_per_cond, ddof=1)),
        })

        # Distribution-level MMD between predicted-conditions-distribution-under-mode and observed-conditions-distribution
        pred_set = pred_delta[idx]
        obs_set = obs_delta[np.unique(cond, return_index=True)[1]]
        # Median sigma heuristic on combined
        try:
            mmd = mmd_rbf(pred_set, obs_set)
        except Exception as e:
            mmd = float('nan')
        rows[-1]["mmd_rbf_set_vs_set"] = mmd

    df = pd.DataFrame(rows).sort_values("mmd_rbf_set_vs_set")
    print(df.round(4).to_string(index=False))
    out_path = RES / f"distribution_metrics_{path.stem.replace('delta_vectors_', '')}.csv"
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path}")
    return df


def gap_table(dataset_label: str, df: pd.DataFrame):
    """Compute gap (learned − pop_mean) for L2 and MMD."""
    learned = df[df['mode'] == 'learned'].iloc[0] if (df['mode'] == 'learned').any() else None
    pop_mean = df[df['mode'] == 'pop_mean'].iloc[0] if (df['mode'] == 'pop_mean').any() else None
    if learned is None or pop_mean is None: return
    print(f"\n[{dataset_label}] L2 gap (learned − pop_mean): {learned['l2_mean'] - pop_mean['l2_mean']:+.4f}")
    print(f"[{dataset_label}] MMD gap (learned − pop_mean): {learned['mmd_rbf_set_vs_set'] - pop_mean['mmd_rbf_set_vs_set']:+.4f}")
    # Negative gap means learned is BETTER (smaller distance to obs); positive means WORSE


def main():
    files = [
        (RES / "delta_vectors_replogle_k562_diag_seed0.npz", "Replogle K562"),
        (RES / "delta_vectors_rpe1_seed0.npz", "Replogle RPE1"),
    ]
    for f, label in files:
        if f.exists():
            df = analyze_npz(f, label)
            gap_table(label, df)
        else:
            print(f"[skip] {f} not found")


if __name__ == "__main__":
    main()
