"""Non-DL baselines (Issue 9 in reviewer feedback) on Replogle K562 0/1.

Reviewer ask: "If the deep model is statistically indistinguishable from a
4-line model-free pop_mean predictor, the paper begs for a battery of non-DL
baselines: ridge regression (Ahlmann-Eltze's baseline), k-NN in expression
space, GSEA-weighted means."

We add three external baselines that each consume a per-gene identity
feature (SVD-of-controls, 64-dim, same vector CPG uses):

1. **Ridge-on-gene-identity**: per HVG gene k, fit a Ridge regression
   mapping `gene_identity[pert_gene]` → mean perturbed delta on gene k
   across cells of that perturbation. At inference, given a held-out
   perturbation g_test and its gene_identity vector, predict the full
   delta vector by stacking the per-gene-k regressions. This is the
   simplest baseline that genuinely uses the gene-identity feature
   without any deep learning.

2. **k-NN by gene-identity (k=10)**: for each held-out test pert g_test,
   find the 10 train perts with highest cosine similarity to g_test in
   gene-identity space; predict delta_test as the mean of those 10
   train perts' observed mean deltas.

3. **pop_mean**: the model-free mean-of-training-perts baseline. Already
   in the audit; reported here for direct comparison.

For each test gene, evaluate per-gene mean predicted delta against the
ground-truth observed mean delta. Metrics: DE-Spearman ρ on top-200
DEGs, Pearson δ on full transcriptome.

Output:
  - results/nondl_baselines_replogle_k562.csv (per test gene × baseline)
  - results/nondl_baselines_summary.csv (aggregated)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_replogle import build_split_replogle, build_gene_vocab_replogle
from src.cpa_cpg import compute_gene_identity_table


def load_replogle_no_hvg(path, max_cells=80000, seed=0):
    a = sc.read_h5ad(path)
    a.obs['nperts'] = a.obs['nperts'].astype(int) if 'nperts' in a.obs.columns else 0
    pert_genes = []
    for p in a.obs['perturbation']:
        s = str(p).strip()
        if s == 'control' or 'NegCtrl' in s or s == 'unassigned':
            pert_genes.append([])
        else:
            base = s.split('.')[0].split(';')[0]
            pert_genes.append([base])
    a.obs['pert_genes'] = np.array(pert_genes, dtype=object)
    a.obs['n_pert'] = [len(x) for x in a.obs['pert_genes']]
    if max_cells is not None and a.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(a.n_obs, size=max_cells, replace=False)
        a = a[np.sort(keep)].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def restrict_to_hvg(a_full, n_top=2000):
    ctrl_mask = (a_full.obs['n_pert'] == 0).values
    ctrl_only = a_full[ctrl_mask].copy()
    sc.pp.highly_variable_genes(ctrl_only, n_top_genes=n_top, subset=False, flavor='seurat')
    return a_full[:, ctrl_only.var['highly_variable'].values].copy()


def get_top_deg_indices_obs(obs_delta, k=200):
    return np.argsort(-np.abs(obs_delta))[:k]


def evaluate_pred(pred_delta, obs_delta, k=200):
    top = get_top_deg_indices_obs(obs_delta, k=k)
    if len(top) > 1:
        de_spear, _ = spearmanr(pred_delta[top], obs_delta[top])
        de_spear = float(de_spear) if np.isfinite(de_spear) else 0.0
    else:
        de_spear = 0.0
    if pred_delta.std() > 1e-10 and obs_delta.std() > 1e-10:
        p_full = float(np.corrcoef(pred_delta, obs_delta)[0, 1])
    else:
        p_full = 0.0
    return de_spear, p_full


def main():
    print("# Non-DL baselines on Replogle K562 essential (Issue 9)")
    K562_PATH = ROOT / "data" / "replogle_k562_essential.h5ad"
    a_full = load_replogle_no_hvg(K562_PATH, max_cells=80000, seed=0)
    print(f"[load] full-panel {a_full.shape}")

    a = restrict_to_hvg(a_full, n_top=2000)
    print(f"[hvg] {a.shape}")
    vocab = build_gene_vocab_replogle(a)
    split = build_split_replogle(a, test_frac_genes=0.2, seed=0)

    # Cap test genes to 80 (matches CPG audit)
    if len(split['test_genes']) > 80:
        rng = np.random.default_rng(0)
        split['test_genes'] = list(rng.choice(np.array(split['test_genes'], dtype=object),
                                              size=80, replace=False))
    print(f"[split] train_cells={len(split['train_cells'])}, test_cells={len(split['test_cells'])}, test_genes={len(split['test_genes'])}")

    # Compute gene-identity table from FULL panel
    gene_id_table = compute_gene_identity_table(
        a_full, vocab, split['ctrl_cells'], d_id=64, seed=0,
    )
    print(f"[gene_id] shape={gene_id_table.shape}")
    del a_full

    # Compute observed deltas per training pert and per test pert
    cell_to_row = {c: i for i, c in enumerate(a.obs.index)}
    X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)
    n_pert = a.obs['n_pert'].values
    pg = a.obs['pert_genes'].values
    ctrl_rows = np.array([cell_to_row[c] for c in split['ctrl_cells']])
    ctrl_mean = X[ctrl_rows].mean(axis=0)

    # Per-train-pert mean delta
    train_pert_to_cells = {}
    for ci in (cell_to_row[c] for c in split['train_cells']):
        if n_pert[ci] == 1:
            g = pg[ci][0]
            if g in vocab:
                train_pert_to_cells.setdefault(g, []).append(ci)
    train_pert_genes = [g for g, cells in train_pert_to_cells.items() if len(cells) >= 5]
    train_pert_deltas = np.stack([X[train_pert_to_cells[g]].mean(axis=0) - ctrl_mean for g in train_pert_genes])
    print(f"[train deltas] {train_pert_deltas.shape} ({len(train_pert_genes)} train perts with >=5 cells)")

    # Per-test-pert observed mean delta
    test_pert_to_cells = {}
    for ci in (cell_to_row[c] for c in split['test_cells']):
        if n_pert[ci] == 1:
            g = pg[ci][0]
            if g in split['test_genes']:
                test_pert_to_cells.setdefault(g, []).append(ci)
    test_genes_with_data = [g for g in split['test_genes'] if g in test_pert_to_cells and len(test_pert_to_cells[g]) >= 5]
    test_pert_deltas = np.stack([X[test_pert_to_cells[g]].mean(axis=0) - ctrl_mean for g in test_genes_with_data])
    print(f"[test deltas] {test_pert_deltas.shape} ({len(test_genes_with_data)} test perts with >=5 cells)")

    # === pop_mean baseline ===
    train_pert_mean_delta = train_pert_deltas.mean(axis=0)

    # === Ridge baseline ===
    # X = gene_identity vectors of training perts; y = per-gene mean delta
    X_train_id = np.stack([gene_id_table[vocab[g]].numpy() for g in train_pert_genes])
    y_train = train_pert_deltas  # (n_train_perts, n_genes)
    print(f"[ridge] X_train_id={X_train_id.shape}, y_train={y_train.shape}")
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_id, y_train)
    print(f"[ridge] fit in {time.time()-t0:.1f}s")
    X_test_id = np.stack([gene_id_table[vocab[g]].numpy() for g in test_genes_with_data])
    ridge_preds = ridge.predict(X_test_id)  # (n_test_perts, n_genes)

    # === k-NN baseline (k=10) ===
    sims = cosine_similarity(X_test_id, X_train_id)  # (n_test, n_train)
    k = 10
    knn_preds = np.zeros_like(test_pert_deltas)
    for i in range(len(test_genes_with_data)):
        nn_idx = np.argsort(-sims[i])[:k]
        knn_preds[i] = train_pert_deltas[nn_idx].mean(axis=0)

    # === Evaluate each baseline ===
    rows = []
    for i, g in enumerate(test_genes_with_data):
        obs_delta = test_pert_deltas[i]
        # pop_mean
        de, p = evaluate_pred(train_pert_mean_delta, obs_delta)
        rows.append({"test_gene": g, "mode": "pop_mean", "DE_Spearman": de, "Pearson_delta_full": p})
        # ridge
        de, p = evaluate_pred(ridge_preds[i], obs_delta)
        rows.append({"test_gene": g, "mode": "ridge_gene_identity", "DE_Spearman": de, "Pearson_delta_full": p})
        # knn
        de, p = evaluate_pred(knn_preds[i], obs_delta)
        rows.append({"test_gene": g, "mode": "knn_gene_identity_k10", "DE_Spearman": de, "Pearson_delta_full": p})

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "nondl_baselines_replogle_k562.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")

    # Summary
    summary = df.groupby("mode").agg(
        DE_mean=("DE_Spearman", "mean"),
        DE_std=("DE_Spearman", "std"),
        P_mean=("Pearson_delta_full", "mean"),
        P_std=("Pearson_delta_full", "std"),
        n=("DE_Spearman", "count"),
    ).reset_index()
    summary["dataset"] = "Replogle_K562"
    print("\n=== Replogle K562 non-DL baselines (1 seed, n=" + str(len(test_genes_with_data)) + " test genes) ===")
    print(summary.to_string(index=False))

    # Compute gap (baseline - pop_mean) per test gene for ridge and knn
    pivot = df.pivot_table(index="test_gene", columns="mode", values=["DE_Spearman", "Pearson_delta_full"])
    gaps = []
    for mode in ["ridge_gene_identity", "knn_gene_identity_k10"]:
        de_gap = (pivot[("DE_Spearman", mode)] - pivot[("DE_Spearman", "pop_mean")]).mean()
        p_gap = (pivot[("Pearson_delta_full", mode)] - pivot[("Pearson_delta_full", "pop_mean")]).mean()
        de_std = (pivot[("DE_Spearman", mode)] - pivot[("DE_Spearman", "pop_mean")]).std(ddof=1)
        gaps.append({"baseline": mode, "DE_gap_mean": de_gap, "DE_gap_std": de_std,
                     "Pearson_gap_mean": p_gap, "n": len(test_genes_with_data)})
    gaps_df = pd.DataFrame(gaps)
    print("\n=== Gap (baseline − pop_mean) on K562 ===")
    print(gaps_df.to_string(index=False))

    summary_path = ROOT / "results" / "nondl_baselines_summary.csv"
    summary.to_csv(summary_path, index=False)
    gaps_df.to_csv(ROOT / "results" / "nondl_baselines_gap_vs_popmean.csv", index=False)
    print(f"[saved summary] {summary_path}")
    print(f"[saved gaps] {ROOT / 'results' / 'nondl_baselines_gap_vs_popmean.csv'}")


if __name__ == "__main__":
    main()
