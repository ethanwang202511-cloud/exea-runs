"""Warm-start experiment: replace random-init test-gene embeddings with k-NN-derived
warm starts and re-audit. Direct test of the gradient-starvation hypothesis.

For each held-out test gene g_test on Replogle K562 (or RPE1):
1. Compute the gene-expression similarity between g_test and every training pert
   gene g_train, using their expression-column profiles in the FULL adata
   (cells × genes) matrix as gene identity vectors.
2. Find the K nearest neighbors (cosine similarity) among training pert genes.
3. Set e_warm(g_test) = mean of e_learned(g_train) over those K neighbors.
4. Re-run the 6-mode audit + a NEW 'warm_start' mode that uses these warm
   embeddings for test genes.

Hypothesis from §4.2: random-init test embeddings → off-trained-subspace decoder
response → learned mode under-performs pop_mean. If warm-start succeeds at flipping
the gap toward zero or positive, the gradient-starvation mechanism is confirmed.
If warm-start gives same result, the mechanism is more subtle.

Run on Mac CPU. Mac CPU ~5 min loading adata + ~5 min auditing.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"

from src.cpa_minimal import CPAMinimal, CPAConfig
from src.audit import predict_under_mode, get_top_deg_indices
from scipy.stats import spearmanr


def load_ckpt(path: Path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = CPAConfig(**ck['cfg'])
    model = CPAMinimal(cfg)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck['vocab'], ck['split']


def compute_gene_identity_matrix(adata, vocab):
    """For each gene in vocab, compute its identity vector = column-vector in
    the (cells × genes) expression matrix.

    But wait: vocab maps PERTURBATION-GENE NAMES to embedding-row indices. The
    identity of pert-gene `g` is the row of the gene `g` in adata.var (i.e., its
    expression profile across cells). We need to look up `g` in adata.var_names
    if it exists (HVG-restricted may exclude some).

    For genes not in adata.var_names, return None as identity (we'll fall back).
    """
    var_to_idx = {name: i for i, name in enumerate(adata.var_names)}
    X = adata.X
    if hasattr(X, 'toarray'):
        Xd = np.asarray(X.toarray()).astype(np.float32)
    else:
        Xd = np.asarray(X).astype(np.float32)
    n_cells, n_genes_hvg = Xd.shape
    print(f"[identity] adata X shape: {Xd.shape}, n vocab genes: {len(vocab)}")
    gene_identities = {}
    n_in_hvg = 0
    for gname, idx in vocab.items():
        if gname in var_to_idx:
            gene_identities[gname] = Xd[:, var_to_idx[gname]]
            n_in_hvg += 1
    print(f"[identity] {n_in_hvg}/{len(vocab)} pert genes are in adata.var_names (HVG)")
    return gene_identities, var_to_idx


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def find_knn(g_test_id_vec: np.ndarray, train_gene_ids: dict, k: int = 10):
    """Return the top-k training pert gene names by cosine similarity."""
    sims = []
    for gname, ivec in train_gene_ids.items():
        s = cosine_sim(g_test_id_vec, ivec)
        sims.append((gname, s))
    sims.sort(key=lambda x: -x[1])
    return sims[:k]


@torch.no_grad()
def run_warmstart_audit(ckpt_path: Path, adata_loader_fn, out_tag: str, k: int = 10):
    print(f"\n=== {out_tag} warm-start audit ===")
    model, vocab, split = load_ckpt(ckpt_path)
    cfg = model.cfg
    print(f"[warmstart] cfg pert_dim={cfg.pert_dim}, n_pert_genes={cfg.n_pert_genes}")

    adata = adata_loader_fn(n_top_hvg=2000, max_cells=80000, seed=0)
    obs = adata.obs
    pg = obs['pert_genes'].values
    n_pert_arr = obs['n_pert'].values
    train_set_cells = set(split['train_cells'])
    test_set_genes = set(split['test_genes']) if 'test_genes' in split else set()
    train_pert_genes = set()
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set_cells:
            for g in pg[ci]:
                if g in vocab: train_pert_genes.add(g)
    print(f"[warmstart] train pert genes: {len(train_pert_genes)}, test genes: {len(test_set_genes)}")

    # Compute gene identity vectors (per-gene expression columns across all cells)
    gene_ids, var_to_idx = compute_gene_identity_matrix(adata, vocab)

    # For each test gene, find k-NN training pert genes and compute warm-start embedding
    train_gene_ids = {g: gene_ids[g] for g in train_pert_genes if g in gene_ids}
    print(f"[warmstart] training pert genes with valid identity vectors: {len(train_gene_ids)}")

    warm_embeddings = {}
    knn_records = []
    for g_test in test_set_genes:
        if g_test not in gene_ids:
            # Fallback: just use mean of all training embeddings
            train_indices = [vocab[g] for g in train_pert_genes if g in vocab]
            warm_e = model.pert_encoder.embed.weight.data[train_indices].mean(0).numpy()
            warm_embeddings[g_test] = warm_e
            knn_records.append({"test_gene": g_test, "neighbor": "FALLBACK_MEAN", "sim": 0, "k": 0})
            continue
        g_test_id = gene_ids[g_test]
        knn = find_knn(g_test_id, train_gene_ids, k=k)
        train_indices = [vocab[g] for g, _ in knn]
        warm_e = model.pert_encoder.embed.weight.data[train_indices].mean(0).numpy()
        warm_embeddings[g_test] = warm_e
        for g_neighbor, sim in knn:
            knn_records.append({"test_gene": g_test, "neighbor": g_neighbor, "sim": sim, "k": k})

    knn_df = pd.DataFrame(knn_records)
    knn_df.to_csv(RES / f"warmstart_knn_{out_tag}.csv", index=False)
    print(f"[warmstart] saved {len(knn_records)} k-NN records")

    # Pre-existing audit setup
    rng = np.random.default_rng(0)
    train_pert_indices = sorted(vocab[g] for g in train_pert_genes)
    train_embed_pool = model.pert_encoder.embed.weight.data[train_pert_indices].clone()
    train_perturbed_cell_ids = []
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set_cells and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)
    ctrl_to_row = {c: i for i, c in enumerate(obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in split['ctrl_cells']])
    X = adata.X
    is_sparse = hasattr(X, 'toarray')
    def get(rows): return np.asarray(X[rows].toarray()).astype(np.float32) if is_sparse else np.asarray(X[rows]).astype(np.float32)
    ctrl_X = get(ctrl_rows)
    train_pert_rows = np.array([ctrl_to_row[c] for c in train_perturbed_cell_ids])
    tpm = get(train_pert_rows).mean(0)
    train_pert_mean = torch.from_numpy(tpm)

    rows_out = []
    for tg in test_set_genes:
        mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5: continue
        pert_X = get(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=200)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]
        basal_rows = rng.choice(ctrl_rows, size=200, replace=True)
        basal = torch.from_numpy(get(basal_rows))
        idx = vocab.get(tg, 0)
        pert_idx = torch.tensor([[idx, 0]] * 200, dtype=torch.long)

        # Standard six modes:
        for mode in ("learned", "mean", "zero", "random", "identity", "pop_mean"):
            x_hat = predict_under_mode(model, basal, pert_idx, mode, train_embed_pool, rng,
                                       train_pert_mean_x=train_pert_mean).numpy()
            pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
            pred_delta_top = pred_delta[deg_idx]
            rho = spearmanr(pred_delta_top, obs_delta_top).statistic
            num = np.dot(pred_delta - pred_delta.mean(), obs_delta_full - obs_delta_full.mean())
            denom = (np.linalg.norm(pred_delta - pred_delta.mean()) *
                     np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
            pearson = float(num / denom)
            rows_out.append({
                "test_gene": tg, "mode": mode,
                "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                "Pearson_delta_full": pearson,
            })

        # NEW: warm_start mode — override the test gene's row with warm-start
        # We use pert_embed_override on the model directly
        warm_e = torch.from_numpy(warm_embeddings[tg]).float().unsqueeze(0).expand(200, -1).clone()
        z = model.encode(basal)
        x_hat_warm = model.decode(z, warm_e).numpy()
        pred_delta_warm = x_hat_warm.mean(0) - ctrl_X.mean(0)
        pred_delta_top_warm = pred_delta_warm[deg_idx]
        rho_warm = spearmanr(pred_delta_top_warm, obs_delta_top).statistic
        num_w = np.dot(pred_delta_warm - pred_delta_warm.mean(), obs_delta_full - obs_delta_full.mean())
        denom_w = (np.linalg.norm(pred_delta_warm - pred_delta_warm.mean()) *
                   np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
        pearson_warm = float(num_w / denom_w)
        rows_out.append({
            "test_gene": tg, "mode": "warm_start",
            "DE_Spearman": float(rho_warm) if not np.isnan(rho_warm) else np.nan,
            "Pearson_delta_full": pearson_warm,
        })

        # ALSO: knn_pop_mean baseline (pathway-overlap-free pop_mean):
        # Predict the mean of training-perturbed cells whose perturbation is NOT in
        # the k-NN-similar set (i.e., genes pathway-DISTANT from test gene).
        knn_set = set(g for g, _ in knn) if g_test in gene_ids else set()
        # Far-from-knn training-pert cells
        far_pert_cells = []
        for ci, cell_id in enumerate(obs.index):
            if cell_id in train_set_cells and n_pert_arr[ci] >= 1:
                cell_pert = pg[ci][0] if len(pg[ci]) == 1 else None
                if cell_pert and cell_pert not in knn_set:
                    far_pert_cells.append(cell_id)
        if len(far_pert_cells) > 100:
            far_rows = np.array([ctrl_to_row[c] for c in far_pert_cells])
            far_pop_mean = get(far_rows).mean(0)
            pred_delta_far = far_pop_mean - ctrl_X.mean(0)
            pred_delta_top_far = pred_delta_far[deg_idx]
            rho_far = spearmanr(pred_delta_top_far, obs_delta_top).statistic
            num_f = np.dot(pred_delta_far - pred_delta_far.mean(), obs_delta_full - obs_delta_full.mean())
            denom_f = (np.linalg.norm(pred_delta_far - pred_delta_far.mean()) *
                       np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
            pearson_far = float(num_f / denom_f)
            rows_out.append({
                "test_gene": tg, "mode": "pop_mean_far",
                "DE_Spearman": float(rho_far) if not np.isnan(rho_far) else np.nan,
                "Pearson_delta_full": pearson_far,
            })

    df = pd.DataFrame(rows_out)
    out_csv = RES / f"audit_warmstart_{out_tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        Pearson_full_mean=('Pearson_delta_full', 'mean'),
        n=('test_gene', 'count'),
    ).round(4)
    print("\n=== Warm-start audit summary ===")
    print(summary.to_string())
    summary.to_csv(RES / f"audit_warmstart_{out_tag}_summary.csv")

    # Per-test-gene gap analysis: warm_start − pop_mean
    piv = df.pivot_table(index='test_gene', columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean', 'warm_start'])
    rng2 = np.random.default_rng(0)
    print("\n=== Paired per-gene gap analysis (DE-Spearman) ===")
    for ablation in ('learned', 'warm_start'):
        gap = piv[ablation] - piv['pop_mean']
        boot = np.array([rng2.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
        d = gap.mean() / gap.std(ddof=1) if gap.std(ddof=1) > 0 else 0
        print(f"  {ablation} − pop_mean: mean={gap.mean():+.4f}, 95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], d={d:+.3f}")
    if 'pop_mean_far' in piv.columns:
        gap_far = piv['learned'] - piv['pop_mean_far']
        boot = np.array([rng2.choice(gap_far.values, size=len(gap_far), replace=True).mean() for _ in range(2000)])
        d = gap_far.mean() / gap_far.std(ddof=1)
        print(f"  learned − pop_mean_far (pathway-overlap-free): mean={gap_far.mean():+.4f}, 95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], d={d:+.3f}")


def main():
    from src.data_replogle import load_replogle
    from src.data_replogle_rpe1 import load_replogle_rpe1
    rk_pt = ROOT / "results" / "model_replogle_k562_diag_seed0.pt"
    if rk_pt.exists():
        run_warmstart_audit(rk_pt, load_replogle, "replogle_k562_diag_seed0", k=10)
    else:
        print(f"[skip k562] {rk_pt} not found")

    rpe_pt = ROOT / "results" / "model_rpe1_seed0_seed0.pt"
    if rpe_pt.exists():
        run_warmstart_audit(rpe_pt, load_replogle_rpe1, "rpe1_seed0", k=10)
    else:
        print(f"[skip rpe1] {rpe_pt} not found")


if __name__ == "__main__":
    main()
