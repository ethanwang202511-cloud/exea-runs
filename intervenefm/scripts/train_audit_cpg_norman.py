"""Train + audit Conditional Perturbation Generator (CPG) on Norman 2019 (0/2 split).

Reviewer-driven generality test: vanilla CPA on Norman 0/2 has gap
(learned − pop_mean) DE-Spearman = +0.120 (positive — the deep model
beats pop_mean on doubles). The constructive CPG fix closes ~64% of
the K562 inversion. Question: does CPG preserve the positive Norman
gap, or does it degrade performance on the easier regime where the
lookup wasn't broken?

This script:
1. Loads Norman 2019 (~89K cells × 5045 HVGs).
2. Builds 0/2 split (held-out double pert pairs; singles for both
   genes also withheld from training).
3. Computes gene-identity table from Norman control cells (~12K cells)
   via truncated-SVD on the full panel.
4. Trains CPGModel on Norman 0/2 for 20 epochs.
5. Audits per-test-pair using all 6 modes (learned, mean, zero, random,
   identity, pop_mean). Modes use the trained MLP for pert encoding,
   matching CPG semantics.

Output:
  results/audit_cpg_norman_seed{seed}.csv  (per-pair × per-mode)
  results/audit_cpg_norman_seed{seed}_summary.csv
  results/training_log_cpg_norman_seed{seed}.csv
  results/model_cpg_norman_seed{seed}.pt
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_norman import (
    load_norman, build_split, build_gene_vocab, PerturbSeqDataset, parse_pert, DATA as NORMAN_DATA
)
from src.cpa_cpg import CPGModel, CPGConfig, compute_gene_identity_table
from src.audit import predict_under_mode, get_top_deg_indices
from scipy.stats import spearmanr
import scanpy as sc


def load_norman_full_panel(max_cells: int | None = None, seed: int = 0):
    """Load Norman without HVG restriction (for full-panel gene-identity computation)."""
    adata = sc.read_h5ad(NORMAN_DATA)
    adata.obs['nperts'] = adata.obs['nperts'].astype(int)
    pert_genes_series = adata.obs['perturbation'].astype(str).apply(parse_pert).astype(object)
    adata.obs['pert_genes'] = pert_genes_series.values
    adata.obs['n_pert'] = [len(x) for x in adata.obs['pert_genes']]
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_cells", type=int, default=60000)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--gene_id_dim", type=int, default=64)
    ap.add_argument("--n_per_pair", type=int, default=100)
    ap.add_argument("--out_tag", default="cpg_norman")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[cpg-norman] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()

    # Load Norman full-panel FIRST (for gene-identity computation on raw
    # gene-expression columns, NOT HVG-restricted). This matches the Replogle
    # CPG path which uses load_replogle_no_hvg → restrict_to_hvg.
    adata_full = load_norman_full_panel(max_cells=args.max_cells, seed=args.seed)
    print(f"[load] Norman full-panel: {adata_full.shape} in {time.time()-t0:.1f}s")

    # HVG-restrict for training (same procedure as load_norman but starting
    # from the full-panel adata we already loaded).
    ctrl_mask = (adata_full.obs['n_pert'] == 0).values
    ctrl_only = adata_full[ctrl_mask].copy()
    sc.pp.highly_variable_genes(
        ctrl_only, n_top_genes=args.hvg, subset=False,
        flavor='seurat', batch_key=None,
    )
    hvg_mask = ctrl_only.var['highly_variable'].values
    adata = adata_full[:, hvg_mask].copy()
    print(f"[hvg] HVG-restricted for training: {adata.shape}")

    vocab = build_gene_vocab(adata)  # built from pert_genes (HVG-independent)
    split = build_split(adata, kind="0/2", seed=args.seed)
    print(f"[split] train_cells={len(split['train_cells'])}, "
          f"test_cells={len(split['test_cells'])}, "
          f"test_pairs={len(split['test_pairs'])}, ctrl={len(split['ctrl_cells'])}")

    # Compute gene-identity table from Norman controls on FULL panel
    # (not HVG-restricted). Most Norman pert-genes are NOT HVGs because
    # HVGs are response genes; computing identity on HVG would give zero
    # vectors for many pert-genes. Reviewer-flagged correctness fix.
    print(f"[gene_id] computing gene identity table from Norman controls (FULL panel)...")
    gene_id_table = compute_gene_identity_table(
        adata_full, vocab, split['ctrl_cells'], d_id=args.gene_id_dim, seed=args.seed,
    )
    print(f"[gene_id] shape={gene_id_table.shape}")
    del adata_full

    cfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab),
                    gene_id_dim=args.gene_id_dim, z_dim=64, pert_dim=32, hidden=256,
                    pert_mlp_hidden=128)
    model = CPGModel(cfg, gene_id_table)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cpg-norman] trainable params {n_params/1e6:.2f}M (excl. frozen identity table)")

    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'],
                                 vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-5)

    train_log = []
    for epoch in range(args.epochs):
        model.train(); ep_loss = 0.0; ep_n = 0
        for basal, target, pidx in train_loader:
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * basal.shape[0]; ep_n += basal.shape[0]
        avg = ep_loss / ep_n
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print(f"[train] epoch {epoch+1}/{args.epochs} mse={avg:.4f} ({time.time()-t0:.1f}s)")
        train_log.append({"epoch": epoch+1, "mse": avg, "elapsed_s": time.time()-t0})

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(train_log).to_csv(out / f"training_log_{args.out_tag}_seed{args.seed}.csv", index=False)

    # === Audit (per-pair on 0/2 split) ===
    obs = adata.obs
    train_set = set(split['train_cells'])
    pg = obs['pert_genes'].values
    n_pert_arr = obs['n_pert'].values

    # Train pert genes (1-indexed in vocab)
    train_pert_genes_set = set()
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set:
            for g in pg[ci]:
                if g in vocab:
                    train_pert_genes_set.add(g)
    train_pert_gene_indices = sorted(vocab[g] for g in train_pert_genes_set)
    print(f"[audit] train pert gene vocab size: {len(train_pert_gene_indices)}")

    train_perturbed_cell_ids = []
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)
    print(f"[audit] {len(train_perturbed_cell_ids)} training perturbed cells (for pop_mean baseline)")

    rng = np.random.default_rng(args.seed)
    # Training-pert "embedding pool" via the trained MLP
    train_idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long)
    train_idx_doubles = torch.cat([train_idx_t.unsqueeze(1),
                                   torch.zeros_like(train_idx_t.unsqueeze(1))], dim=1)
    with torch.no_grad():
        train_embed_pool = model.pert_encoder(train_idx_doubles)
    print(f"[audit] train_embed_pool shape: {train_embed_pool.shape}")

    cell_to_row = {c: i for i, c in enumerate(obs.index)}
    ctrl_rows = np.array([cell_to_row[c] for c in split['ctrl_cells']])
    X = adata.X
    is_sparse = hasattr(X, 'toarray')
    def get(rows): return np.asarray(X[rows].toarray()).astype(np.float32) if is_sparse else np.asarray(X[rows]).astype(np.float32)
    ctrl_X = get(ctrl_rows)
    train_pert_rows = np.array([cell_to_row[c] for c in train_perturbed_cell_ids])
    tpm = get(train_pert_rows).mean(0)
    train_pert_mean = torch.from_numpy(tpm)

    rows_out = []
    for (g1, g2) in split['test_pairs']:
        # Find observed cells for this pair
        mask = np.array([
            (set(g) == {g1, g2}) and (len(g) == 2)
            for g in pg
        ])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5:
            continue
        pert_X = get(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=200)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]

        basal_rows = rng.choice(ctrl_rows, size=args.n_per_pair, replace=True)
        basal = torch.from_numpy(get(basal_rows))
        idx_g1 = vocab.get(g1, 0)
        idx_g2 = vocab.get(g2, 0)
        pert_idx = torch.tensor([[idx_g1, idx_g2]] * args.n_per_pair, dtype=torch.long)

        with torch.no_grad():
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
                    "pair": f"{g1}_{g2}",
                    "mode": mode,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    "Pearson_delta_full": pearson,
                })

    df = pd.DataFrame(rows_out)
    out_csv = out / f"audit_{args.out_tag}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        Pearson_full_mean=('Pearson_delta_full', 'mean'),
        n=('pair', 'count'),
    ).round(4)
    print(f"\n=== CPG (Norman 0/2) audit summary, seed={args.seed} ===")
    print(summary.to_string())
    summary.to_csv(out / f"audit_{args.out_tag}_seed{args.seed}_summary.csv")

    # Per-pair gap (learned - pop_mean)
    piv = df.pivot_table(index='pair', columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean'])
    gap = piv['learned'] - piv['pop_mean']
    rng2 = np.random.default_rng(0)
    boot = np.array([rng2.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
    print(f"\n[CPG Norman] gap (learned − pop_mean) DE-Spearman: n_pairs={len(gap)}, mean={gap.mean():+.4f}, "
          f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], "
          f"d={gap.mean()/gap.std(ddof=1):+.3f}")

    # Save model
    torch.save({"state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "vocab": vocab},
               out / f"model_{args.out_tag}_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
