"""Train + audit the Conditional Perturbation Generator (CPG) on Replogle K562
or RPE1. The decisive test: does replacing nn.Embedding with an MLP-over-
gene-identity escape gradient starvation under a 0/1 split?

Pre-registered prediction (frozen 2026-05-04 after CPG design):
- Standard CPA on Replogle K562 has gap (learned − pop_mean) = −0.16, d=−0.65
- CPG should have a gap closer to zero (improvement on the negative gap),
  potentially positive, because:
  1. The CPG's MLP weights are updated during training via ALL training
     pert-genes, so the function MLP(gene_identity → emb) is well-defined.
  2. For test genes, the MLP can interpolate (the identity vector is in the
     trained-input distribution if the gene is a HVG / high-coverage gene).
  3. Therefore "learned" mode for test genes uses a *trained* function applied
     to the test gene's actual identity vector, not a random-init lookup row.

If the CPG gap is *not* closer to zero, the gradient-starvation hypothesis is
refined: even with non-lookup architecture, the decoder may have entered
posterior collapse onto μ_pop (Section 3.7 warm-start failure).
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

from src.data_replogle import build_split_replogle, build_gene_vocab_replogle
from src.data_replogle_rpe1 import load_replogle_rpe1
from src.data_norman import PerturbSeqDataset
from src.cpa_cpg import CPGModel, CPGConfig, compute_gene_identity_table, compute_gene_identity_table_go
from src.audit import predict_under_mode, get_top_deg_indices
from scipy.stats import spearmanr
import scanpy as sc


def load_replogle_no_hvg(path: Path, max_cells: int = 80000, seed: int = 0):
    """Load Replogle adata WITHOUT HVG restriction (for full-panel gene-identity computation)."""
    adata = sc.read_h5ad(path)
    adata.obs['nperts'] = adata.obs['nperts'].astype(int) if 'nperts' in adata.obs.columns else 0
    pert_genes = []
    for p in adata.obs['perturbation']:
        s = str(p).strip()
        if s == 'control' or 'NegCtrl' in s or s == 'unassigned':
            pert_genes.append([])
        else:
            base = s.split('.')[0].split(';')[0]
            pert_genes.append([base])
    adata.obs['pert_genes'] = np.array(pert_genes, dtype=object)
    adata.obs['n_pert'] = [len(x) for x in adata.obs['pert_genes']]

    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata  # full-panel


def restrict_to_hvg(adata_full, n_top_hvg: int = 2000):
    """Apply HVG restriction (controls only) to a full-panel adata."""
    ctrl_mask = (adata_full.obs['n_pert'] == 0).values
    ctrl_only = adata_full[ctrl_mask].copy()
    sc.pp.highly_variable_genes(ctrl_only, n_top_genes=n_top_hvg, subset=False, flavor='seurat')
    hvg_mask = ctrl_only.var['highly_variable'].values
    return adata_full[:, hvg_mask].copy()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_cells", type=int, default=80000)
    ap.add_argument("--max_test_genes", type=int, default=80)
    ap.add_argument("--n_per_pair", type=int, default=200)
    ap.add_argument("--gene_id_dim", type=int, default=64)
    ap.add_argument("--out_tag", default="cpg_replogle_k562")
    ap.add_argument("--dataset", default="replogle_k562", choices=["replogle_k562", "replogle_rpe1"])
    ap.add_argument("--gene_id_source", default="svd", choices=["svd", "go"],
                    help="Gene identity source: 'svd' (expression PCA) or 'go' (GO annotation SVD)")
    ap.add_argument("--gaf_path", default=None,
                    help="Path to GAF file (required if --gene_id_source=go)")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[cpg] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()
    if args.dataset == "replogle_k562":
        DATA_PATH = ROOT / "data" / "replogle_k562_essential.h5ad"
    else:
        DATA_PATH = ROOT / "data" / "replogle_rpe1_essential.h5ad"
    adata_full = load_replogle_no_hvg(DATA_PATH, max_cells=args.max_cells, seed=args.seed)
    print(f"[cpg] full-panel adata {adata_full.shape} in {time.time()-t0:.1f}s")

    # HVG-restricted view for the model
    adata = restrict_to_hvg(adata_full, n_top_hvg=2000)
    print(f"[cpg] HVG-restricted: {adata.shape}")
    vocab = build_gene_vocab_replogle(adata)
    split = build_split_replogle(adata, test_frac_genes=0.2, seed=args.seed)

    if len(split['test_genes']) > args.max_test_genes:
        rng = np.random.default_rng(args.seed)
        split['test_genes'] = list(rng.choice(np.array(split['test_genes'], dtype=object),
                                              size=args.max_test_genes, replace=False))
        keep = set(split['test_genes'])
        new_test = []
        n_pert_arr = adata.obs['n_pert'].values
        pg_arr = adata.obs['pert_genes'].values
        for ci, cell_id in enumerate(adata.obs.index):
            if n_pert_arr[ci] == 1 and pg_arr[ci][0] in keep:
                new_test.append(cell_id)
        split['test_cells'] = new_test
        train_cells = []
        for ci, cell_id in enumerate(adata.obs.index):
            if n_pert_arr[ci] == 0:
                train_cells.append(cell_id)
            elif n_pert_arr[ci] == 1 and pg_arr[ci][0] not in keep:
                train_cells.append(cell_id)
        split['train_cells'] = train_cells
        print(f"[cpg] capped test_genes={len(split['test_genes'])}, train={len(train_cells)}, test={len(new_test)}")

    if args.gene_id_source == "go":
        gaf = args.gaf_path or str(ROOT / "data" / "goa_human.gaf.gz")
        print(f"[cpg] computing gene identity table from GO annotations ({gaf})...")
        gene_id_table = compute_gene_identity_table_go(gaf, vocab, d_id=args.gene_id_dim, seed=args.seed)
        del adata_full
    else:
        print(f"[cpg] computing gene identity table from expression SVD...")
        gene_id_table = compute_gene_identity_table(
            adata_full, vocab, split['ctrl_cells'], d_id=args.gene_id_dim, seed=args.seed,
        )
        del adata_full
    print(f"[cpg] gene_id_table shape: {gene_id_table.shape}")

    cfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab),
                    gene_id_dim=args.gene_id_dim, z_dim=64, pert_dim=32, hidden=256,
                    pert_mlp_hidden=128)
    model = CPGModel(cfg, gene_id_table)
    print(f"[cpg] params {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M (excluding frozen identity table)")

    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)

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
    torch.save({"state_dict": model.state_dict(),
                "cfg": cfg.__dict__,
                "gene_id_table_shape": list(gene_id_table.shape),
                "vocab": vocab,
                "split": split},
               out / f"model_{args.out_tag}_seed{args.seed}.pt")

    # === Audit ===
    obs = adata.obs
    train_set = set(split['train_cells'])
    pg = obs['pert_genes'].values
    n_pert_arr = obs['n_pert'].values
    train_pert_genes_set = set()
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set:
            for g in pg[ci]:
                if g in vocab: train_pert_genes_set.add(g)
    train_pert_gene_indices = sorted(vocab[g] for g in train_pert_genes_set)

    train_perturbed_cell_ids = []
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)

    rng = np.random.default_rng(args.seed)
    # NB: For CPG, "training pert embedding pool" is computed by passing all
    # training pert ids through the MLP (not by reading from a lookup table).
    train_idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long)
    train_idx_doubles = train_idx_t.unsqueeze(1)  # (T, 1) — single perturbation per "row"
    # Pad to max_perts=2 by appending zeros
    train_idx_doubles = torch.cat([train_idx_doubles, torch.zeros_like(train_idx_doubles)], dim=1)  # (T, 2)
    with torch.no_grad():
        train_embed_pool = model.pert_encoder(train_idx_doubles)  # (T, pert_dim)

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
    for tg in split['test_genes']:
        mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5: continue
        pert_X = get(pert_rows)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=200)
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)
        obs_delta_top = obs_delta_full[deg_idx]
        basal_rows = rng.choice(ctrl_rows, size=args.n_per_pair, replace=True)
        basal = torch.from_numpy(get(basal_rows))
        idx = vocab.get(tg, 0)
        pert_idx = torch.tensor([[idx, 0]] * args.n_per_pair, dtype=torch.long)
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
                    "test_gene": tg, "mode": mode,
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
        n=('test_gene', 'count'),
    ).round(4)
    print(f"\n=== CPG ({args.dataset}) audit summary ===")
    print(summary.to_string())
    summary.to_csv(out / f"audit_{args.out_tag}_seed{args.seed}_summary.csv")

    # Per-gene paired gap
    piv = df.pivot_table(index='test_gene', columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean'])
    gap = piv['learned'] - piv['pop_mean']
    rng2 = np.random.default_rng(0)
    boot = np.array([rng2.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
    print(f"\n[CPG] gap (learned − pop_mean) DE-Spearman: n={len(gap)}, mean={gap.mean():+.4f}, "
          f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], "
          f"d={gap.mean()/gap.std(ddof=1):+.3f}")


if __name__ == "__main__":
    main()
