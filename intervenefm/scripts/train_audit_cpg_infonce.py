"""CPG + InfoNCE-style multi-negative contrastive loss (reviewer's lede:
"push toward 100% closure with a contrastive objective").

The single-margin hinge variant (`train_audit_cpg_contrastive.py`) tested
α∈{0.5, 2.0} and margin∈{0, 0.005}; closure plateaued / degraded.

This script tests a *multi-negative* contrastive objective that is closer
to InfoNCE: for each (basal, target, correct_pidx) triple, we compare
the correct-embedding prediction's MSE against K_neg≥1 *random*
embeddings' predictions, and apply a temperature-scaled softmax cross
entropy that pushes the correct embedding to be the closest match.

    sim_k = -MSE(pred_k, target) / d_genes      (per-cell, per-pert)
    p_correct = exp(sim_correct / τ) / sum_k exp(sim_k / τ)
    L_infonce = -log(p_correct)
    L_total = L_mse + α * L_infonce

For each batch of B cells, sample K_neg random perts per cell. Total
forward passes per epoch = B*(1 + K_neg). At K_neg=4 this is 5× the
default CPG cost (~15 min/40 epochs on Replogle K562, vs ~3 min for
default CPG).

Run:
  python3 scripts/train_audit_cpg_infonce.py --seed 0 --epochs 40 --alpha 0.5 --tau 0.1 --k_neg 4
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
from src.data_norman import PerturbSeqDataset
from src.cpa_cpg import CPGModel, CPGConfig, compute_gene_identity_table
from src.audit import predict_under_mode, get_top_deg_indices
from scipy.stats import spearmanr
import scanpy as sc


def load_replogle_no_hvg(path: Path, max_cells: int = 80000, seed: int = 0):
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
    ap.add_argument("--alpha", type=float, default=0.5, help="InfoNCE loss weight")
    ap.add_argument("--tau", type=float, default=0.1, help="InfoNCE temperature")
    ap.add_argument("--k_neg", type=int, default=4, help="Number of negative perts per cell")
    ap.add_argument("--out_tag", default="cpg_infonce")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[cpg_infonce] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    t0 = time.time()
    DATA_PATH = ROOT / "data" / "replogle_k562_essential.h5ad"
    adata_full = load_replogle_no_hvg(DATA_PATH, max_cells=args.max_cells, seed=args.seed)
    print(f"[load] full-panel {adata_full.shape}")
    adata = restrict_to_hvg(adata_full, n_top=2000)
    print(f"[hvg] restricted: {adata.shape}")

    vocab = build_gene_vocab_replogle(adata)
    split = build_split_replogle(adata, test_frac_genes=0.2, seed=args.seed)
    print(f"[split] train={len(split['train_cells'])}, test={len(split['test_cells'])}")

    if len(split['test_genes']) > args.max_test_genes:
        rng = np.random.default_rng(args.seed)
        split['test_genes'] = list(rng.choice(np.array(split['test_genes'], dtype=object),
                                              size=args.max_test_genes, replace=False))
        keep = set(split['test_genes'])
        n_pert_arr = adata.obs['n_pert'].values
        pg_arr = adata.obs['pert_genes'].values
        new_test = []
        for ci, cell_id in enumerate(adata.obs.index):
            if n_pert_arr[ci] == 1 and pg_arr[ci][0] in keep:
                new_test.append(cell_id)
        split['test_cells'] = new_test

    print(f"[gene_id] computing identity table from full panel...")
    gene_id_table = compute_gene_identity_table(adata_full, vocab, split['ctrl_cells'],
                                                 d_id=args.gene_id_dim, seed=args.seed)
    print(f"[gene_id] shape {gene_id_table.shape}")
    del adata_full

    cfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab),
                    gene_id_dim=args.gene_id_dim, z_dim=64, pert_dim=32,
                    hidden=256, pert_mlp_hidden=128)
    model = CPGModel(cfg, gene_id_table)
    print(f"[params] {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable")

    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)

    n_genes = adata.n_vars
    train_log = []
    n_train_perts = len(vocab)  # for sampling negatives

    for epoch in range(args.epochs):
        model.train()
        ep_mse = 0.0; ep_nce = 0.0; ep_n = 0
        for basal, target, pidx in train_loader:
            B = basal.shape[0]
            # Forward correct
            pred_correct = model(basal, pidx)
            loss_mse = F.mse_loss(pred_correct, target)

            # InfoNCE: similarity = -MSE
            # Per-cell MSE for correct prediction
            mse_correct = ((pred_correct - target) ** 2).mean(dim=1)  # (B,)
            # Sample K_neg negatives per cell
            neg_pidx = torch.randint(low=1, high=n_train_perts + 1,
                                     size=(args.k_neg, B, 2),
                                     dtype=torch.long)
            # CRITICAL: Replogle K562 is singles-only; zero col-1 to match
            # the training distribution. Without this, negatives are sampled
            # as DOUBLE perturbations and the contrastive signal compares
            # "correct single" vs "random double" rather than "correct single"
            # vs "random single" (reviewer-flagged correctness fix).
            neg_pidx[:, :, 1] = 0
            # Replace collisions (neg_pidx == correct pidx, ≈0.26/batch at
            # B=128, K=4, n_perts=1973) by shifting to next vocab index.
            collision = (neg_pidx[:, :, 0] == pidx[:, 0].unsqueeze(0))  # (K, B)
            if collision.any():
                neg_pidx[..., 0] = torch.where(
                    collision,
                    (neg_pidx[..., 0] % n_train_perts) + 1,
                    neg_pidx[..., 0],
                )
            # Forward each negative
            mse_negs = []
            for k in range(args.k_neg):
                pred_neg = model(basal, neg_pidx[k])
                mse_neg = ((pred_neg - target) ** 2).mean(dim=1)  # (B,)
                mse_negs.append(mse_neg)
            mse_negs = torch.stack(mse_negs, dim=0)  # (K, B)

            # similarity = -mse, scaled by 1/τ
            sim_correct = -mse_correct / args.tau         # (B,)
            sim_negs = -mse_negs / args.tau               # (K, B)
            # InfoNCE: -log(exp(sim_correct) / (exp(sim_correct) + sum_k exp(sim_negs[k])))
            # Use logsumexp for numerical stability
            all_sims = torch.cat([sim_correct.unsqueeze(0), sim_negs], dim=0)  # (1+K, B)
            log_norm = torch.logsumexp(all_sims, dim=0)                       # (B,)
            log_prob_correct = sim_correct - log_norm
            loss_nce = -log_prob_correct.mean()

            loss = loss_mse + args.alpha * loss_nce
            opt.zero_grad(); loss.backward(); opt.step()
            ep_mse += loss_mse.item() * B
            ep_nce += loss_nce.item() * B
            ep_n += B

        train_log.append({"epoch": epoch+1, "mse": ep_mse/ep_n, "nce": ep_nce/ep_n,
                          "elapsed_s": time.time()-t0})
        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            print(f"[train] epoch {epoch+1}/{args.epochs} mse={ep_mse/ep_n:.4f} nce={ep_nce/ep_n:.4f} ({time.time()-t0:.1f}s)")

    out = ROOT / "results"
    pd.DataFrame(train_log).to_csv(out / f"training_log_{args.out_tag}_seed{args.seed}.csv", index=False)

    # === Audit (same protocol as train_audit_cpg.py) ===
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
    train_perturbed_cell_ids = [c for ci, c in enumerate(obs.index)
                                if c in train_set and n_pert_arr[ci] >= 1]

    rng = np.random.default_rng(args.seed)
    train_idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long)
    train_idx_doubles = torch.cat([train_idx_t.unsqueeze(1),
                                    torch.zeros_like(train_idx_t.unsqueeze(1))], dim=1)
    with torch.no_grad():
        train_embed_pool = model.pert_encoder(train_idx_doubles)

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
                rows_out.append({"test_gene": tg, "mode": mode,
                                 "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                                 "Pearson_delta_full": pearson})

    df = pd.DataFrame(rows_out)
    out_csv = out / f"audit_{args.out_tag}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        Pearson_full_mean=('Pearson_delta_full', 'mean'),
        n=('test_gene', 'count'),
    ).round(4)
    print(f"\n=== CPG-InfoNCE audit summary, seed={args.seed} ===")
    print(summary.to_string())
    summary.to_csv(out / f"audit_{args.out_tag}_seed{args.seed}_summary.csv")

    piv = df.pivot_table(index='test_gene', columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean'])
    gap = piv['learned'] - piv['pop_mean']
    rng2 = np.random.default_rng(0)
    boot = np.array([rng2.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
    print(f"\n[CPG-InfoNCE] gap (learned − pop_mean) DE-Spearman: n={len(gap)}, mean={gap.mean():+.4f}, "
          f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], "
          f"d={gap.mean()/gap.std(ddof=1):+.3f}")


if __name__ == "__main__":
    main()
