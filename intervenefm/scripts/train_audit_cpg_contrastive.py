"""CPG + counterfactual-consistency loss — the reviewer's specific recommendation
in paper §3.7: "Closing the remaining gap likely requires a different training
objective (counterfactual-consistency or contrastive) rather than further
embedding-layer change."

Implementation: in addition to the standard MSE(pred, target) loss, add a
counterfactual-consistency term that penalizes the model when the *correct*
perturbation embedding gives a worse prediction than a *random* (wrong)
perturbation embedding from the same batch.

    L = L_mse + α * max(0, MSE(pred_correct, target) - MSE(pred_random, target) + margin)

This forces the embedding to actually carry useful per-perturbation information:
the gradient pushes the encoder/decoder to produce predictions that are
better-aligned to the *correct* embedding than to a random one.

Run:
  python3 scripts/train_audit_cpg_contrastive.py --seed 0 --epochs 40 --alpha 0.5 --margin 0.0
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


def load_replogle_no_hvg(path, max_cells: int = 80000, seed: int = 0):
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
    return adata


def restrict_to_hvg(adata_full, n_top_hvg: int = 2000):
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
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight of counterfactual-consistency loss")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="margin: penalty applies if mse_correct - mse_random > -margin")
    ap.add_argument("--max_cells", type=int, default=80000)
    ap.add_argument("--max_test_genes", type=int, default=80)
    ap.add_argument("--n_per_pair", type=int, default=200)
    ap.add_argument("--gene_id_dim", type=int, default=64)
    ap.add_argument("--out_tag", default="cpg_contrastive_replogle_k562")
    ap.add_argument("--dataset", default="replogle_k562", choices=["replogle_k562", "replogle_rpe1"])
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[cpg_cf] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.dataset == "replogle_k562":
        DATA_PATH = ROOT / "data" / "replogle_k562_essential.h5ad"
    else:
        DATA_PATH = ROOT / "data" / "replogle_rpe1_essential.h5ad"

    t0 = time.time()
    adata_full = load_replogle_no_hvg(DATA_PATH, max_cells=args.max_cells, seed=args.seed)
    print(f"[cpg_cf] full-panel adata {adata_full.shape} in {time.time()-t0:.1f}s")
    adata = restrict_to_hvg(adata_full, n_top_hvg=2000)
    print(f"[cpg_cf] HVG-restricted: {adata.shape}")

    vocab = build_gene_vocab_replogle(adata)
    split = build_split_replogle(adata, test_frac_genes=0.2, seed=args.seed)
    print(f"[cpg_cf] train: {len(split['train_cells'])}, test: {len(split['test_cells'])}, ctrl: {len(split['ctrl_cells'])}")

    print("[cpg_cf] computing gene identity table on full panel...")
    gene_id_table = compute_gene_identity_table(adata_full, vocab, split['ctrl_cells'],
                                                 d_id=args.gene_id_dim, seed=args.seed)
    print(f"[cpg_cf] gene_id_table shape: {gene_id_table.shape}")

    cfg = CPGConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab),
                    gene_id_dim=args.gene_id_dim, z_dim=64, pert_dim=32,
                    hidden=256, pert_mlp_hidden=128)
    model = CPGModel(cfg, gene_id_table)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[cpg_cf] params {n_params/1e6:.2f}M (excl. frozen identity table)")

    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)

    train_log = []
    for epoch in range(args.epochs):
        model.train()
        ep_loss_total = 0.0; ep_loss_mse = 0.0; ep_loss_cf = 0.0; ep_n = 0
        for basal, target, pidx in train_loader:
            B = basal.shape[0]
            # Standard MSE branch
            pred_correct = model(basal, pidx)
            loss_mse = F.mse_loss(pred_correct, target)

            # Counterfactual-consistency branch: random pert_idx from same batch
            perm = torch.randperm(B)
            # Avoid case where permuted idx == original idx (very rare for B≥32)
            same = (pidx == pidx[perm]).all(dim=1)
            if same.any():
                # Roll by 1 for those
                fallback = torch.roll(torch.arange(B), shifts=1)
                perm = torch.where(same, fallback, perm)
            pidx_random = pidx[perm]
            pred_random = model(basal, pidx_random)
            mse_correct_per_sample = ((pred_correct - target) ** 2).mean(dim=1)
            mse_random_per_sample = ((pred_random - target) ** 2).mean(dim=1)
            # Hinge: penalize if mse_correct >= mse_random - margin
            # i.e., if the wrong embedding does AT LEAST as well as the correct one
            loss_cf = torch.clamp(mse_correct_per_sample - mse_random_per_sample + args.margin, min=0.0).mean()

            loss = loss_mse + args.alpha * loss_cf
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss_total += loss.item() * B
            ep_loss_mse += loss_mse.item() * B
            ep_loss_cf += loss_cf.item() * B
            ep_n += B
        train_log.append({"epoch": epoch+1,
                          "mse": ep_loss_mse/ep_n,
                          "cf": ep_loss_cf/ep_n,
                          "total": ep_loss_total/ep_n,
                          "elapsed_s": time.time()-t0})
        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            print(f"[train] epoch {epoch+1}/{args.epochs} "
                  f"mse={ep_loss_mse/ep_n:.4f} cf={ep_loss_cf/ep_n:.4f} "
                  f"total={ep_loss_total/ep_n:.4f} ({time.time()-t0:.1f}s)")

    out_dir = ROOT / "results"
    pd.DataFrame(train_log).to_csv(out_dir / f"training_log_{args.out_tag}_seed{args.seed}.csv", index=False)

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

    train_perturbed_cell_ids = []
    for ci, cell_id in enumerate(obs.index):
        if cell_id in train_set and n_pert_arr[ci] >= 1:
            train_perturbed_cell_ids.append(cell_id)

    rng = np.random.default_rng(args.seed)
    train_idx_t = torch.tensor(train_pert_gene_indices, dtype=torch.long)
    train_idx_doubles = train_idx_t.unsqueeze(1)
    train_idx_doubles = torch.cat([train_idx_doubles, torch.zeros_like(train_idx_doubles)], dim=1)
    with torch.no_grad():
        train_embed_pool = model.pert_encoder(train_idx_doubles)

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
    out_csv = out_dir / f"audit_{args.out_tag}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    summary = df.groupby('mode').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        Pearson_full_mean=('Pearson_delta_full', 'mean'),
        n=('test_gene', 'count'),
    ).round(4)
    print(f"\n=== {args.out_tag} (alpha={args.alpha}, margin={args.margin}) audit summary ===")
    print(summary.to_string())
    summary.to_csv(out_dir / f"audit_{args.out_tag}_seed{args.seed}_summary.csv")

    piv = df.pivot_table(index='test_gene', columns='mode', values='DE_Spearman').dropna(subset=['learned','pop_mean'])
    gap = piv['learned'] - piv['pop_mean']
    rng2 = np.random.default_rng(0)
    boot = np.array([rng2.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
    print(f"\n[cpg_cf] gap (learned - pop_mean) DE-Spearman: n={len(gap)}, mean={gap.mean():+.4f}, "
          f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], d={gap.mean()/gap.std(ddof=1):+.3f}")
    print(f"\nReference: standard CPA -0.161; SVD-feature -0.089; CPG -0.058")


if __name__ == "__main__":
    main()
