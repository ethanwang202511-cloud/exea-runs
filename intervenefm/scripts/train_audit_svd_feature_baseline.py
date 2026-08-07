"""SVD-Feature Lookup baseline — the 2x2 ablation reviewer specifically demanded.

Replaces the CPG's MLP(SVD[g]) with a *frozen lookup*: the perturbation
embedding for gene g is set to its SVD-of-controls vector (projected to
pert_dim=32) and is NOT trained. The encoder, decoder, and rest of the model
train normally.

This isolates the contribution of the SVD-of-controls *feature* from the
contribution of the *MLP head wrapping it*:

    - If gap_SVDfeature ≈ gap_CPG ≈ -0.058   → contribution is the SVD feature
    - If gap_SVDfeature ≈ gap_CPA  ≈ -0.161  → contribution is the MLP head
    - If between                              → both contribute partially

Run:
  python3 scripts/train_audit_svd_feature_baseline.py --seed 0 --epochs 40
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_replogle import build_split_replogle, build_gene_vocab_replogle
from src.data_norman import PerturbSeqDataset
from src.cpa_cpg import compute_gene_identity_table  # reuse SVD computation
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


class FrozenSVDPertEncoder(nn.Module):
    """The SVD-feature lookup baseline. Stores frozen SVD vectors as the
    embedding table; embeddings are NOT trained. For doubles, sums across perts."""
    def __init__(self, gene_identity_table: torch.Tensor, pert_dim: int):
        super().__init__()
        n_perts_plus_one, d_id = gene_identity_table.shape
        # Project d_id -> pert_dim if needed via a FROZEN linear projection
        # (PCA top-pert_dim components of the gene-identity space)
        if d_id == pert_dim:
            proj = gene_identity_table.float()
        else:
            # Take first `pert_dim` PCs (which correspond to top singular values)
            proj = gene_identity_table[:, :pert_dim].float()
        # Normalize per-gene to unit L2 norm so they're comparable to standard
        # nn.Embedding init magnitudes (~sqrt(d) for N(0,1) init)
        # Don't normalize row 0 (pad) since it's all zeros
        norms = proj.norm(dim=1, keepdim=True).clamp_min(1e-8)
        target_norm = float(np.sqrt(pert_dim))
        proj_normed = proj / norms * target_norm
        # Pad row stays zero
        proj_normed[0] = 0.0
        self.register_buffer("embed_table", proj_normed)
        self.pert_dim = pert_dim

    def forward(self, pert_idx: torch.Tensor) -> torch.Tensor:
        # pert_idx: (B, k). Lookup embeddings, sum over k.
        e = self.embed_table[pert_idx]  # (B, k, pert_dim)
        mask = (pert_idx > 0).unsqueeze(-1).float()
        e = e * mask
        return e.sum(dim=1)


from types import SimpleNamespace

class CPAFrozenSVDModel(nn.Module):
    """Same encoder/decoder as CPAMinimal, but with FrozenSVDPertEncoder."""
    def __init__(self, n_genes, gene_identity_table, z_dim=64, pert_dim=32, hidden=256, dropout=0.1):
        super().__init__()
        # Provide cfg attribute for audit.predict_under_mode compatibility
        self.cfg = SimpleNamespace(n_genes=n_genes, z_dim=z_dim, pert_dim=pert_dim, hidden=hidden)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, z_dim),
        )
        self.pert_encoder = FrozenSVDPertEncoder(gene_identity_table, pert_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + pert_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, n_genes),
        )
        self.pert_dim = pert_dim

    def encode(self, x):
        return self.encoder(x)
    def get_pert_embed(self, pert_idx):
        return self.pert_encoder(pert_idx)
    def decode(self, z, e):
        return self.decoder(torch.cat([z, e], dim=-1))
    def forward(self, x, pert_idx, pert_embed_override=None):
        z = self.encode(x)
        e = pert_embed_override if pert_embed_override is not None else self.get_pert_embed(pert_idx)
        return self.decode(z, e)


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
    ap.add_argument("--pert_dim", type=int, default=32)
    ap.add_argument("--out_tag", default="svd_feature_baseline")
    ap.add_argument("--dataset", default="replogle_k562", choices=["replogle_k562", "replogle_rpe1"])
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"[svd_baseline] {vars(args)}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    if args.dataset == "replogle_k562":
        DATA_PATH = ROOT / "data" / "replogle_k562_essential.h5ad"
    else:
        DATA_PATH = ROOT / "data" / "replogle_rpe1_essential.h5ad"

    t0 = time.time()
    adata_full = load_replogle_no_hvg(DATA_PATH, max_cells=args.max_cells, seed=args.seed)
    print(f"[svd_baseline] full-panel adata {adata_full.shape} in {time.time()-t0:.1f}s")
    adata = restrict_to_hvg(adata_full, n_top_hvg=2000)
    print(f"[svd_baseline] HVG-restricted: {adata.shape}")

    # Use Replogle vocab
    vocab = build_gene_vocab_replogle(adata)
    split = build_split_replogle(adata, test_frac_genes=0.2, seed=args.seed)
    print(f"[svd_baseline] train: {len(split['train_cells'])}, test: {len(split['test_cells'])}, "
          f"ctrl: {len(split['ctrl_cells'])}")

    # Compute gene-identity (SVD) table on full panel
    print("[svd_baseline] computing gene identity table on full panel...")
    gene_id_table = compute_gene_identity_table(
        adata_full, vocab, split['ctrl_cells'], d_id=args.gene_id_dim, seed=args.seed
    )
    print(f"[svd_baseline] gene_id_table shape: {gene_id_table.shape}")

    # Build model
    model = CPAFrozenSVDModel(
        n_genes=adata.n_vars,
        gene_identity_table=gene_id_table,
        z_dim=64, pert_dim=args.pert_dim, hidden=256, dropout=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = gene_id_table.numel()
    print(f"[svd_baseline] trainable params {n_params/1e6:.2f}M  (frozen embed table: {n_frozen}={n_frozen/1024:.1f}K floats)")

    # Train (encoder/decoder only — embed table is frozen via register_buffer)
    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)

    train_log = []
    t0_train = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; ep_n = 0
        for basal, target, pidx in train_loader:
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * basal.shape[0]; ep_n += basal.shape[0]
        avg = ep_loss / ep_n
        train_log.append({"epoch": epoch+1, "mse": avg, "elapsed_s": time.time()-t0_train})
        if (epoch+1) % 10 == 0 or epoch == args.epochs-1:
            print(f"[train] epoch {epoch+1}/{args.epochs} mse={avg:.4f} ({time.time()-t0_train:.1f}s)")
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
    print(f"\n=== {args.out_tag} ({args.dataset}) audit summary ===")
    print(summary.to_string())
    summary.to_csv(out_dir / f"audit_{args.out_tag}_seed{args.seed}_summary.csv")

    piv = df.pivot_table(index='test_gene', columns='mode', values='DE_Spearman').dropna(subset=['learned','pop_mean'])
    gap = piv['learned'] - piv['pop_mean']
    rng2 = np.random.default_rng(0)
    boot = np.array([rng2.choice(gap.values, size=len(gap), replace=True).mean() for _ in range(2000)])
    print(f"\n[svd_baseline] gap (learned - pop_mean) DE-Spearman: n={len(gap)}, mean={gap.mean():+.4f}, "
          f"95% CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}], d={gap.mean()/gap.std(ddof=1):+.3f}")


if __name__ == "__main__":
    main()
