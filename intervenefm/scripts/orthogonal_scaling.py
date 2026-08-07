"""
Orthogonal-direction scaling sweep.

Reviewer critique: scaling along the LEARNED direction shows monotone-saturating
ρ; this leaves open whether scaling along an ORTHOGONAL direction (random unit
vector) is destructive at all magnitudes.

For α in [0, 0.5, 1.0, 2.0, 3.0]:
  Pick a random unit vector u in pert-embedding space, orthogonal-projected
  away from the learned direction.
  e_scaled = α * ||e_learned|| * u
  measure DE-Spearman.

If ρ degrades monotonically with α along the orthogonal direction, the directional
interpretation of mean<zero is locked.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_norman import load_norman, build_split, build_gene_vocab, PerturbSeqDataset
from src.cpa_minimal import CPAMinimal, CPAConfig
from src.audit import get_top_deg_indices


def train_default():
    np.random.seed(0); torch.manual_seed(0)
    adata = load_norman(n_top_hvg=2000, max_cells=60000, seed=0)
    vocab = build_gene_vocab(adata)
    split = build_split(adata, kind="0/2", seed=0)
    train_ds = PerturbSeqDataset(adata, split['train_cells'], split['ctrl_cells'], vocab, max_perts=2, seed=0)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    cfg = CPAConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), z_dim=64, pert_dim=32, hidden=256)
    model = CPAMinimal(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    print(f"[ortho] training 20 epochs...")
    for epoch in range(20):
        model.train()
        for basal, target, pidx in loader:
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, adata, vocab, split


def main():
    model, adata, vocab, split = train_default()
    rng = np.random.default_rng(1)  # different from scaling_sweep seed
    ctrl_to_row = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in split['ctrl_cells']])
    X = adata.X
    is_sparse = hasattr(X, "toarray")
    def get_X(rows):
        return np.asarray(X[rows].toarray() if is_sparse else X[rows]).astype(np.float32)
    ctrl_X = get_X(ctrl_rows)
    obs_pg = adata.obs['pert_genes'].values

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    n_per_pair = 200
    rows = []
    with torch.no_grad():
        for (g1, g2) in split['test_pairs']:
            mask = np.array([(set(g) == {g1, g2}) and len(g) == 2 for g in obs_pg])
            pert_rows = np.where(mask)[0]
            if len(pert_rows) < 5:
                continue
            pert_X = get_X(pert_rows)
            deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=200)
            obs_delta = pert_X.mean(0) - ctrl_X.mean(0)
            obs_delta_top = obs_delta[deg_idx]

            basal_rows = rng.choice(ctrl_rows, size=n_per_pair, replace=True)
            basal = torch.from_numpy(get_X(basal_rows))
            i1 = vocab.get(g1, 0); i2 = vocab.get(g2, 0)
            pidx = torch.tensor([[i1, i2]] * n_per_pair, dtype=torch.long)

            e_learned = model.get_pert_embed(pidx)  # (B, pert_dim)
            e_norm = e_learned.norm(dim=-1, keepdim=True).mean()  # scalar avg ||e||

            # Random unit vector orthogonal to learned direction (per cell — but learned is identical across batch since pidx is the same)
            e_dir = e_learned[0]  # (pert_dim,)
            e_dir_unit = e_dir / (e_dir.norm() + 1e-9)
            # Sample random vector, project out the learned direction
            r = torch.from_numpy(rng.normal(size=(model.cfg.pert_dim,)).astype(np.float32))
            r_orth = r - (r @ e_dir_unit) * e_dir_unit
            r_orth = r_orth / (r_orth.norm() + 1e-9)
            # Sanity: |r_orth . e_dir_unit| should be ~0
            ortho_score = float(torch.abs(r_orth @ e_dir_unit))

            for alpha in alphas:
                e_scaled = alpha * e_norm * r_orth.unsqueeze(0).expand(n_per_pair, -1)
                z = model.encode(basal)
                x_hat = model.decode(z, e_scaled).numpy()
                pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                pred_delta_top = pred_delta[deg_idx]
                rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                rows.append({
                    "pair": f"{g1}_{g2}",
                    "alpha_orth": alpha,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    "ortho_score": ortho_score,
                })
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "orthogonal_scaling.csv"
    df.to_csv(out, index=False)
    summary = df.groupby('alpha_orth').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        DE_Spearman_sem=('DE_Spearman', lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=('pair', 'count'),
    ).round(4)
    print(summary)
    summary.to_csv(ROOT / "results" / "orthogonal_scaling_summary.csv")
    print(f"[saved] {out}")

if __name__ == "__main__":
    main()
