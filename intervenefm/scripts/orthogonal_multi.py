"""
Multi-direction orthogonal scaling: sample 8 random orthogonal directions per
test pair (instead of 1), then aggregate. Locks the directional claim against
"single random vector might be unlucky."
"""
from __future__ import annotations
import sys, time
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

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0]
N_ORTHO_DRAWS = 8


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
    print(f"[ortho_multi] training 20 epochs...")
    for epoch in range(20):
        model.train()
        for basal, target, pidx in loader:
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, adata, vocab, split


def main():
    t0 = time.time()
    model, adata, vocab, split = train_default()
    print(f"[ortho_multi] training done {time.time()-t0:.1f}s")
    rng = np.random.default_rng(42)  # different seed
    ctrl_to_row = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in split['ctrl_cells']])
    X = adata.X
    is_sparse = hasattr(X, "toarray")
    def get_X(rows):
        return np.asarray(X[rows].toarray() if is_sparse else X[rows]).astype(np.float32)
    ctrl_X = get_X(ctrl_rows)
    obs_pg = adata.obs['pert_genes'].values

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

            e_learned = model.get_pert_embed(pidx)
            e_dir = e_learned[0]
            e_dir_unit = e_dir / (e_dir.norm() + 1e-9)
            e_norm = e_learned.norm(dim=-1).mean()

            # Sample N_ORTHO_DRAWS random orthogonal unit vectors
            for draw_idx in range(N_ORTHO_DRAWS):
                r = torch.from_numpy(rng.normal(size=(model.cfg.pert_dim,)).astype(np.float32))
                r_orth = r - (r @ e_dir_unit) * e_dir_unit
                r_orth = r_orth / (r_orth.norm() + 1e-9)

                for alpha in ALPHAS:
                    e_scaled = (alpha * e_norm * r_orth).unsqueeze(0).expand(n_per_pair, -1)
                    z = model.encode(basal)
                    x_hat = model.decode(z, e_scaled).numpy()
                    pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                    pred_delta_top = pred_delta[deg_idx]
                    rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                    rows.append({
                        "pair": f"{g1}_{g2}",
                        "draw": draw_idx,
                        "alpha_orth": alpha,
                        "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                    })
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "orthogonal_multi.csv"
    df.to_csv(out, index=False)
    summary = df.groupby('alpha_orth').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        DE_Spearman_sem=('DE_Spearman', lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=('pair', 'count'),
    ).round(4)
    print(summary)
    summary.to_csv(ROOT / "results" / "orthogonal_multi_summary.csv")
    print(f"[saved] {out}")
    print(f"[total] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
