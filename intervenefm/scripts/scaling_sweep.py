"""
Partial-magnitude embedding sweep on a trained CPA-minimal model.

For α in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
  pred_e = α * learned_e
  measure DE-Spearman.

If DE-Spearman is non-monotonic in α (peaks somewhere ≠ 1), that's evidence
the decoder is non-linear in the embedding magnitude (which explains why
mean-of-embeddings ablation < zero ablation in some configs).

Re-uses the trained model from the multiseed sweep (seed 0).
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

# Re-train the model fresh (the prior runs didn't save model weights)
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
    print(f"[scaling] training for 20 epochs...")
    for epoch in range(20):
        model.train()
        for basal, target, pidx in loader:
            x_hat = model(basal, pidx)
            loss = F.mse_loss(x_hat, target)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 5 == 0:
            print(f"  epoch {epoch+1}/20 loss={loss.item():.4f}")
    return model, adata, vocab, split


def scaling_sweep(model, adata, vocab, split):
    rng = np.random.default_rng(0)
    ctrl_cell_ids = split['ctrl_cells']
    test_pairs = split['test_pairs']
    ctrl_to_row = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in ctrl_cell_ids])
    X = adata.X
    is_sparse = hasattr(X, "toarray")
    def get_X(rows):
        return np.asarray(X[rows].toarray() if is_sparse else X[rows]).astype(np.float32)
    ctrl_X = get_X(ctrl_rows)
    obs_pg = adata.obs['pert_genes'].values

    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    rows = []
    n_per_pair = 200
    with torch.no_grad():
        for (g1, g2) in test_pairs:
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

            # Get the actual learned embedding for this pair
            e_learned = model.get_pert_embed(pidx)  # (B, pert_dim)

            for alpha in alphas:
                e_scaled = alpha * e_learned
                z = model.encode(basal)
                x_hat = model.decode(z, e_scaled).numpy()
                pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
                pred_delta_top = pred_delta[deg_idx]
                rho = spearmanr(pred_delta_top, obs_delta_top).statistic
                rows.append({
                    "pair": f"{g1}_{g2}",
                    "alpha": alpha,
                    "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                })
    return pd.DataFrame(rows)


def main():
    model, adata, vocab, split = train_default()
    df = scaling_sweep(model, adata, vocab, split)
    out = ROOT / "results" / "scaling_sweep.csv"
    df.to_csv(out, index=False)
    print(f"[saved] {out}")
    summary = df.groupby('alpha').agg(
        DE_Spearman_mean=('DE_Spearman', 'mean'),
        DE_Spearman_sem=('DE_Spearman', lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        n=('pair', 'count'),
    ).round(4)
    print(summary)
    summary.to_csv(ROOT / "results" / "scaling_sweep_summary.csv")

if __name__ == "__main__":
    main()
