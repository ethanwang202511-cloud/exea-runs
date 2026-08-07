"""Re-run audit on saved Norman-default and RPE1 models, saving FULL delta vectors.

The original audit_*.csv files only stored summary metrics (DE-Spearman, Pearson δ).
For PCA / UMAP visualization of the delta-vector space across modes, we need the
full HVG-dimensional delta vectors.

Output: results/delta_vectors_{tag}.npz with arrays:
  - condition (str array): test condition names
  - mode (str array): mode names
  - pred_delta (n_obs × n_genes): predicted delta-log expression
  - obs_delta (n_obs × n_genes): observed delta-log expression
  - ctrl_mean (n_genes,): control-cell mean
  - hvg_names (str array): gene names
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.cpa_minimal import CPAMinimal, CPAConfig
from src.audit import predict_under_mode, get_top_deg_indices


def reload_model(ckpt_path: Path) -> tuple:
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = CPAConfig(**ck['cfg'])
    model = CPAMinimal(cfg)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck['vocab'], ck['split']


@torch.no_grad()
def save_replogle_or_rpe1(model_pt: Path, adata_loader_fn, out_tag: str,
                           is_rpe1: bool = False):
    print(f"[save_delta] {out_tag} from {model_pt.name}")
    model, vocab, split = reload_model(model_pt)
    adata = adata_loader_fn(n_top_hvg=2000, max_cells=80000, seed=0)
    obs = adata.obs
    pg = obs['pert_genes'].values
    n_pert_arr = obs['n_pert'].values
    train_set = set(split['train_cells'])

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

    rng = np.random.default_rng(0)
    train_embed_pool = model.pert_encoder.embed.weight.data[train_pert_gene_indices].clone()
    ctrl_to_row = {c: i for i, c in enumerate(obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in split['ctrl_cells']])
    X = adata.X
    is_sparse = hasattr(X, 'toarray')
    def get(rows): return np.asarray(X[rows].toarray()).astype(np.float32) if is_sparse else np.asarray(X[rows]).astype(np.float32)
    ctrl_X = get(ctrl_rows)
    ctrl_mean = ctrl_X.mean(0)
    train_pert_rows = np.array([ctrl_to_row[c] for c in train_perturbed_cell_ids])
    tpm = get(train_pert_rows).mean(0)
    train_pert_mean = torch.from_numpy(tpm)

    conds, mode_arr, pred_deltas, obs_deltas = [], [], [], []
    modes = ("learned", "mean", "zero", "random", "identity", "pop_mean")
    for tg in split['test_genes']:
        mask = np.array([(len(g) == 1 and g[0] == tg) for g in pg])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5: continue
        pert_X = get(pert_rows)
        obs_pert_mean = pert_X.mean(0)
        obs_delta = obs_pert_mean - ctrl_mean

        basal_rows = rng.choice(ctrl_rows, size=200, replace=True)
        basal = torch.from_numpy(get(basal_rows))
        idx = vocab.get(tg, 0)
        pert_idx = torch.tensor([[idx, 0]] * 200, dtype=torch.long)
        for mode in modes:
            x_hat = predict_under_mode(model, basal, pert_idx, mode, train_embed_pool, rng,
                                       train_pert_mean_x=train_pert_mean).numpy()
            pred_delta = x_hat.mean(0) - ctrl_mean
            conds.append(tg); mode_arr.append(mode)
            pred_deltas.append(pred_delta); obs_deltas.append(obs_delta)

    np.savez(ROOT / "results" / f"delta_vectors_{out_tag}.npz",
             condition=np.array(conds), mode=np.array(mode_arr),
             pred_delta=np.stack(pred_deltas), obs_delta=np.stack(obs_deltas),
             ctrl_mean=ctrl_mean,
             hvg_names=np.array(adata.var_names.tolist()))
    print(f"[saved] delta_vectors_{out_tag}.npz: {len(conds)} (cond × mode) rows")


def main():
    from src.data_replogle_rpe1 import load_replogle_rpe1
    from src.data_replogle import load_replogle

    rpe1_pt = ROOT / "results" / "model_rpe1_seed0_seed0.pt"
    if rpe1_pt.exists():
        save_replogle_or_rpe1(rpe1_pt, load_replogle_rpe1, "rpe1_seed0", is_rpe1=True)
    else:
        print(f"[skip rpe1] {rpe1_pt} not found")

    # K562 from the diagnostic retrain
    rk_pt = ROOT / "results" / "model_replogle_k562_diag_seed0.pt"
    if rk_pt.exists():
        save_replogle_or_rpe1(rk_pt, load_replogle, "replogle_k562_diag_seed0")
    else:
        print(f"[skip k562] {rk_pt} not found")


if __name__ == "__main__":
    main()
