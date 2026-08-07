"""
Intervention-gap audit: ablate the perturbation embedding at inference and
measure the DE-Spearman ρ drop.

Four embedding-injection modes:
  1. learned     : use the actual learned embedding e(g1) + e(g2)
  2. mean        : replace each per-gene e(g) with the MEAN embedding over training genes
  3. zero        : replace with zero vector
  4. random      : replace each per-gene e(g) with a random *other* training gene's e
                   (matched cardinality; preserves ||e|| distribution)

For each test pair (g1, g2):
  - Sample K control cells (basal); produce predictions under each mode
  - Compute log fold change vs. control mean for each gene (top-200 DEGs)
  - DE-Spearman ρ = Spearman(pred_LFC, obs_LFC) on the top-200 DEGs of OBSERVED
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from .cpa_minimal import CPAMinimal
from .data_norman import PerturbSeqDataset


def get_top_deg_indices(perturbed_x: np.ndarray, control_x: np.ndarray, k: int = 200) -> np.ndarray:
    """Compute top-k DEGs on observed data by absolute mean-difference (in log-normalized space).

    Inputs are log1p-normalized expression vectors. Their mean-difference (pert - ctrl)
    IS the log fold-change in the log-normalized space (delta-log = log fold-change).
    """
    pert_mean = perturbed_x.mean(axis=0)
    ctrl_mean = control_x.mean(axis=0)
    delta = pert_mean - ctrl_mean
    return np.argsort(-np.abs(delta))[:k]


@torch.no_grad()
def predict_under_mode(
    model: CPAMinimal,
    basal: torch.Tensor,         # (B, n_genes)
    pert_idx: torch.Tensor,      # (B, max_perts)
    mode: str,
    train_embed_pool: torch.Tensor,    # (n_train_pert_genes, pert_dim) — the learned matrix subset for *training* pert-genes only (excluding pad index 0)
    rng: np.random.Generator,
    train_pert_mean_x: torch.Tensor | None = None,  # (n_genes,) avg perturbed-cell expression over training (for pop_mean mode)
) -> torch.Tensor:
    """Run the decoder under the requested ablation mode.

    Modes:
      learned   : actual learned per-gene embedding (the published behaviour)
      mean      : replace per-gene embedding with mean(over training pert-gene embeddings)
      zero      : zero embedding
      random    : random training-gene embedding(s) of matched cardinality
      basal     : just decode z without ANY perturbation embedding contribution to the latent (zero, no encoder change). Equivalent to zero but separated for clarity.
      pop_mean  : non-model baseline — predict the average TRAINING-perturbed-cell expression. Tests whether the model's predictions are no better than predicting the population mean.
      identity  : non-model baseline — predict basal expression unchanged. Tests how far the perturbation effect actually moves x_hat.
    """
    if mode == "learned":
        x_hat = model(basal, pert_idx)
        return x_hat

    if mode == "pop_mean":
        if train_pert_mean_x is None:
            raise ValueError("pop_mean mode requires train_pert_mean_x")
        return train_pert_mean_x.unsqueeze(0).expand(basal.shape[0], -1).clone()
    if mode == "identity":
        return basal.clone()

    z = model.encode(basal)
    B = basal.shape[0]
    pert_dim = model.cfg.pert_dim

    if mode == "zero" or mode == "basal":
        e = torch.zeros(B, pert_dim, device=basal.device)
    elif mode == "mean":
        n_p = (pert_idx > 0).sum(dim=1, keepdim=True).float()
        mean_e = train_embed_pool.mean(dim=0, keepdim=True)
        e = mean_e * n_p
    elif mode == "random":
        # For each cell, replace each perturbation slot with a random training-gene embedding.
        # Sampled WITHOUT replacement within a single cell (so a double becomes 2 distinct
        # random genes, matching the actual cardinality structure).
        n_p = (pert_idx > 0).sum(dim=1).cpu().numpy()
        e = torch.zeros(B, pert_dim, device=basal.device)
        T = train_embed_pool.shape[0]
        for i in range(B):
            k = int(n_p[i])
            if k == 0:
                continue
            picks = rng.choice(T, size=min(k, T), replace=False)
            e[i] = train_embed_pool[picks].sum(dim=0)
    else:
        raise ValueError(mode)

    x_hat = model.decode(z, e)
    return x_hat


@torch.no_grad()
def run_audit(
    model: CPAMinimal,
    adata,
    test_pairs: List[Tuple[str, str]],
    ctrl_cell_ids: List[str],
    gene_vocab: Dict[str, int],
    train_pert_gene_indices: List[int],     # 1-indexed list of pert genes that appear in train singles
    n_per_pair: int = 100,
    seed: int = 0,
    deg_k: int = 200,
    train_perturbed_cell_ids: List[str] | None = None,
) -> pd.DataFrame:
    """Run intervention-gap audit; return per-(test_pair, mode) DE-Spearman ρ."""
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device

    # Build a tensor of training-pert-gene embeddings (NOT including pad-index 0)
    train_embed_pool = model.pert_encoder.embed.weight.data[train_pert_gene_indices].clone()  # (T, pert_dim)
    # Pre-extract control X
    ctrl_to_row = {c: i for i, c in enumerate(adata.obs.index)}
    ctrl_rows = np.array([ctrl_to_row[c] for c in ctrl_cell_ids])
    X = adata.X
    is_sparse = hasattr(X, "toarray")

    def get_X_rows(rows: np.ndarray) -> np.ndarray:
        if is_sparse:
            return np.asarray(X[rows].toarray()).astype(np.float32)
        return np.asarray(X[rows]).astype(np.float32)

    ctrl_X = get_X_rows(ctrl_rows)            # (n_ctrl, n_genes)
    obs_pert_genes = adata.obs['pert_genes'].values

    # Compute training-perturbed-cell mean (for pop_mean baseline)
    train_pert_mean = None
    if train_perturbed_cell_ids is not None and len(train_perturbed_cell_ids) > 0:
        train_pert_rows = np.array([ctrl_to_row[c] for c in train_perturbed_cell_ids if c in ctrl_to_row])
        if len(train_pert_rows) > 0:
            tpx = get_X_rows(train_pert_rows)
            train_pert_mean = torch.from_numpy(tpx.mean(0)).to(device)
            print(f"[audit] pop_mean baseline computed from {len(train_pert_rows)} training perturbed cells")

    rows = []
    for (g1, g2) in test_pairs:
        # Find observed perturbed cells for this pair
        mask = np.array([
            (set(g) == {g1, g2}) and (len(g) == 2)
            for g in obs_pert_genes
        ])
        pert_rows = np.where(mask)[0]
        if len(pert_rows) < 5:
            continue
        pert_X = get_X_rows(pert_rows)        # (n_pert, n_genes)

        # Pre-registered top-200 DEGs (computed on OBSERVED data only — fixed before model query)
        deg_idx = get_top_deg_indices(pert_X, ctrl_X, k=deg_k)
        # delta-log expression (log-fold-change in log-normalized space)
        obs_delta = pert_X.mean(0) - ctrl_X.mean(0)
        obs_delta_top = obs_delta[deg_idx]

        # Sample n_per_pair controls as basal inputs to the model
        basal_rows = rng.choice(ctrl_rows, size=n_per_pair, replace=True)
        basal = torch.from_numpy(get_X_rows(basal_rows)).to(device)
        # Build pert_idx: this pair, repeated
        idx_g1 = gene_vocab.get(g1, 0)
        idx_g2 = gene_vocab.get(g2, 0)
        pert_idx = torch.tensor([[idx_g1, idx_g2]] * n_per_pair, dtype=torch.long, device=device)

        # Pre-compute observed full-transcriptome stats for additional metrics
        obs_pert_mean = pert_X.mean(0)
        obs_delta_full = obs_pert_mean - ctrl_X.mean(0)

        modes = ["learned", "mean", "zero", "random", "identity"]
        if train_pert_mean is not None:
            modes.append("pop_mean")
        for mode in modes:
            x_hat = predict_under_mode(
                model, basal, pert_idx, mode, train_embed_pool, rng,
                train_pert_mean_x=train_pert_mean,
            ).cpu().numpy()
            # Predicted delta-log vs control (LFC in log-normalized space)
            pred_delta = x_hat.mean(0) - ctrl_X.mean(0)
            pred_delta_top = pred_delta[deg_idx]
            rho = spearmanr(pred_delta_top, obs_delta_top).statistic
            # Pearson on full transcriptome delta (Wu/Mejia metric)
            pred_delta_full = pred_delta
            num = np.dot(pred_delta_full - pred_delta_full.mean(), obs_delta_full - obs_delta_full.mean())
            denom = (np.linalg.norm(pred_delta_full - pred_delta_full.mean()) *
                     np.linalg.norm(obs_delta_full - obs_delta_full.mean()) + 1e-12)
            pearson_delta_full = float(num / denom)
            # L1 distance on top-200 DEGs delta-log
            mae_delta = float(np.mean(np.abs(pred_delta_top - obs_delta_top)))
            # Sample-mean MSE
            mse = float(((x_hat - obs_pert_mean[None, :]) ** 2).mean())
            # Variance-ratio in HVG-mean: do predictions show comparable spread to observed?
            pred_var = float(x_hat.var(0).mean())
            obs_var = float(pert_X.var(0).mean())
            var_ratio = pred_var / (obs_var + 1e-9)
            rows.append({
                "pair": f"{g1}_{g2}",
                "mode": mode,
                "DE_Spearman": float(rho) if not np.isnan(rho) else np.nan,
                "Pearson_delta_full": pearson_delta_full,
                "mae_delta_top200": mae_delta,
                "mse_to_pert_mean": mse,
                "var_ratio_hvg_mean": var_ratio,
                "n_obs_pert": len(pert_rows),
            })
    return pd.DataFrame(rows)
