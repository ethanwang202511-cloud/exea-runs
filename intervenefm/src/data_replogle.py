"""
Replogle K562 essential Perturb-seq loader for the audit.

Replogle 2022 essentials: ~310K cells × 8.5K genes, ~2057 perturbations,
all SINGLE perturbations + control. We construct a single-pert held-out split:
- pick test_frac of perturbed genes for testing (held out)
- train on remaining singles + controls
- at test time, audit how much of the prediction depends on the embedding
  vs the population mean
"""
from __future__ import annotations
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "replogle_k562_essential.h5ad"


def load_replogle(n_top_hvg: int = 2000, max_cells: int | None = 60000, seed: int = 0):
    adata = sc.read_h5ad(DATA)
    adata.obs['nperts'] = adata.obs['nperts'].astype(int)
    # Replogle: 'perturbation' column = gene name (or 'control')
    pert_genes = []
    for p in adata.obs['perturbation']:
        s = str(p).strip()
        if s == 'control' or s == 'NegCtrl' or 'NegCtrl' in s or s == 'unassigned':
            pert_genes.append([])
        else:
            # Some Replogle pert ids are e.g. "BUB1" or "BUB1.1"; strip suffixes
            base = s.split('.')[0].split(';')[0]
            pert_genes.append([base])
    adata.obs['pert_genes'] = np.array(pert_genes, dtype=object)
    adata.obs['n_pert'] = [len(x) for x in adata.obs['pert_genes']]

    # Subsample for tractability
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG on controls only (no leakage)
    ctrl_mask = (adata.obs['n_pert'] == 0).values
    ctrl_only = adata[ctrl_mask].copy()
    sc.pp.highly_variable_genes(
        ctrl_only, n_top_genes=n_top_hvg, subset=False,
        flavor='seurat', batch_key=None,
    )
    hvg_mask = ctrl_only.var['highly_variable'].values
    adata = adata[:, hvg_mask].copy()
    return adata


def build_split_replogle(adata: ad.AnnData, test_frac_genes: float = 0.2, seed: int = 0,
                          min_cells_per_gene: int = 50) -> Dict:
    """Hold out a fraction of perturbation genes for testing.

    For each gene with >= min_cells_per_gene perturbed cells, decide train vs test.
    Test set = cells whose pert gene is in the test gene list.
    Train set = control + cells whose pert gene is NOT in test list.
    """
    rng = np.random.default_rng(seed)
    obs = adata.obs

    # Count cells per perturbation gene
    gene_counts = {}
    for genes in obs['pert_genes']:
        if not genes:
            continue
        for g in genes:
            gene_counts[g] = gene_counts.get(g, 0) + 1
    eligible = [g for g, n in gene_counts.items() if n >= min_cells_per_gene]
    print(f"[split_replogle] eligible genes (>= {min_cells_per_gene} cells): {len(eligible)}")

    n_test = int(round(test_frac_genes * len(eligible)))
    test_genes = set(rng.choice(np.array(eligible, dtype=object), size=n_test, replace=False))
    print(f"[split_replogle] held-out test genes: {n_test}")

    train_cells, test_cells, ctrl_cells = [], [], []
    n_pert_arr = obs['n_pert'].values
    pg_arr = obs['pert_genes'].values
    for ci, cell_id in enumerate(obs.index):
        if n_pert_arr[ci] == 0:
            train_cells.append(cell_id); ctrl_cells.append(cell_id)
        else:
            g = pg_arr[ci][0] if pg_arr[ci] else None
            if g in test_genes:
                test_cells.append(cell_id)
            else:
                train_cells.append(cell_id)
    print(f"[split_replogle] train: {len(train_cells)}, test: {len(test_cells)}, ctrl: {len(ctrl_cells)}")

    return {
        "train_cells": train_cells,
        "test_cells": test_cells,
        "ctrl_cells": ctrl_cells,
        "test_genes": list(test_genes),
        "kind": "0/1",
    }


def build_gene_vocab_replogle(adata: ad.AnnData) -> Dict[str, int]:
    all_p = set()
    for gs in adata.obs['pert_genes']:
        all_p.update(gs)
    return {g: i + 1 for i, g in enumerate(sorted(all_p))}
