"""Replogle RPE1 essential Perturb-seq loader for the audit.

Replogle 2022 RPE1 (retinal pigment epithelium): ~250K cells × 8.5K genes,
~2393 single perturbations + control. Same processing as K562 essential.
This loader is identical in logic to data_replogle.py with a different file path.
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
DATA = ROOT / "data" / "replogle_rpe1_essential.h5ad"


def load_replogle_rpe1(n_top_hvg: int = 2000, max_cells: int | None = 80000, seed: int = 0):
    adata = sc.read_h5ad(DATA)
    # Normalize column names — the RPE1 file has the same schema as K562
    if 'nperts' in adata.obs.columns:
        adata.obs['nperts'] = adata.obs['nperts'].astype(int)
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

    ctrl_mask = (adata.obs['n_pert'] == 0).values
    ctrl_only = adata[ctrl_mask].copy()
    sc.pp.highly_variable_genes(
        ctrl_only, n_top_genes=n_top_hvg, subset=False,
        flavor='seurat', batch_key=None,
    )
    hvg_mask = ctrl_only.var['highly_variable'].values
    adata = adata[:, hvg_mask].copy()
    return adata
