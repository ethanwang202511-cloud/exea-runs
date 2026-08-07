"""
Norman 2019 Perturb-seq loader for the audit.

Provides:
  - load_norman(): scanpy AnnData with normalize_total + log1p
  - build_split(adata, kind): return train_idx / test_idx by 0/2 / 1/2 / 2/2 logic
  - PerturbSeqDataset: yields (basal_x, target_x, pert_idx) triplets aligned
    by *matched control sampling* (each perturbed cell paired with a random control)
"""

from __future__ import annotations
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "norman_2019.h5ad"


def parse_pert(p: str) -> List[str]:
    """'SET_KLF1' -> ['SET','KLF1']; 'control' or 'NegCtrl0' -> []."""
    s = str(p).strip()
    if s in ("control", "Control", "ctrl", "0"):
        return []
    if "NegCtrl" in s:
        return []
    parts = [g for g in s.split("_") if g and "NegCtrl" not in g]
    return parts


def load_norman(n_top_hvg: int = 5000, max_cells: int | None = None, seed: int = 0):
    """Load + normalize + HVG-select Norman."""
    adata = sc.read_h5ad(DATA)
    # 'perturbation' column: 'GENE1' for single, 'GENE1_GENE2' for double, 'control' for ctrl
    # nperts: '0' control, '1' single, '2' double
    adata.obs['nperts'] = adata.obs['nperts'].astype(int)
    # Cast to plain object dtype (pandas categorical doesn't support list values)
    pert_genes_series = adata.obs['perturbation'].astype(str).apply(parse_pert).astype(object)
    adata.obs['pert_genes'] = pert_genes_series.values
    adata.obs['n_pert'] = [len(x) for x in adata.obs['pert_genes']]

    # Limit size for tractability on Mac CPU.
    if max_cells is not None and adata.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        keep = rng.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[np.sort(keep)].copy()

    # Normalize + log1p (standard).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # HVG selection — fit on CONTROL cells only to avoid leakage (was a bug previously
    # which fit on full perturbed+control matrix; flagged by self-check audit).
    ctrl_mask = (adata.obs['n_pert'] == 0).values
    ctrl_only = adata[ctrl_mask].copy()
    sc.pp.highly_variable_genes(
        ctrl_only, n_top_genes=n_top_hvg, subset=False,
        flavor='seurat', batch_key=None,
    )
    hvg_mask = ctrl_only.var['highly_variable'].values
    adata = adata[:, hvg_mask].copy()
    return adata


def build_split(adata: ad.AnnData, kind: str = "0/2", seed: int = 0,
                test_frac_doubles: float = 0.3) -> Dict:
    """Build a GEARS-style 0/2 / 1/2 / 2/2 split for Norman 2019.

    Norman 2019 has every gene-in-a-double also covered as a single, so to construct
    a 0/2 split we must additionally hold out the SINGLE perturbations of the test
    doubles' constituent genes. We follow the standard GEARS protocol:

    - Choose a set of TEST PAIRS (gene1, gene2).
    - 0/2: also remove the singles for gene1 and gene2 from training.
           Test = doubles in test pairs.
    - 1/2: remove only one of (gene1, gene2)'s single from training.
    - 2/2: keep both singles in training (easiest split).
    """
    rng = np.random.default_rng(seed)
    obs = adata.obs

    # Distinct gene pairs in doubles
    double_idx = obs.index[obs['n_pert'] == 2]
    pair_to_cells = {}
    for cell_id in double_idx:
        gs = obs.loc[cell_id, 'pert_genes']
        if len(gs) != 2:
            continue
        key = tuple(sorted(gs))
        pair_to_cells.setdefault(key, []).append(cell_id)
    all_pairs = sorted(pair_to_cells.keys())
    print(f"[split] {len(all_pairs)} unique double-perturbation pairs.")

    # Choose test pairs (random subset)
    n_test_pairs = max(1, int(round(test_frac_doubles * len(all_pairs))))
    test_pair_idx = rng.choice(len(all_pairs), size=n_test_pairs, replace=False)
    test_pairs = set(all_pairs[i] for i in test_pair_idx)
    print(f"[split] held-out pairs: {n_test_pairs} ({100*n_test_pairs/len(all_pairs):.0f}%)")

    # Genes appearing in test pairs
    test_genes = set()
    for g1, g2 in test_pairs:
        test_genes.add(g1); test_genes.add(g2)

    # For each test pair, decide which singles to also withhold (per kind)
    if kind == "2/2":
        single_genes_to_withhold = set()    # keep all singles in training
    elif kind == "1/2":
        # withhold ONE single per test pair (random)
        single_genes_to_withhold = set()
        for g1, g2 in test_pairs:
            single_genes_to_withhold.add(rng.choice([g1, g2]))
    elif kind == "0/2":
        single_genes_to_withhold = set(test_genes)
    else:
        raise ValueError(kind)

    # Single-genes that REMAIN in training
    single_idx = obs.index[obs['n_pert'] == 1]
    train_single_genes = set()
    for cell_id in single_idx:
        gs = obs.loc[cell_id, 'pert_genes']
        if not gs:
            continue
        if gs[0] not in single_genes_to_withhold:
            train_single_genes.add(gs[0])
    print(f"[split] kind={kind}: {len(test_genes)} test genes, {len(single_genes_to_withhold)} singles withheld, "
          f"{len(train_single_genes)} single-genes remain in train.")

    # Verify split statistics: how many of the test-pair genes appear in train singles?
    test_genes_in_train = sum(1 for g in test_genes if g in train_single_genes)
    print(f"[split]   test genes also in train singles: {test_genes_in_train}/{len(test_genes)}")

    # Classify cells
    train_cells = []
    test_cells = []
    ctrl_cells = list(obs.index[obs['n_pert'] == 0])

    n_pert_arr = obs['n_pert'].values
    pert_genes_arr = obs['pert_genes'].values
    cell_ids = obs.index.tolist()
    for ci, cell_id in enumerate(cell_ids):
        n_p = n_pert_arr[ci]
        gs = pert_genes_arr[ci]
        if n_p == 0:
            train_cells.append(cell_id)
        elif n_p == 1:
            if gs[0] in single_genes_to_withhold:
                continue   # drop these singles from train (they pertain to test genes)
            train_cells.append(cell_id)
        else:  # double
            key = tuple(sorted(gs))
            if key in test_pairs:
                test_cells.append(cell_id)
            else:
                train_cells.append(cell_id)

    print(f"[split] train cells: {len(train_cells)}, test cells: {len(test_cells)}, ctrl cells: {len(ctrl_cells)}")

    # Defensive assertion: verify no test gene's single is in train singles
    if kind == "0/2":
        for tg in test_genes:
            assert tg not in train_single_genes, f"LEAKAGE: test gene {tg} also in train singles"
        # And: no test pair's cells leaked to train
        train_set = set(train_cells); test_set = set(test_cells)
        assert len(train_set & test_set) == 0, "LEAKAGE: train/test cell overlap"

    return {
        "train_cells": train_cells,
        "test_cells": test_cells,
        "ctrl_cells": ctrl_cells,
        "test_pairs": list(test_pairs),
        "test_genes": list(test_genes),
        "train_single_genes": list(train_single_genes),
        "withheld_single_genes": list(single_genes_to_withhold),
        "kind": kind,
    }


class PerturbSeqDataset(Dataset):
    """Yield (basal_x, target_x, pert_idx) for each perturbed cell.

    'basal_x' = a randomly-paired CONTROL cell's expression (CPA convention).
    'target_x' = the perturbed cell's observed expression.
    'pert_idx' = padded indices into the gene-vocabulary.

    For control cells (n_pert=0), pert_idx is all zeros (the padding/null index)
    and target_x = basal_x (model should learn identity on controls).
    """
    def __init__(
        self,
        adata: ad.AnnData,
        cell_ids: List[str],
        ctrl_cell_ids: List[str],
        gene_vocab: Dict[str, int],   # gene name -> index (1-indexed; 0 = pad/null)
        max_perts: int = 2,
        seed: int = 0,
    ):
        self.cells = list(cell_ids)
        self.ctrl_cells = list(ctrl_cell_ids)
        self.gene_vocab = gene_vocab
        self.max_perts = max_perts
        self.adata = adata
        self.rng = np.random.default_rng(seed)

        # Pre-fetch indices for fast access
        self._cell_to_row = {c: i for i, c in enumerate(adata.obs.index)}
        self._cell_rows = np.array([self._cell_to_row[c] for c in self.cells])
        self._ctrl_rows = np.array([self._cell_to_row[c] for c in self.ctrl_cells])

        self._X = adata.X
        self._is_sparse = hasattr(self._X, "toarray")

    def __len__(self):
        return len(self.cells)

    def _row(self, idx: int) -> np.ndarray:
        if self._is_sparse:
            return np.asarray(self._X[idx].toarray()).ravel()
        return np.asarray(self._X[idx]).ravel()

    def __getitem__(self, i):
        cell_id = self.cells[i]
        row = self._cell_rows[i]
        target = self._row(row).astype(np.float32)

        # Pair with random control
        ctrl_row = int(self.rng.choice(self._ctrl_rows))
        basal = self._row(ctrl_row).astype(np.float32)

        # Build pert_idx (padded to max_perts)
        gs = self.adata.obs.iloc[row]['pert_genes']
        pidx = np.zeros(self.max_perts, dtype=np.int64)
        for j, g in enumerate(gs[: self.max_perts]):
            pidx[j] = self.gene_vocab.get(g, 0)

        return torch.from_numpy(basal), torch.from_numpy(target), torch.from_numpy(pidx)


def build_gene_vocab(adata: ad.AnnData) -> Dict[str, int]:
    """Map each gene that appears in any perturbation to a 1-indexed integer."""
    all_pert_genes = set()
    for gs in adata.obs['pert_genes']:
        all_pert_genes.update(gs)
    vocab = {g: i + 1 for i, g in enumerate(sorted(all_pert_genes))}
    return vocab
