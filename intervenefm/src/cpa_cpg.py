"""Conditional Perturbation Generator (CPG): replaces the nn.Embedding lookup
with a small MLP conditioned on a precomputed per-gene identity vector.

Constructive test of the gradient-starvation hypothesis (§4.2 of the paper).
The standard CPA uses `nn.Embedding(n_perts+1, pert_dim)`, so a held-out test
gene's row never receives gradient signal during training and remains at random
init at audit time. The CPG replaces the lookup with:

    e_g = MLP(gene_identity[g])

where gene_identity[g] is a d_id-dim vector derived from gene g's expression
profile across control cells (PCA-compressed). The MLP weights ARE updated
during training (via gradients flowing through ALL training pert genes); for
test genes, the identity vector is in the trained-input distribution, so the
MLP can interpolate. This breaks the gradient starvation pathway.

For doubles, e_(g1, g2) = MLP(id[g1]) + MLP(id[g2]) (additive, matching CPA).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class CPGConfig:
    n_genes: int               # number of HVGs we model
    n_pert_genes: int          # number of distinct *target* genes
    gene_id_dim: int = 64      # dim of gene-identity vector
    z_dim: int = 64
    pert_dim: int = 32
    hidden: int = 256
    pert_mlp_hidden: int = 128
    dropout: float = 0.1


class CPGPertEncoder(nn.Module):
    """Maps perturbation indices → embedding via a small MLP over gene identities.

    Stores a frozen gene_identity_table that is precomputed from control-cell
    expression columns (PCA-reduced).
    """
    def __init__(self, cfg: CPGConfig, gene_identity_table: torch.Tensor):
        """
        gene_identity_table: (n_pert_genes+1, gene_id_dim) — row 0 is reserved
            for control/pad and is set to all zeros. Other rows are precomputed
            gene-identity vectors. The table is registered as a buffer (frozen).
        """
        super().__init__()
        self.cfg = cfg
        assert gene_identity_table.shape == (cfg.n_pert_genes + 1, cfg.gene_id_dim), \
            f"got {gene_identity_table.shape}, expected ({cfg.n_pert_genes + 1}, {cfg.gene_id_dim})"
        self.register_buffer("gene_identity_table", gene_identity_table.float())
        self.mlp = nn.Sequential(
            nn.Linear(cfg.gene_id_dim, cfg.pert_mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.pert_mlp_hidden, cfg.pert_dim),
        )

    def forward(self, pert_idx: torch.Tensor) -> torch.Tensor:
        # pert_idx: (B, k). Look up identity vectors, MLP each, sum.
        ids = self.gene_identity_table[pert_idx]  # (B, k, gene_id_dim)
        # Mask: index 0 (pad) maps to zero identity → MLP(zero) ≠ 0 in general.
        # Force pad → 0 contribution by zeroing those rows post-MLP.
        e = self.mlp(ids)  # (B, k, pert_dim)
        mask = (pert_idx > 0).unsqueeze(-1).float()  # (B, k, 1)
        e = e * mask
        return e.sum(dim=1)  # (B, pert_dim)


class CPGModel(nn.Module):
    """Same encoder/decoder as CPAMinimal; pert_encoder swapped for CPGPertEncoder."""
    def __init__(self, cfg: CPGConfig, gene_identity_table: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.n_genes, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.z_dim),
        )
        self.pert_encoder = CPGPertEncoder(cfg, gene_identity_table)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.z_dim + cfg.pert_dim, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.n_genes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_pert_embed(self, pert_idx: torch.Tensor) -> torch.Tensor:
        return self.pert_encoder(pert_idx)

    def decode(self, z: torch.Tensor, pert_embed: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, pert_embed], dim=-1))

    def forward(self, x, pert_idx, pert_embed_override=None):
        z = self.encode(x)
        e = pert_embed_override if pert_embed_override is not None else self.get_pert_embed(pert_idx)
        return self.decode(z, e)


def compute_gene_identity_table(
    adata_full,           # AnnData WITHOUT HVG restriction
    vocab,                # dict gene_name -> pert_idx (>0 for non-pad)
    ctrl_cell_ids,        # list of control cell ids
    d_id: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """For each pert gene, compute a d_id-dim identity vector via PCA over
    its expression column across control cells. Returns shape (n_perts+1, d_id).
    Row 0 is all zeros (pad / control).

    Uses the FULL gene panel (n_genes_full) not the HVG-restricted matrix.
    Genes in vocab not in adata.var_names get a zero identity (warning printed).
    """
    import scipy.sparse as sp
    import numpy as np

    var_to_idx = {name: i for i, name in enumerate(adata_full.var_names)}
    cell_to_idx = {cid: i for i, cid in enumerate(adata_full.obs.index)}
    ctrl_idx = np.array([cell_to_idx[c] for c in ctrl_cell_ids if c in cell_to_idx])
    print(f"[gene_id] {len(ctrl_idx)} control cells; {len(vocab)} pert genes; PCA target d={d_id}")

    # Subset adata to control cells × all genes
    Xc = adata_full.X[ctrl_idx]
    if sp.issparse(Xc):
        Xc = np.asarray(Xc.toarray()).astype(np.float32)
    else:
        Xc = np.asarray(Xc).astype(np.float32)
    print(f"[gene_id] ctrl-cell expression matrix: {Xc.shape}")

    # For each pert gene, get its column. PCA over genes (rows of Xc.T).
    n_full_genes = Xc.shape[1]
    # Center genes (each gene's expression vector across cells)
    Xc_T = Xc.T  # (n_genes_full, n_ctrl)
    # Standardize per-gene (z-score across cells)
    Xc_T_centered = Xc_T - Xc_T.mean(1, keepdims=True)
    # Truncated SVD for top-d_id components — much faster than full SVD on large matrices
    from scipy.sparse.linalg import svds
    # svds returns (U, S, Vt) with smallest singular values first; reverse.
    print(f"[gene_id] running truncated SVD with k={d_id}...")
    n_min = min(Xc_T_centered.shape) - 1
    k = min(d_id, n_min)
    U, S, Vt = svds(Xc_T_centered, k=k)
    # Reverse order
    idx_order = np.argsort(-S)
    U = U[:, idx_order]; S = S[idx_order]
    gene_coords = U * S  # (n_genes_full, d_id)
    print(f"[gene_id] PCA done; first 5 singular values: {S[:5]}")

    # Build the table: (n_perts + 1, d_id) with row 0 = zeros
    n_perts = max(vocab.values()) if vocab else 0
    table = np.zeros((n_perts + 1, d_id), dtype=np.float32)
    n_resolved = 0
    n_missing = 0
    for gname, pidx in vocab.items():
        if gname in var_to_idx:
            table[pidx] = gene_coords[var_to_idx[gname]]
            n_resolved += 1
        else:
            n_missing += 1
    print(f"[gene_id] {n_resolved}/{len(vocab)} pert genes resolved to identity vectors; {n_missing} missing")
    return torch.from_numpy(table)


def compute_gene_identity_table_go(
    gaf_path: str,
    vocab: dict,
    d_id: int = 64,
    seed: int = 0,
) -> torch.Tensor:
    """Gene identity via GO annotation binary matrix → truncated SVD.

    For each pert gene, builds a binary vector over all GO terms it's annotated
    with, then reduces to d_id dimensions via SVD. Genes not in GAF get zeros.
    Returns shape (n_perts+1, d_id) with row 0 = zeros (pad/control).
    """
    import gzip
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    gene_to_terms: dict[str, set[str]] = {}
    opener = gzip.open if str(gaf_path).endswith('.gz') else open
    with opener(gaf_path, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 15:
                continue
            symbol = parts[2]
            go_term = parts[4]
            if symbol not in gene_to_terms:
                gene_to_terms[symbol] = set()
            gene_to_terms[symbol].add(go_term)
    print(f"[gene_id_go] parsed {len(gene_to_terms)} genes, "
          f"{sum(len(v) for v in gene_to_terms.values())} annotations from GAF")

    all_terms = sorted({t for ts in gene_to_terms.values() for t in ts})
    term_to_idx = {t: i for i, t in enumerate(all_terms)}
    n_terms = len(all_terms)
    print(f"[gene_id_go] {n_terms} unique GO terms")

    pert_genes = sorted(vocab.keys())
    n_pert_in_gaf = sum(1 for g in pert_genes if g in gene_to_terms)
    print(f"[gene_id_go] {n_pert_in_gaf}/{len(pert_genes)} pert genes found in GAF "
          f"({100*n_pert_in_gaf/len(pert_genes):.1f}%)")

    rows, cols = [], []
    for gi, gname in enumerate(pert_genes):
        if gname in gene_to_terms:
            for t in gene_to_terms[gname]:
                rows.append(gi)
                cols.append(term_to_idx[t])

    data = np.ones(len(rows), dtype=np.float32)
    M = csr_matrix((data, (rows, cols)), shape=(len(pert_genes), n_terms))

    k = min(d_id, min(M.shape) - 1)
    print(f"[gene_id_go] running truncated SVD with k={k} on ({M.shape[0]}, {M.shape[1]}) binary matrix...")
    U, S, Vt = svds(M.astype(np.float64), k=k)
    idx_order = np.argsort(-S)
    U = U[:, idx_order]
    S = S[idx_order]
    gene_coords = (U * S).astype(np.float32)
    print(f"[gene_id_go] SVD done; first 5 singular values: {S[:5]}")

    n_perts = max(vocab.values()) if vocab else 0
    table = np.zeros((n_perts + 1, d_id), dtype=np.float32)
    gene_to_row = {g: i for i, g in enumerate(pert_genes)}
    n_resolved = 0
    for gname, pidx in vocab.items():
        if gname in gene_to_row and gname in gene_to_terms:
            vec = gene_coords[gene_to_row[gname]]
            if len(vec) < d_id:
                table[pidx, :len(vec)] = vec
            else:
                table[pidx] = vec[:d_id]
            n_resolved += 1
    print(f"[gene_id_go] {n_resolved}/{len(vocab)} pert genes resolved; "
          f"{len(vocab)-n_resolved} get zero vectors")
    return torch.from_numpy(table)
