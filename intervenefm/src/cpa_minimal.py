"""
Minimal CPA-style model for the intervention-gap audit.

CPA (Lotfollahi et al. Mol Sys Bio 2023) = Compositional Perturbation Autoencoder:
  encoder(x) → z_basal
  pert_emb(p) → e_p (learned per-gene; doubles compose additively: e_{g1+g2} = e_{g1} + e_{g2})
  decoder(z_basal + e_p) → x_pred

For the audit, we want to know: how much does the model rely on e_p vs. on z_basal?
Test: replace e_p with mean / zero / random at INFERENCE time. Measure DE-Spearman drop.

This is a deliberately small reproduction (~100K params) that runs on Mac CPU in <30 min.
The point is the audit, not the model. If the audit gives a clear signal here,
we then run the same audit on real CPA / scGPT-perturb / GEARS for the paper.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple


@dataclass
class CPAConfig:
    n_genes: int               # number of HVGs we model
    n_pert_genes: int          # number of distinct *target* genes that can be perturbed
    z_dim: int = 64
    pert_dim: int = 32
    hidden: int = 256
    dropout: float = 0.1


class PertEncoder(nn.Module):
    """Maps a multi-hot vector of perturbed genes → composed perturbation embedding.

    For singles: one-hot over n_pert_genes (+1 for control = no perturbation).
    For doubles: two-hot. The composition is *additive* through the embedding matrix,
    matching CPA's published behaviour.
    """
    def __init__(self, n_pert_genes: int, pert_dim: int):
        super().__init__()
        # +1 for "control / no perturbation" sentinel — index 0
        self.embed = nn.Embedding(n_pert_genes + 1, pert_dim, padding_idx=0)

    def forward(self, pert_idx: torch.Tensor) -> torch.Tensor:
        # pert_idx shape: (batch, max_perts_per_cell). Padding index = 0 contributes 0.
        # Sum across the perturbation axis = additive composition.
        e = self.embed(pert_idx)               # (B, k, d)
        return e.sum(dim=1)                    # (B, d)


class CPAMinimal(nn.Module):
    def __init__(self, cfg: CPAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.n_genes, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.LayerNorm(cfg.hidden), nn.GELU(),
            nn.Linear(cfg.hidden, cfg.z_dim),
        )
        self.pert_encoder = PertEncoder(cfg.n_pert_genes, cfg.pert_dim)
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

    def forward(
        self,
        x: torch.Tensor,
        pert_idx: torch.Tensor,
        pert_embed_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, n_genes) basal expression (we use control-cell or unperturbed expression
                                         as the encoder input).
        pert_idx: (B, k) integer perturbation indices.
        pert_embed_override: (B, pert_dim) — if provided, use this instead of looking
            up pert_idx. Used by the audit to inject mean/zero/random embeddings.
        """
        z = self.encode(x)
        e = pert_embed_override if pert_embed_override is not None else self.get_pert_embed(pert_idx)
        x_hat = self.decode(z, e)
        return x_hat


def gaussian_nll_loss(x_hat: torch.Tensor, x_true: torch.Tensor, log_var: float = 0.0) -> torch.Tensor:
    """Per-cell Gaussian NLL with shared scalar log-variance (cheap baseline)."""
    sq = (x_hat - x_true) ** 2
    return 0.5 * (sq / np.exp(log_var) + log_var).mean()
