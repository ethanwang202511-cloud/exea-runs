"""PertFormer: a self-contained, pure-PyTorch attention model for the
intervention-gap audit (P20).

Answers the reviewer complaint that only additive architectures (CPA/CPG) were
tested. Uses ONLY `torch.nn.TransformerEncoderLayer` (SDPA backend under the
hood on torch>=2) — no flash-attn / xformers / apex anywhere, per the harness's
ROCm safety rules.

Per-gene token = E_id(gene) + E_val(control_expr) + E_flag(is_perturbed).
A perturbation query token q_pert is prepended; the decoder head is a
per-token Linear(d_model -> 1) predicting the perturbation-induced delta vs
control for that gene.

Two swappable variants for q_pert (this is the actual experiment):
  Variant A (lookup):        q_pert = nn.Embedding(num_perts, d)
      -> dormancy-prone: a held-out pert gene's row never receives gradient
         under a 0/1 (or 0/2) split, exactly like CPA's nn.Embedding.
  Variant B (CPG-attention):  q_pert = MLP(gene_identity_vector)
      -> feature-grounded: the identity vector (reusing the SVD/GO tables
         from cpa_cpg.py) is in-distribution for held-out genes even though
         the MLP never saw that specific gene during training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class PertFormerConfig:
    n_genes: int                 # modeled gene panel size (sequence length, excl. q_pert token)
    n_pert_genes: int            # distinct target genes (vocab size, 1-indexed; 0 = pad/control)
    variant: str = "A"           # "A" (embedding lookup) or "B" (CPG-attention MLP)
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    gene_id_dim: int = 64        # only used by variant B


class PertFormer(nn.Module):
    def __init__(self, cfg: PertFormerConfig, gene_identity_table: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # Per-gene tokens: E_id(gene) + E_val(control_expr) + E_flag(is_perturbed)
        self.gene_id_emb = nn.Embedding(cfg.n_genes, d)
        self.val_proj = nn.Linear(1, d)
        self.flag_emb = nn.Embedding(2, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(d, 1)

        if cfg.variant == "A":
            # +1 for control/pad sentinel at index 0 (never trained on -> the
            # dormancy mechanism under test).
            self.q_pert = nn.Embedding(cfg.n_pert_genes + 1, d, padding_idx=0)
        elif cfg.variant == "B":
            assert gene_identity_table is not None, "variant B requires a gene_identity_table"
            assert gene_identity_table.shape == (cfg.n_pert_genes + 1, cfg.gene_id_dim), (
                f"got {tuple(gene_identity_table.shape)}, "
                f"expected ({cfg.n_pert_genes + 1}, {cfg.gene_id_dim})"
            )
            self.register_buffer("gene_identity_table", gene_identity_table.float())
            self.q_pert_mlp = nn.Sequential(
                nn.Linear(cfg.gene_id_dim, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, d),
            )
        else:
            raise ValueError(f"unknown PertFormer variant: {cfg.variant!r}")

    # ------------------------------------------------------------------ q_pert
    def get_q_pert(self, pert_idx: torch.Tensor) -> torch.Tensor:
        """pert_idx: (B, k) padded gene-vocab indices (0 = pad). Returns (B, 1, d)."""
        if self.cfg.variant == "A":
            e = self.q_pert(pert_idx)                           # (B, k, d)
            mask = (pert_idx > 0).unsqueeze(-1).float()
            return (e * mask).sum(dim=1, keepdim=True)           # (B, 1, d)
        else:
            ids = self.gene_identity_table[pert_idx]             # (B, k, gene_id_dim)
            e = self.q_pert_mlp(ids)                              # (B, k, d)
            mask = (pert_idx > 0).unsqueeze(-1).float()
            return (e * mask).sum(dim=1, keepdim=True)            # (B, 1, d)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        control_expr: torch.Tensor,                # (B, n_genes)
        pert_idx: torch.Tensor,                     # (B, k)
        is_perturbed_mask: Optional[torch.Tensor] = None,   # (B, n_genes) long {0,1}
        q_override: Optional[torch.Tensor] = None,  # (B, 1, d) or (1, 1, d) -- audit ablations
    ) -> torch.Tensor:
        """Returns delta_pred: (B, n_genes), the predicted perturbation-induced
        change vs. control for each modeled gene."""
        B, G = control_expr.shape
        device = control_expr.device
        gene_ids = torch.arange(G, device=device).unsqueeze(0).expand(B, G)
        tok = self.gene_id_emb(gene_ids) + self.val_proj(control_expr.unsqueeze(-1))
        if is_perturbed_mask is None:
            is_perturbed_mask = torch.zeros(B, G, dtype=torch.long, device=device)
        tok = tok + self.flag_emb(is_perturbed_mask)

        q = q_override if q_override is not None else self.get_q_pert(pert_idx)
        if q.shape[0] == 1 and B > 1:
            q = q.expand(B, -1, -1)
        seq = torch.cat([q, tok], dim=1)              # (B, 1+G, d)
        out = self.encoder(seq)                        # (B, 1+G, d)
        gene_out = out[:, 1:, :]
        delta = self.head(gene_out).squeeze(-1)         # (B, G)
        return delta


def build_is_perturbed_mask(pert_idx: torch.Tensor, pos_lookup: torch.Tensor, n_genes: int) -> torch.Tensor:
    """pert_idx: (B, k) vocab indices. pos_lookup: (n_pert_genes+1,) long tensor
    mapping vocab-index -> position in the modeled gene panel, or -1 if the
    perturbed gene is not itself in the panel (common: most target genes are
    not HVGs). Returns (B, n_genes) long {0,1}."""
    B = pert_idx.shape[0]
    device = pert_idx.device
    mask = torch.zeros(B, n_genes, dtype=torch.long, device=device)
    positions = pos_lookup.to(device)[pert_idx]        # (B, k), -1 for "not in panel" / pad(0)->pos_lookup[0]
    for b in range(B):
        valid = positions[b][positions[b] >= 0]
        if len(valid) > 0:
            mask[b, valid] = 1
    return mask
