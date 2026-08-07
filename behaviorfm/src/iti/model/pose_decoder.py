"""Identity-conditioned pose decoder.

Year-1 Q1 finding F-Q1-4 / F-Q1-8: the v1.0 FiLM-only identity
injection is structurally too weak — token-only adaptation produced
null signal on both giraffe (cos 0.72) and elephant (cos 0.50).

Year-1 Q2 (this file): replace FiLM with **cross-attention identity
injection**. The per-id token is included as an additional context
token alongside the 256 patch tokens, so each per-keypoint query
cross-attends to (256 patches + 1 id_token). The id_token thus has
a direct, expressive path to per-keypoint prediction via attention
weights, rather than the narrow γ/β multiplicative-additive path
of FiLM.

Optional: also keep a small FiLM modulation as an auxiliary path
(controlled by the `keep_film` flag), so we can ablate the two
mechanisms.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Per-identity feature-wise modulation conditioned on the id token."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, 2 * hidden)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feats: torch.Tensor, id_token: torch.Tensor) -> torch.Tensor:
        gb = self.proj(id_token)
        gamma, beta = gb.chunk(2, dim=-1)
        return feats * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class CrossAttnBlock(nn.Module):
    """One round of (queries) attend (context). Pre-norm transformer-decoder block."""

    def __init__(self, dim: int, heads: int = 6) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        out, _ = self.attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + out
        q = q + self.mlp(self.norm2(q))
        return q


class PoseDecoder(nn.Module):
    """Identity-conditioned coord-regression decoder (Q2: cross-attention id-injection).

    Pipeline:
        patch_features (B, N, 768)              # frozen DINOv2
        id_token       (B,    768)              # learned per-identity

        [optional FiLM]: patches' = FiLM(patches, id_token)
        in_proj:        x = Linear(patches')   (B, N, decoder_dim)  + pos_emb
        encoder:        ctx = TransformerEncoder(x)                  (B, N, decoder_dim)
        id_proj:        id_ctx = Linear(id_token)                    (B, 1, decoder_dim)
        ctx_with_id:    [ctx; id_ctx]                                (B, N+1, decoder_dim)
        K queries cross-attend to ctx_with_id:
            q (B, K, decoder_dim) = CrossAttn(kp_queries, ctx_with_id)
        coord head:     (B, K, 2) in [0,1]
    """

    def __init__(
        self,
        in_dim: int = 768,
        n_keypoints: int = 17,
        grid: int = 16,
        n_layers: int = 2,
        decoder_dim: int = 384,
        heatmap_size: int = 64,                     # kept for API compat (unused)
        n_cross_attn_layers: int = 2,
        keep_film: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_keypoints = n_keypoints
        self.grid = grid
        self.heatmap_size = heatmap_size
        self.keep_film = keep_film

        if keep_film:
            self.film = FiLM(in_dim)
        else:
            self.film = None

        # Project patch features to decoder dim and add 2D positional embedding.
        self.in_proj = nn.Linear(in_dim, decoder_dim)
        self.pos_emb = nn.Parameter(torch.zeros(grid * grid, decoder_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=6, dim_feedforward=4 * decoder_dim,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Project id_token into decoder space; this is the new injection path.
        self.id_proj = nn.Linear(in_dim, decoder_dim)
        # Distinct positional offset for the id_token slot vs patch positions.
        self.id_pos = nn.Parameter(torch.zeros(1, decoder_dim))
        nn.init.normal_(self.id_pos, std=0.02)

        # Per-keypoint learned queries.
        self.kp_queries = nn.Parameter(torch.zeros(n_keypoints, decoder_dim))
        nn.init.normal_(self.kp_queries, std=0.02)

        # Stack of cross-attention blocks (queries attend to [ctx; id_ctx]).
        self.cross_attn_blocks = nn.ModuleList(
            [CrossAttnBlock(decoder_dim, heads=6) for _ in range(n_cross_attn_layers)]
        )

        # Coord head: per-keypoint MLP projecting query -> (x, y).
        self.coord_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 2),
        )

    def forward(
        self,
        patch_features: torch.Tensor,    # (B, N=grid*grid, in_dim)
        id_token: torch.Tensor,          # (B, in_dim)
    ) -> torch.Tensor:
        """Returns coords (B, K, 2) in [0, 1]."""
        B, N, _ = patch_features.shape
        assert N == self.grid * self.grid
        x = patch_features
        if self.film is not None:
            x = self.film(x, id_token)
        x = self.in_proj(x) + self.pos_emb.unsqueeze(0)            # (B, N, D)
        ctx = self.encoder(x)                                       # (B, N, D)

        id_ctx = self.id_proj(id_token).unsqueeze(1) + self.id_pos.unsqueeze(0)   # (B, 1, D)
        ctx_with_id = torch.cat([ctx, id_ctx], dim=1)              # (B, N+1, D)

        q = self.kp_queries.unsqueeze(0).expand(B, -1, -1)         # (B, K, D)
        for block in self.cross_attn_blocks:
            q = block(q, ctx_with_id)
        coords = self.coord_head(q)                                # (B, K, 2)
        return torch.sigmoid(coords)
