"""ITI: Identity-Token Interpolation, top-level model.

Compositional structure (architecture.md §0):

    image  -->  FrozenDINOv2Base    -->  patch_features (B, 256, 768)
                                              |
    species_id -->  IdentityTokenBank --> id_token (B, 768) ----+
                                                                |
                                          PoseDecoder (FiLM + transformer + deconv)
                                                                |
                                                                v
                                                    heatmaps (B, K, H, W)

For Year-1 Q1 we omit the LoRA adapter on the decoder (the architecture
spec defaults rank-r LoRA on a subset of decoder weights — for our small
~4M decoder, full training of the decoder is comparable in trainable-
param count to a rank-8 LoRA on a larger decoder; we revisit when the
decoder grows in Year-1 Q2).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .frozen_backbone import FrozenDINOv2Base, DINOV2_BASE_GRID, DINOV2_BASE_HIDDEN
from .identity_token import IdentityTokenBank, NovelIdentityToken
from .pose_decoder import PoseDecoder
from .interpolation_head import InterpolationHead, InterpolationHeadV2


@dataclass
class ITIConfig:
    n_training_identities: int = 50
    hidden: int = DINOV2_BASE_HIDDEN
    grid: int = DINOV2_BASE_GRID
    n_keypoints: int = 17
    decoder_layers: int = 2
    decoder_dim: int = 384
    heatmap_size: int = 64
    use_interpolation_head: bool = True
    interp_head_version: str = "v1"   # "v1" (mean-pool cosine) or "v2" (per-demo cross-attn)


class ITIModel(nn.Module):
    """Trainable parts: identity tokens, decoder, (optional) interpolation head.

    Frozen: backbone (DINOv2-base).
    """

    def __init__(self, cfg: ITIConfig, backbone: Optional[FrozenDINOv2Base] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone or FrozenDINOv2Base()
        self.tokens = IdentityTokenBank(cfg.n_training_identities, cfg.hidden)
        self.decoder = PoseDecoder(
            in_dim=cfg.hidden, n_keypoints=cfg.n_keypoints, grid=cfg.grid,
            n_layers=cfg.decoder_layers, decoder_dim=cfg.decoder_dim,
            heatmap_size=cfg.heatmap_size,
        )
        if cfg.use_interpolation_head:
            if cfg.interp_head_version == "v2":
                self.interpolation_head = InterpolationHeadV2(hidden=cfg.hidden)
            else:
                self.interpolation_head = InterpolationHead(hidden=cfg.hidden)
        else:
            self.interpolation_head = None

    # ------------------------------------------------------------------
    # Forward — pretraining/eval with backbone

    def forward_from_pixels(
        self,
        pixel_values: torch.Tensor,        # (B, 3, H, W)
        identity_ids: torch.Tensor,        # (B,) long
    ) -> torch.Tensor:
        patches = self.backbone.patch_features(pixel_values)
        return self.forward_from_features(patches, identity_ids)

    def forward_from_features(
        self,
        patch_features: torch.Tensor,      # (B, N_patches, hidden)
        identity_ids: torch.Tensor,        # (B,) long
    ) -> torch.Tensor:
        id_tokens = self.tokens(identity_ids)
        return self.decoder(patch_features, id_tokens)

    def forward_with_token(
        self,
        patch_features: torch.Tensor,      # (B, N_patches, hidden)
        id_token: torch.Tensor,            # (B, hidden)
    ) -> torch.Tensor:
        """Use a supplied id_token (e.g. a novel identity's adapted token)."""
        return self.decoder(patch_features, id_token)

    # ------------------------------------------------------------------

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_decoder_params(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters())
