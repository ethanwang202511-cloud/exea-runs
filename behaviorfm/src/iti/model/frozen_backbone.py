"""Frozen DINOv2-base wrapper.

Implements the L2-frozen-backbone commitment from architecture.md §2.1:
the backbone is NEVER updated. We use facebook/dinov2-base from HF
transformers (~85M params, 768-d embeddings, 14x14 patch size).

For a 224x224 input the backbone returns 1 CLS + (224/14)^2 = 256
patch tokens of dim 768.
"""

from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DINOV2_BASE_HF_ID = "facebook/dinov2-base"
DINOV2_BASE_HIDDEN = 768
DINOV2_BASE_PATCH = 14
DINOV2_BASE_INPUT = 224           # default eval resolution
DINOV2_BASE_GRID = DINOV2_BASE_INPUT // DINOV2_BASE_PATCH   # 16


class FrozenDINOv2Base(nn.Module):
    """Wraps HF DINOv2-base; output shape: (B, 1+256, 768).

    The backbone is set to eval() and all parameters have
    requires_grad=False. Forward pass is wrapped in torch.no_grad()
    by default — call with grad=True only if you have a reason
    (you don't, per architecture commitment).
    """

    def __init__(self, hf_id: str = DINOV2_BASE_HF_ID, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        from transformers import AutoModel  # lazy import
        self.hf_id = hf_id
        self.model = AutoModel.from_pretrained(hf_id, torch_dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = DINOV2_BASE_HIDDEN
        self.patch_size = DINOV2_BASE_PATCH
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("FrozenDINOv2Base(%s): %d params (frozen)", hf_id, n_params)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (B, 3, H, W); returns (B, 1+N_patches, 768)."""
        out = self.model(pixel_values=pixel_values).last_hidden_state
        return out

    @torch.no_grad()
    def patch_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Drop CLS, return only patch tokens (B, N_patches, 768)."""
        return self.forward(pixel_values)[:, 1:]
