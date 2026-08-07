"""Per-identity learnable tokens.

Architecture.md §2.2: one learned token e_id ∈ R^768 per training
identity. At adaptation time on a novel identity: ONLY a new
e_new ∈ R^768 token is fit (~768 trainable params).

We expose two interfaces:
  - IdentityTokenBank: standard nn.Embedding lookup over a fixed
    set of training identities.
  - NovelIdentityToken: a single nn.Parameter for the held-out identity
    (the H1 mechanism's adaptation surface).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class IdentityTokenBank(nn.Module):
    """Fixed-size lookup table of per-identity tokens."""

    def __init__(self, n_identities: int, hidden: int = 768) -> None:
        super().__init__()
        self.n_identities = n_identities
        self.hidden = hidden
        self.emb = nn.Embedding(n_identities, hidden)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, identity_ids: torch.Tensor) -> torch.Tensor:
        """identity_ids: (B,) long; returns (B, hidden)."""
        return self.emb(identity_ids)

    @property
    def n_params(self) -> int:
        return self.n_identities * self.hidden


class NovelIdentityToken(nn.Module):
    """Single learnable token for a novel identity (H1 adaptation surface)."""

    def __init__(self, init: torch.Tensor | None = None, hidden: int = 768) -> None:
        super().__init__()
        if init is None:
            init = torch.zeros(hidden)
            nn.init.normal_(init, mean=0.0, std=0.02)
        assert init.shape == (hidden,), init.shape
        self.token = nn.Parameter(init.clone())

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.token.unsqueeze(0).expand(batch_size, -1)

    @property
    def n_params(self) -> int:
        return self.token.numel()
