"""Stage-2 episodic meta-loss training (architecture.md §3.1).

The within-species pilot revealed that without Stage-2 the model
collapses to "ignore the per-identity token" (Q1 finding 2026-05-04).
Stage-2 fixes this by explicitly training the model to be amenable to
token-only adaptation:

    For each batch of B episodes:
      1. Pick a held-out training species s_h (sampled uniformly).
      2. Sample k support + q query examples from s_h.
      3. Inner loop:
            - Init token via interpolation head over k support
              (using the OTHER training species' tokens as the
              dictionary; s_h's token is excluded from the dictionary).
            - n_inner SGD steps on the k support examples, updating
              ONLY the token (decoder/FiLM/interp_head frozen during
              inner; outer loop unfreezes them).
      4. Outer loss: query-set coord-loss with the adapted token.
      5. First-order MAML: backprop the outer loss through the
         adapted-token expression (using the inner-loop SGD path).

This is the mechanism that the H1 paper hinges on. The Q1 negative
result above showed it cannot be skipped.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.iti import ITIModel

logger = logging.getLogger(__name__)


def _coord_loss(
    pred_unit: torch.Tensor, gt_xy: torch.Tensor, vis: torch.Tensor, out_size: int = 224,
) -> torch.Tensor:
    target = gt_xy / out_size
    per_kp = ((pred_unit - target) ** 2).sum(dim=-1)
    denom = vis.sum().clamp_min(1.0)
    return (per_kp * vis).sum() / denom


def episodic_step(
    model: ITIModel,
    held_out_local_id: int,
    support_features: torch.Tensor,    # (k, n_patches, 768)
    support_kp: torch.Tensor,          # (k, K, 2) in 224 px
    support_vis: torch.Tensor,         # (k, K)
    query_features: torch.Tensor,      # (q, n_patches, 768)
    query_kp: torch.Tensor,            # (q, K, 2)
    query_vis: torch.Tensor,           # (q, K)
    n_inner: int = 5,
    inner_lr: float = 1.0,
    out_size: int = 224,
    use_interpolation: bool = True,
) -> torch.Tensor:
    """One episode meta-step: returns the OUTER (query) loss as a tensor with grad flowing
    back through the inner-SGD adaptation path. Uses functional inner-step style so the
    adapted token retains computational dependence on (interp_head, support_features)."""

    n_train_ids = model.tokens.emb.weight.shape[0]
    other_ids = [i for i in range(n_train_ids) if i != held_out_local_id]
    other_tokens = model.tokens.emb.weight[other_ids]                  # (n-1, 768)

    interp_aux_loss = None
    if use_interpolation and model.interpolation_head is not None:
        ex_feat = support_features.mean(dim=1, keepdim=False).unsqueeze(0)  # (1, k, 768)
        e_init, _ = model.interpolation_head(ex_feat, other_tokens)
        token = e_init.squeeze(0)                                        # (768,)
        # Explicit interp-head supervision. The held-out species' OWN
        # trained token is the target for e_init. Forces the interp head
        # to actually use demo features to predict the held-out token
        # rather than collapse to the centroid.
        target_token = model.tokens.emb.weight[held_out_local_id].detach()
        interp_aux_loss = ((token - target_token) ** 2).mean()
    else:
        token = other_tokens.mean(dim=0)

    # Inner-loop SGD on support set, token-only.
    # FOMAML (first-order): we don't backprop through the inner-loop SGD steps
    # (PyTorch CPU SDP-attention lacks double-backward). The outer grad still
    # flows to (a) the interp_head via token_init, and (b) the decoder via
    # the outer pred_q forward pass — sufficient for meta-learning of
    # token-amenable representations.
    for _ in range(n_inner):
        B = support_features.shape[0]
        pred = model.forward_with_token(support_features, token.unsqueeze(0).expand(B, -1))
        loss_in = _coord_loss(pred, support_kp, support_vis, out_size)
        grad = torch.autograd.grad(loss_in, token, create_graph=False, retain_graph=False)[0]
        token = token - inner_lr * grad.detach()

    # Outer loss on query.
    Bq = query_features.shape[0]
    pred_q = model.forward_with_token(query_features, token.unsqueeze(0).expand(Bq, -1))
    loss_out = _coord_loss(pred_q, query_kp, query_vis, out_size)
    return loss_out, interp_aux_loss


def sample_episode_indices(
    species_to_indices: dict[int, np.ndarray],
    held_out_local_id: int,
    k_support: int,
    q_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pool = species_to_indices[held_out_local_id]
    if len(pool) < k_support + q_query:
        # Sample with replacement as a fallback; rare in practice
        idx = rng.choice(pool, size=k_support + q_query, replace=True)
    else:
        idx = rng.choice(pool, size=k_support + q_query, replace=False)
    return idx[:k_support], idx[k_support : k_support + q_query]


@dataclass
class EpisodicTrainConfig:
    n_steps: int = 400
    k_support: int = 5
    q_query: int = 5
    n_inner: int = 3
    inner_lr: float = 1.0
    outer_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 30
    log_every: int = 25
    out_size: int = 224
    seed: int = 0
    device: str = "cpu"
    use_interpolation: bool = True
