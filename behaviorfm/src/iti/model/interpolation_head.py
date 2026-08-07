"""Learned interpolation head — the 'I' in ITI.

Given k example demonstrations of a novel identity (their backbone
features), produce interpolation weights alpha_i over the N training
identity tokens such that

    e_new_init = sum_i alpha_i * e_id_i

The head is trained jointly with the model during pretraining
(episodic meta-loss, Stage 2).

Two implementations:
  - InterpolationHead (v1): mean-pool over k demos, then cosine-softmax
    over training-id tokens. Empirically degenerate (paper §5.11
    Finding F-Q6-2): produces uniform alphas at any k; head ignores
    demos. KEPT for backward-compat / ablation.

  - InterpolationHeadV2: per-demo cross-attention to training-id
    tokens, then aggregate per-demo logits. Each demo can independently
    "vote" for which training species it most resembles; the resulting
    aggregated logits are softmax'd to produce alphas. ~5x more
    parameters but should capture per-demo evidence.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolationHead(nn.Module):
    """Similarity-based interpolation weights (v1 — empirically degenerate).

    Given mean-pooled backbone features of k example demos, project
    into the token embedding space and compute softmax similarities
    to each training-identity token. Returns the convex combination.

    ~50k params for hidden=768 (a single linear projection).
    """

    def __init__(self, hidden: int = 768, temperature: float = 1.0) -> None:
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature
        self.proj = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(
        self,
        example_features: torch.Tensor,    # (B, k, F) backbone features pooled per demo
        training_id_tokens: torch.Tensor,  # (N, F)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (e_new_init (B, F), alphas (B, N))."""
        query = example_features.mean(dim=1)                # (B, F)
        query = self.proj(query)                            # (B, F)
        # cosine similarity
        q = F.normalize(query, dim=-1)                      # (B, F)
        k = F.normalize(training_id_tokens, dim=-1)         # (N, F)
        logits = q @ k.t() / self.temperature               # (B, N)
        alphas = F.softmax(logits, dim=-1)                  # (B, N)
        e_new_init = alphas @ training_id_tokens            # (B, F)
        return e_new_init, alphas


class InterpolationHeadV2(nn.Module):
    """Per-demo attention-based interpolation head (v2).

    Given (B, k, F) demo features and (N, F) training-id tokens:
      1. Project each demo independently to query space (per-demo Q).
      2. Project training-id tokens to key/value space.
      3. For each demo, compute cross-attention scores to training tokens.
         Each demo "votes" for the closest training species.
      4. Aggregate per-demo logits by mean (a permutation-invariant pool
         that, unlike feature-mean-pool, preserves per-demo evidence
         because the projection happens BEFORE pooling).
      5. softmax over training tokens -> alphas; mix to produce e_new.

    Crucially, when demos disagree (which they should, given variation
    within a species), the per-demo voting gives a sharper aggregate
    than mean-pooling features and then projecting once.

    Parameter count: 3 * hidden * hidden ~ 1.77M for hidden=768.
    """

    def __init__(self, hidden: int = 768, temperature: float = 1.0,
                 head_dim: int | None = None) -> None:
        super().__init__()
        self.hidden = hidden
        self.temperature = temperature
        self.head_dim = head_dim or hidden
        self.q_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        # initialize so init alphas are near-uniform
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        # v_proj initialized to identity so the output is approximately
        # the value-projected mean of training tokens at init -> matches v1 behavior at init
        nn.init.eye_(self.v_proj.weight)

    def forward(
        self,
        example_features: torch.Tensor,    # (B, k, F)
        training_id_tokens: torch.Tensor,  # (N, F)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, k, F_ = example_features.shape
        N = training_id_tokens.shape[0]
        q = self.q_proj(example_features)                       # (B, k, head_dim)
        kk = self.k_proj(training_id_tokens)                    # (N, head_dim)
        v = self.v_proj(training_id_tokens)                     # (N, hidden)
        # per-demo logits to each training token
        scale = self.head_dim ** 0.5
        logits = (q @ kk.t()) / (scale * self.temperature)      # (B, k, N)
        # aggregate over demos (mean of per-demo logits — equivalent to log-sum if we use logsumexp,
        # but mean is gentler and avoids overcommitting to one demo)
        agg_logits = logits.mean(dim=1)                          # (B, N)
        alphas = torch.softmax(agg_logits, dim=-1)              # (B, N)
        e_new_init = alphas @ v                                  # (B, hidden)
        return e_new_init, alphas
