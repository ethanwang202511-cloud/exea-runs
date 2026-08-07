"""Per-backbone Wasserstein distance between an identity's example demos
and the convex hull of training-identity tokens / patch features.

Operationalizes the H0 metric of proposal §4. Three implementations
are provided so we can scout dynamic range cheaply:

  (A) "token-hull-mean":  distance from a per-instance feature vector
      (CLS-style: mean over patches) to the convex hull of training
      species' mean feature vectors. We use the L2 distance to the
      closest convex combination (a small QP via scipy linprog/cvxopt
      would be exact; here we use the simpler "distance to the closest
      training species mean" plus a "distance to the centroid of all
      training species means").

  (B) "patch-wasserstein":  proper 2-Wasserstein distance between the
      empirical patch-token distribution of a held-out instance (256
      points in R^768) and the empirical distribution of patch tokens
      from a single training-species instance (also 256 points). Using
      POT's Sinkhorn for tractability. This is the "per-backbone
      Wasserstein in feature space" that proposal §H0 refers to in its
      strict reading.

  (C) "instance-set-wasserstein":  Sinkhorn 2-Wasserstein between the
      pooled feature distribution of K examples of a held-out identity
      (K x 768) and the pooled feature distribution of M examples per
      training identity (M x 768). This is closer to the H0 phase
      diagram operational definition.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a @ b.T


def euclid_dist_to_set(x: np.ndarray, anchors: np.ndarray) -> tuple[float, float, float]:
    """L2 distances from x (D,) to each anchor row (N, D). Returns (min, mean, max)."""
    diff = anchors - x[None, :]
    d = np.linalg.norm(diff, axis=-1)
    return float(d.min()), float(d.mean()), float(d.max())


def sinkhorn_2wass(
    X: np.ndarray,                  # (n, d) source samples
    Y: np.ndarray,                  # (m, d) target samples
    reg_frac: float = 0.05,
    n_iter: int = 200,
) -> float:
    """Sinkhorn 2-Wasserstein distance.

    reg_frac picks the entropic regularization as a fraction of the median
    cost-matrix value, which avoids the numerical-underflow trap when costs
    are large (the failure mode observed on raw 768-d DINOv2 features).
    Returns NaN if POT not installed.
    """
    try:
        import ot
    except ImportError:
        logger.warning("POT not installed; Sinkhorn unavailable.")
        return float("nan")
    a = np.ones(len(X)) / len(X)
    b = np.ones(len(Y)) / len(Y)
    XX = np.sum(X ** 2, axis=-1, keepdims=True)
    YY = np.sum(Y ** 2, axis=-1, keepdims=True).T
    M = XX + YY - 2 * (X @ Y.T)
    M = np.clip(M, 0.0, None)
    reg = float(np.median(M)) * reg_frac
    if reg <= 0.0:
        return 0.0
    val = ot.sinkhorn2(a, b, M, reg=reg, numItermax=n_iter, log=False, stopThr=1e-7)
    return float(np.sqrt(max(val, 0.0)))


def aggregate_per_species(
    features_NxNpxD: np.ndarray,    # (N_examples, N_patches, D)
    species_id_per_example: np.ndarray,
) -> dict[int, np.ndarray]:
    """Returns species_id -> (n_examples_for_species, N_patches, D) view."""
    out: dict[int, np.ndarray] = {}
    for sid in np.unique(species_id_per_example):
        mask = species_id_per_example == sid
        out[int(sid)] = features_NxNpxD[mask]
    return out


@dataclass
class WasserstainScout:
    """Run the dynamic-range scout for the H0 distance metric on a held-out
    species vs a pool of training species. Returns a small report dict."""

    train_features_per_species: dict[int, np.ndarray]   # sid -> (n, n_patches, D)
    n_examples_per_pool: int = 32                        # M examples per training species

    def __post_init__(self) -> None:
        # Build per-species "centroid feature" (mean over examples, mean over patches)
        self.train_means = {
            sid: arr.reshape(-1, arr.shape[-1]).mean(axis=0)
            for sid, arr in self.train_features_per_species.items()
        }
        self.training_centroid = np.stack(list(self.train_means.values())).mean(axis=0)
        # Pre-pool: M random patches per training species for distribution comparisons
        rng = np.random.default_rng(0)
        self.pooled_per_species = {}
        for sid, arr in self.train_features_per_species.items():
            flat = arr.reshape(-1, arr.shape[-1])
            sample = rng.choice(len(flat),
                                size=min(self.n_examples_per_pool, len(flat)),
                                replace=False)
            self.pooled_per_species[sid] = flat[sample]

    def distance_to_hull_mean(self, novel_demo_features: np.ndarray) -> dict:
        """novel_demo_features: (n_demos, N_patches, D). Returns dict of distance metrics."""
        flat = novel_demo_features.reshape(-1, novel_demo_features.shape[-1])
        novel_mean = flat.mean(axis=0)                   # (D,)
        # distances to training species means
        anchors = np.stack(list(self.train_means.values()))    # (n_species, D)
        d_min, d_mean, d_max = euclid_dist_to_set(novel_mean, anchors)
        d_centroid = float(np.linalg.norm(novel_mean - self.training_centroid))
        # cosine sim to closest training species
        c = cosine_similarity(novel_mean[None, :], anchors).flatten()
        return {
            "euclid_d_min_to_train_species_mean": d_min,
            "euclid_d_mean_to_train_species_mean": d_mean,
            "euclid_d_max_to_train_species_mean": d_max,
            "euclid_d_to_training_centroid": d_centroid,
            "cos_sim_max_to_train_species_mean": float(c.max()),
            "cos_sim_mean_to_train_species_mean": float(c.mean()),
        }

    def sinkhorn_to_each_train_species(
        self, novel_demo_features: np.ndarray, max_pts: int = 32, reg_frac: float = 0.05,
    ) -> dict:
        """Sinkhorn Wasserstein from novel demos (pooled patches) to each training species' pooled patches."""
        flat = novel_demo_features.reshape(-1, novel_demo_features.shape[-1])
        rng = np.random.default_rng(1)
        if len(flat) > max_pts:
            sample = rng.choice(len(flat), size=max_pts, replace=False)
            flat = flat[sample]
        out: dict[int, float] = {}
        for sid, pool in self.pooled_per_species.items():
            out[sid] = sinkhorn_2wass(flat, pool, reg_frac=reg_frac)
        ws = list(out.values())
        return {
            "sinkhorn_w2_per_train_species": out,
            "sinkhorn_w2_min": float(np.min(ws)) if ws else float("nan"),
            "sinkhorn_w2_mean": float(np.mean(ws)) if ws else float("nan"),
            "sinkhorn_w2_max": float(np.max(ws)) if ws else float("nan"),
        }
