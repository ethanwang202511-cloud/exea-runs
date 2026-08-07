"""PCK (Percentage of Correct Keypoints) and pose RMSE metrics.

PCK@α: keypoint counted as correct if its predicted location is
within α * normalizing_distance of ground truth. AP-10K convention:
normalize by bbox diagonal in the original (un-cropped) image.

We report:
    PCK@0.05  (tight threshold)
    PCK@0.10  (looser threshold; reported in many pose papers)
    RMSE_normalized  (in units of bbox diagonal — comparable across crops)
"""

from __future__ import annotations
import numpy as np


def per_keypoint_distance_normalized(
    pred_xy_in_crop: np.ndarray,    # (B, K, 2) in crop pixel coords
    gt_xy_in_crop: np.ndarray,      # (B, K, 2)
    vis: np.ndarray,                # (B, K)
    crop_side: np.ndarray,          # (B,)  side length of square crop in original-image px
    out_size: int = 224,
) -> np.ndarray:
    """Returns (B, K) per-keypoint pixel distance, in original-image pixel units.

    Invisible keypoints get NaN so callers can mask them out.
    """
    scale = crop_side / out_size                                 # (B,)
    diff = pred_xy_in_crop - gt_xy_in_crop                       # (B, K, 2)
    dist = np.linalg.norm(diff, axis=-1)                         # (B, K) in crop px
    dist_orig = dist * scale[:, None]                            # in original px
    dist_orig = np.where(vis > 0, dist_orig, np.nan)
    return dist_orig


def pck_metrics(
    distances_orig_px: np.ndarray,   # (B, K) in original px (NaN if invisible)
    bbox_diag_orig: np.ndarray,      # (B,)
    thresholds: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    """Aggregate PCK at several thresholds + RMSE normalized to bbox diagonal."""
    norm = bbox_diag_orig[:, None]                               # (B, 1)
    dist_norm = distances_orig_px / norm                         # (B, K) normalized
    out: dict[str, float] = {}
    flat = dist_norm[~np.isnan(dist_norm)]
    for a in thresholds:
        out[f"pck@{a:.2f}"] = float((flat < a).mean()) if flat.size else float("nan")
    out["rmse_norm"] = float(np.sqrt((flat ** 2).mean())) if flat.size else float("nan")
    out["mae_norm"] = float(flat.mean()) if flat.size else float("nan")
    out["n_keypoints_evaluated"] = int(flat.size)
    return out


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = 1000, ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo = float(np.quantile(means, (1 - ci) / 2))
    hi = float(np.quantile(means, 1 - (1 - ci) / 2))
    return float(values.mean()), lo, hi
