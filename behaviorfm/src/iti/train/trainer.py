"""Within-species ITI training on cached DINOv2 features.

Architecture.md §3 Stage 1 (joint-task pretraining), simplified for
the Year-1 Q1 pilot to feature-cached training. The decoder + identity
tokens + interpolation head are jointly trained; the backbone is
frozen and not invoked.

Loss = MSE on Gaussian heatmaps over visible keypoints (architecture.md §3.3).
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.feature_cache import CachedFeatureDataset, collate_cached
from ..eval.pck import per_keypoint_distance_normalized, pck_metrics, bootstrap_ci
from ..model.iti import ITIModel, ITIConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    n_steps: int = 800
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    log_every: int = 50
    eval_every: int = 200
    seed: int = 0
    out_size: int = 224
    device: str = "cpu"
    # Token-utility auxiliary loss (paper §6 item 4 — fires after Q1 finding F-Q1-8
    # showed the model architecturally CAN use the id-token but training does not
    # incentivize it. The aux loss makes the id-token informationally necessary.)
    token_utility_weight: float = 0.0    # 0 = off; 0.1 = paper's default
    token_utility_margin: float = 0.0    # min margin: loss(wrong_token) - loss(right_token)


def coord_loss(
    pred_unit: torch.Tensor,        # (B, K, 2) in [0,1]
    gt_xy_pixels: torch.Tensor,     # (B, K, 2) in pixel coords (out_size)
    vis: torch.Tensor,              # (B, K)
    out_size: int = 224,
) -> torch.Tensor:
    target = gt_xy_pixels / out_size                      # to [0,1]
    per_kp = ((pred_unit - target) ** 2).sum(dim=-1)      # (B, K) — squared L2 in unit space
    denom = vis.sum().clamp_min(1.0)
    return (per_kp * vis).sum() / denom


def token_utility_loss(
    model,
    feat: torch.Tensor,             # (B, N_patches, 768)
    gt_xy: torch.Tensor,            # (B, K, 2) in 224-px
    vis: torch.Tensor,              # (B, K)
    correct_ids: torch.Tensor,      # (B,) long
    margin: float = 0.0,
    out_size: int = 224,
) -> torch.Tensor:
    """Margin loss: pose error with correct id-token must be strictly less
    than with a random WRONG id-token (uniformly sampled from the other
    training species). Specifically: per-example,
        L_aux = max(0, loss_right - loss_wrong + margin)
    Which is positive (penalty) iff the wrong token is no worse than the
    right one. Drives the model to make pose error sensitive to id-token.
    """
    B = feat.shape[0]
    N_ids = model.tokens.emb.weight.shape[0]
    if N_ids < 2:
        return feat.new_zeros(())
    # Pick a wrong-id-per-example uniformly from {ids != correct_id}
    rand_ids = torch.randint(0, N_ids, (B,), device=correct_ids.device)
    same = rand_ids == correct_ids
    if same.any():
        # bump same ones
        rand_ids = torch.where(same, (rand_ids + 1) % N_ids, rand_ids)
    right_tok = model.tokens(correct_ids)
    wrong_tok = model.tokens(rand_ids)
    # Forward pass twice (one with right, one with wrong)
    pred_r = model.forward_with_token(feat, right_tok)
    pred_w = model.forward_with_token(feat, wrong_tok)
    target = gt_xy / out_size
    per_kp_r = ((pred_r - target) ** 2).sum(dim=-1)        # (B, K)
    per_kp_w = ((pred_w - target) ** 2).sum(dim=-1)
    # per-example mean masked by vis
    mask = vis
    per_ex_r = (per_kp_r * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    per_ex_w = (per_kp_w * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
    margin_violations = torch.relu(per_ex_r - per_ex_w + margin)
    return margin_violations.mean()


def evaluate(
    model: ITIModel,
    loader: DataLoader,
    cfg: TrainConfig,
) -> dict[str, float]:
    model.eval()
    dists, diags = [], []
    with torch.no_grad():
        for batch in loader:
            feat = batch["feature"].to(cfg.device)
            ids = batch["identity_id"].to(cfg.device)
            kp_gt = batch["keypoints_xy"]               # (B, K, 2) in 224 px
            vis = batch["vis"]                          # (B, K)
            pred_unit = model.forward_from_features(feat, ids).cpu()
            pred_xy = pred_unit * cfg.out_size           # back to crop pixels
            d = per_keypoint_distance_normalized(
                pred_xy.numpy(), kp_gt.numpy(), vis.numpy(),
                batch["crop_side"].numpy(), out_size=cfg.out_size,
            )
            dists.append(d)
            diags.append(batch["bbox_diag_orig"].numpy())
    dists = np.concatenate(dists, axis=0)
    diags = np.concatenate(diags, axis=0)
    metrics = pck_metrics(dists, diags)
    # bootstrap CI on rmse_norm
    flat = (dists / diags[:, None]).flatten()
    flat = flat[~np.isnan(flat)]
    if flat.size:
        # rmse on per-keypoint distances
        sq = flat ** 2
        m, lo, hi = bootstrap_ci(sq)
        metrics["rmse_norm_mean"] = float(np.sqrt(m))
        metrics["rmse_norm_ci95_lo"] = float(np.sqrt(lo))
        metrics["rmse_norm_ci95_hi"] = float(np.sqrt(hi))
    return metrics


def train_within_species(
    model: ITIModel,
    train_ds: CachedFeatureDataset,
    val_ds: CachedFeatureDataset,
    cfg: TrainConfig,
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = model.to(cfg.device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info("Trainable params: %d", n_trainable)
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        # cosine decay to 0.1× over remaining steps
        progress = (step - cfg.warmup_steps) / max(cfg.n_steps - cfg.warmup_steps, 1)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_cached, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_cached,
    )

    history = []
    best_val = {"rmse_norm": float("inf")}
    step = 0
    t0 = time.time()
    while step < cfg.n_steps:
        for batch in train_loader:
            if step >= cfg.n_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            feat = batch["feature"].to(cfg.device)
            ids = batch["identity_id"].to(cfg.device)
            kp = batch["keypoints_xy"].to(cfg.device)
            vis = batch["vis"].to(cfg.device)
            model.train()
            pred = model.forward_from_features(feat, ids)
            loss_main = coord_loss(pred, kp, vis, out_size=cfg.out_size)
            if cfg.token_utility_weight > 0.0:
                loss_aux = token_utility_loss(
                    model, feat, kp, vis, ids,
                    margin=cfg.token_utility_margin, out_size=cfg.out_size,
                )
                loss = loss_main + cfg.token_utility_weight * loss_aux
            else:
                loss_aux = torch.zeros((), device=loss_main.device)
                loss = loss_main
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()

            if step % cfg.log_every == 0:
                logger.info(
                    "step %4d/%d  loss=%.4e  (main=%.4e  aux=%.4e)  lr=%.2e  (%.1fs)",
                    step, cfg.n_steps, loss.item(), loss_main.item(),
                    loss_aux.item(), lr_at(step), time.time() - t0,
                )
            if step > 0 and step % cfg.eval_every == 0:
                m = evaluate(model, val_loader, cfg)
                m["step"] = step
                logger.info("EVAL step %d: %s", step, m)
                history.append(m)
                if m["rmse_norm"] < best_val["rmse_norm"]:
                    best_val = m
            step += 1

    final = evaluate(model, val_loader, cfg)
    final["step"] = step
    history.append(final)
    return {
        "history": history,
        "final": final,
        "best": best_val if best_val["rmse_norm"] < final["rmse_norm"] else final,
        "n_trainable_params": n_trainable,
    }
