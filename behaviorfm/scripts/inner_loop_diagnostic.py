"""Diagnose why 50 SGD inner steps on the per-id token produce zero RMSE change.

Hypothesis A: gradient on the token is too small (cross-attention's id_ctx
              has tiny attention weight relative to 256 patches).
Hypothesis B: inner_lr is too small for the loss landscape's gradient scale.
Hypothesis C: the loss landscape is FLAT in token-direction (= the id-token
              has no measurable effect on output, regardless of value).
Hypothesis D: 50 SGD steps converges to a token nearby init, but that nearby
              token doesn't actually move predictions.

We test by:
  1. Logging the per-step token L2 norm change and inner support loss.
  2. Sweeping inner_lr in {1e-3, 1e-2, 1e-1, 1.0, 10.0}.
  3. Reporting whether the model output (val pred) changes between
     init and post-adapt.
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset, collate_cached
from iti.eval.pck import per_keypoint_distance_normalized, pck_metrics
from iti.model.iti import ITIModel, ITIConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("inner_loop_diagnostic")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "results/q2_episodic/model.pt")
    p.add_argument("--train-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_train_split1_elephant")
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all")
    p.add_argument("--held-out-species", type=str, default="elephant")
    p.add_argument("--device", type=str, default="mps")
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                       if k in ITIConfig.__dataclass_fields__})
    model = ITIModel(cfg).to(args.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    train_ds = CachedFeatureDataset(args.train_cache)
    val_ds = CachedFeatureDataset(args.val_cache)
    name_to_gid = {r["species_name"]: r["identity_id"] for r in train_ds.records}
    held_gid = name_to_gid[args.held_out_species]
    train_held = CachedFeatureDataset(args.train_cache, identity_subset={held_gid})
    val_held = CachedFeatureDataset(args.val_cache, identity_subset={held_gid})

    train_loader = DataLoader(train_held, batch_size=len(train_held), shuffle=False,
                              collate_fn=collate_cached)
    train_batch = next(iter(train_loader))
    val_loader = DataLoader(val_held, batch_size=len(val_held), shuffle=False,
                            collate_fn=collate_cached)
    val_batch = next(iter(val_loader))

    rng = np.random.default_rng(0)
    k = 5
    sup_idx = rng.choice(len(train_held), size=k, replace=False)
    sup_f = train_batch["feature"][sup_idx].to(args.device)
    sup_kp = train_batch["keypoints_xy"][sup_idx].to(args.device)
    sup_vis = train_batch["vis"][sup_idx].to(args.device)

    # Inits to test
    id_bank = model.tokens.emb.weight.detach()
    mean_tok = id_bank.mean(dim=0)
    rand_tok = torch.randn_like(mean_tok) * 0.02

    # Output sensitivity probe — is the model output even sensitive to id_token?
    val_f = val_batch["feature"].to(args.device)
    val_kp = val_batch["keypoints_xy"].numpy()
    val_vis = val_batch["vis"].numpy()
    val_side = val_batch["crop_side"].numpy()
    val_diag = val_batch["bbox_diag_orig"].numpy()
    Bv = val_f.shape[0]

    def eval_with(token: torch.Tensor) -> tuple[float, np.ndarray]:
        with torch.no_grad():
            pred = model.forward_with_token(val_f, token.unsqueeze(0).expand(Bv, -1)).cpu()
        d = per_keypoint_distance_normalized(
            (pred * 224).numpy(), val_kp, val_vis, val_side, out_size=224,
        )
        m = pck_metrics(d, val_diag)
        return m["rmse_norm"], pred.numpy()

    # 1) Sensitivity: per-coord std of pred across many random tokens
    rmse_init = []
    preds_per_token = []
    for _ in range(8):
        tok = torch.randn(768, device=args.device) * 0.02
        r, p_ = eval_with(tok)
        rmse_init.append(r)
        preds_per_token.append(p_)
    preds_per_token = np.stack(preds_per_token)   # (8, Bv, K, 2)
    per_pt_std = preds_per_token.std(axis=0).mean()    # avg per-(B,K,2) std across tokens
    logger.info("Output sensitivity to token: mean per-coord std across 8 random tokens = %.5f (in [0,1] units)",
                per_pt_std)
    logger.info("RMSE_norm spread across 8 random tokens: min=%.4f mean=%.4f max=%.4f",
                min(rmse_init), float(np.mean(rmse_init)), max(rmse_init))

    # 2) Sweep inner_lr; report token movement and support+val loss trajectory
    for lr in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        tok = nn.Parameter(rand_tok.clone())
        opt = torch.optim.SGD([tok], lr=lr, momentum=0.9)
        movements = []
        sup_losses = []
        val_rmses = []
        prev = tok.detach().clone()
        for step in range(50):
            B = sup_f.shape[0]
            pred = model.forward_with_token(sup_f, tok.unsqueeze(0).expand(B, -1))
            target = sup_kp / 224
            per_kp = ((pred - target) ** 2).sum(dim=-1)
            loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            grad_norm = tok.grad.norm().item() if tok.grad is not None else 0.0
            opt.step()
            move = (tok.detach() - prev).norm().item()
            movements.append(move)
            sup_losses.append(loss.item())
            prev = tok.detach().clone()
            if step in (0, 10, 25, 49):
                rv, _ = eval_with(tok.detach())
                val_rmses.append((step, rv))
        rv_final, _ = eval_with(tok.detach())
        logger.info(
            "lr=%-6g  sup_loss[0,49]=[%.4e, %.4e]  total_token_movement=%.4f  "
            "init->final val_rmse: %.4f -> %.4f",
            lr, sup_losses[0], sup_losses[-1],
            sum(movements), val_rmses[0][1], rv_final,
        )


if __name__ == "__main__":
    main()
