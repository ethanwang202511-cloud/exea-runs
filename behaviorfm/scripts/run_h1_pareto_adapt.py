"""H1 Pareto-frontier adaptation: a third parameter-budget point between
768 (token-only) and 8.5M (full decoder-FT).

Variants tested at adaptation time on a held-out species:
  - "token_only"        : update id_token only (768 params)
  - "token_plus_id_proj": update id_token + decoder.id_proj (295,296 params)
  - "token_plus_xattn"  : update id_token + decoder.cross_attn_blocks[0].attn (~590K params)

All decoder weights frozen except the named subset; backbone always frozen.
Initialization: interp head output for token; original ckpt weights for the
adapted modules.
"""

from __future__ import annotations
import argparse
import json
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
logger = logging.getLogger("h1_pareto")


def select_modules(model: ITIModel, variant: str) -> list[nn.Parameter]:
    """Return the list of nn.Parameters to update at adaptation time.

    The id_token is always trained; this function returns the OTHER params.
    """
    if variant == "token_only":
        return []
    if variant == "token_plus_coord_head_last":
        # Just the LAST Linear of coord_head (~770 params): a quick H7 partial
        # test (per-species coord-head adaptation, paper §6 item 5).
        return list(model.decoder.coord_head[-1].parameters())
    if variant == "token_plus_coord_head":
        # The whole coord_head (~150K params): more aggressive H7 partial test
        return list(model.decoder.coord_head.parameters())
    if variant == "token_plus_id_proj":
        return list(model.decoder.id_proj.parameters())
    if variant == "token_plus_xattn0":
        return list(model.decoder.cross_attn_blocks[0].parameters())
    raise ValueError(variant)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "results/q2_within_species/iti_xattn_aux01_seed0/model.pt")
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all")
    p.add_argument("--held-out-species", type=str, required=True)
    p.add_argument("--out-csv", type=Path,
                   default=ROOT / "results/h1_pareto.csv")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-adapt-steps", type=int, default=50)
    p.add_argument("--adapt-lr", type=float, default=1.0)
    p.add_argument("--n-seeds", type=int, default=3)
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                       if k in ITIConfig.__dataclass_fields__})

    train_full = CachedFeatureDataset(args.train_cache)
    val_full = CachedFeatureDataset(args.val_cache)
    name_to_gid = {r["species_name"]: r["identity_id"] for r in train_full.records}
    gid = name_to_gid[args.held_out_species]
    train_held = CachedFeatureDataset(args.train_cache, identity_subset={gid})
    val_held = CachedFeatureDataset(args.val_cache, identity_subset={gid})

    val_batches = list(DataLoader(val_held, batch_size=32, shuffle=False,
                                  collate_fn=collate_cached))
    train_batches = list(DataLoader(train_held, batch_size=len(train_held),
                                    shuffle=False, collate_fn=collate_cached))
    train_batch = train_batches[0]
    train_features = train_batch["feature"].to(args.device)

    rows = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(train_held), size=args.k_shot, replace=False)
        sup_f = train_features[idx]
        sup_kp = train_batch["keypoints_xy"][idx].to(args.device)
        sup_vis = train_batch["vis"][idx].to(args.device)

        for variant in ("token_only", "token_plus_coord_head_last",
                        "token_plus_coord_head", "token_plus_id_proj",
                        "token_plus_xattn0"):
            # Fresh model per variant
            model = ITIModel(cfg).to(args.device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            for p_ in model.parameters():
                p_.requires_grad_(False)

            # Init token via interp head (using the 5-species id_bank)
            id_bank = model.tokens.emb.weight.detach()
            with torch.no_grad():
                ex_feat = sup_f.mean(dim=1).unsqueeze(0)
                init_tok, _ = model.interpolation_head(ex_feat, id_bank)
                init_tok = init_tok.squeeze(0)

            tok = nn.Parameter(init_tok.clone())
            extra_params = select_modules(model, variant)
            for p_ in extra_params:
                p_.requires_grad_(True)
            params_to_opt = [tok] + extra_params
            n_adapt_params = sum(p.numel() for p in params_to_opt)
            opt = torch.optim.SGD(params_to_opt, lr=args.adapt_lr, momentum=0.9)

            for step in range(args.n_adapt_steps):
                B = sup_f.shape[0]
                pred = model.forward_with_token(sup_f, tok.unsqueeze(0).expand(B, -1))
                target = sup_kp / 224
                per_kp = ((pred - target) ** 2).sum(dim=-1)
                loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Evaluate
            dists, diags = [], []
            with torch.no_grad():
                for batch in val_batches:
                    fv = batch["feature"].to(args.device)
                    Bv = fv.shape[0]
                    pred_unit = model.forward_with_token(fv, tok.detach().unsqueeze(0).expand(Bv, -1)).cpu()
                    pred_xy = pred_unit * 224
                    d = per_keypoint_distance_normalized(
                        pred_xy.numpy(), batch["keypoints_xy"].numpy(),
                        batch["vis"].numpy(), batch["crop_side"].numpy(), out_size=224,
                    )
                    dists.append(d)
                    diags.append(batch["bbox_diag_orig"].numpy())
            d = np.concatenate(dists); diag = np.concatenate(diags)
            m = pck_metrics(d, diag)
            row = {"species": args.held_out_species, "variant": variant,
                   "n_adapt_params": n_adapt_params, "seed": seed,
                   "n_adapt_steps": args.n_adapt_steps, "adapt_lr": args.adapt_lr,
                   **m}
            rows.append(row)
            logger.info("[seed=%d] %-22s  n_params=%-7d  rmse=%.4f  pck@0.05=%.3f",
                        seed, variant, n_adapt_params, m["rmse_norm"], m["pck@0.05"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    cols = ["species", "variant", "n_adapt_params", "seed", "n_adapt_steps", "adapt_lr",
            "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20", "n_keypoints_evaluated"]
    with args.out_csv.open("a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    logger.info("appended %d rows to %s", len(rows), args.out_csv)

    # summary per variant
    for variant in ("token_only", "token_plus_coord_head_last",
                    "token_plus_coord_head", "token_plus_id_proj",
                    "token_plus_xattn0"):
        sub = [r for r in rows if r["variant"] == variant]
        if sub:
            rmses = np.asarray([r["rmse_norm"] for r in sub])
            pcks = np.asarray([r["pck@0.05"] for r in sub])
            logger.info("SUMMARY %s n_params=%d  rmse_norm=%.4f ± %.4f  pck@0.05=%.3f",
                        variant, sub[0]["n_adapt_params"],
                        rmses.mean(), rmses.std(), pcks.mean())


if __name__ == "__main__":
    main()
