"""Step 4 (proposal §7 attack-order): H1 mini-pilot on giraffe (held-out).

Loads the within-species-pilot checkpoint (trained on dog/sheep/cat/
antelope/zebra). Evaluates 5 adaptation strategies on giraffe val:

  (1) NO-ADAPT-RANDOM-TOKEN   : random init token, no SGD             (control)
  (2) NO-ADAPT-MEAN-TOKEN     : init = mean of training tokens, no SGD (control)
  (3) NO-ADAPT-INTERP-TOKEN   : init via interpolation head over k=5 demos, no SGD
  (4) ADAPT-VANILLA-SOFT-PROMPT  : random init token + 50 SGD steps on k=5 demos
  (5) ADAPT-ITI-FULL          : interp-init token + 50 SGD steps on k=5 demos

The decoder, FiLM, and interpolation head are FROZEN throughout.
Only the per-identity token is updated when adaptation happens.

For "decoder upper bound" we run a separate script that retrains the
decoder on giraffe data — see run_within_species_pilot.py with
training-species=giraffe.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset, collate_cached
from iti.eval.pck import per_keypoint_distance_normalized, pck_metrics, bootstrap_ci
from iti.model.iti import ITIModel, ITIConfig
from iti.model.identity_token import IdentityTokenBank

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("h1_mini_pilot")


def load_trained_model(ckpt_path: Path, device: str) -> tuple[ITIModel, dict]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state["cfg"]
    cfg = ITIConfig(**{k: v for k, v in cfg_dict.items()
                       if k in ITIConfig.__dataclass_fields__})
    model = ITIModel(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model, state


def evaluate_with_token(
    model: ITIModel,
    eval_loader_batches: list[dict],
    id_token: torch.Tensor,             # (768,) on the right device
    device: str,
    out_size: int = 224,
) -> dict[str, float]:
    model.eval()
    dists, diags = [], []
    with torch.no_grad():
        for batch in eval_loader_batches:
            feat = batch["feature"].to(device)
            B = feat.shape[0]
            tok = id_token.unsqueeze(0).expand(B, -1)
            pred_unit = model.forward_with_token(feat, tok).cpu()
            pred_xy = pred_unit * out_size
            d = per_keypoint_distance_normalized(
                pred_xy.numpy(), batch["keypoints_xy"].numpy(),
                batch["vis"].numpy(), batch["crop_side"].numpy(), out_size=out_size,
            )
            dists.append(d)
            diags.append(batch["bbox_diag_orig"].numpy())
    dists = np.concatenate(dists)
    diags = np.concatenate(diags)
    metrics = pck_metrics(dists, diags)
    flat = (dists / diags[:, None]).flatten()
    flat = flat[~np.isnan(flat)]
    if flat.size:
        sq = flat ** 2
        m, lo, hi = bootstrap_ci(sq, n_resamples=500)
        metrics["rmse_norm_ci95_lo"] = float(np.sqrt(lo))
        metrics["rmse_norm_ci95_hi"] = float(np.sqrt(hi))
    return metrics


def adapt_token(
    model: ITIModel,
    init_token: torch.Tensor,
    support_features: torch.Tensor,         # (k, n_patches, 768) on device
    support_kp: torch.Tensor,               # (k, K, 2) on device, in 224 px
    support_vis: torch.Tensor,              # (k, K) on device
    n_steps: int = 50,
    lr: float = 1e-2,
    out_size: int = 224,
) -> torch.Tensor:
    tok = nn.Parameter(init_token.clone())
    opt = torch.optim.SGD([tok], lr=lr, momentum=0.9)
    model.eval()           # decoder/FiLM/etc frozen
    for p in model.parameters():
        p.requires_grad_(False)
    target = support_kp / out_size                                     # (k, K, 2) in [0,1]
    for step in range(n_steps):
        B = support_features.shape[0]
        pred = model.forward_with_token(support_features, tok.unsqueeze(0).expand(B, -1))
        per_kp = ((pred - target) ** 2).sum(dim=-1)                    # (k, K)
        denom = support_vis.sum().clamp_min(1.0)
        loss = (per_kp * support_vis).sum() / denom
        opt.zero_grad()
        loss.backward()
        opt.step()
    return tok.detach()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "results/within_species_pilot/iti_full_seed0/model.pt")
    p.add_argument("--train-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_train_split1_pilot6")
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_pilot6")
    p.add_argument("--held-out-species", type=str, default="giraffe")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-adapt-steps", type=int, default=50)
    p.add_argument("--adapt-lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--out-csv", type=Path, default=ROOT / "results/h1_mini_pilot.csv")
    p.add_argument("--paired-ci", action="store_true",
                   help="run a paired-bootstrap CI on the per-(seed) H1 deltas")
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    model, state = load_trained_model(args.ckpt, device)
    train_species_names = state["train_species_names"]
    logger.info("Training species: %s", train_species_names)
    logger.info("Decoder/token trainable params (post-load): %d",
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Resolve held-out species global id
    train_full = CachedFeatureDataset(args.train_cache)
    val_full = CachedFeatureDataset(args.val_cache)
    name_to_gid = {r["species_name"]: r["identity_id"] for r in train_full.records}
    if args.held_out_species not in name_to_gid:
        raise SystemExit(f"held-out species {args.held_out_species} missing in train cache")
    held_gid = name_to_gid[args.held_out_species]
    train_held_ds = CachedFeatureDataset(args.train_cache,
                                         identity_subset={held_gid})
    val_held_ds = CachedFeatureDataset(args.val_cache,
                                       identity_subset={held_gid})
    logger.info("Held-out species %s (gid=%d): %d train, %d val examples",
                args.held_out_species, held_gid, len(train_held_ds), len(val_held_ds))

    # Pre-pull all val batches to a list (small dataset)
    from torch.utils.data import DataLoader
    val_batches = list(DataLoader(val_held_ds, batch_size=32, shuffle=False,
                                  collate_fn=collate_cached))
    train_batches = list(DataLoader(train_held_ds, batch_size=len(train_held_ds),
                                    shuffle=False, collate_fn=collate_cached))
    train_batch = train_batches[0]
    train_features_full = train_batch["feature"].to(device)               # (N_train, 256, 768)

    # Get the trained training-id-token bank.
    id_bank = model.tokens.emb.weight.detach()           # (n_train_ids, 768) on device
    interp_head = model.interpolation_head

    rows = []
    for seed in range(args.seed, args.seed + args.n_seeds):
        rng = np.random.default_rng(seed)
        # Pick k support examples for this seed
        N_train = len(train_held_ds)
        idx = rng.choice(N_train, size=min(args.k_shot, N_train), replace=False)
        support_features = train_features_full[idx]
        support_kp = train_batch["keypoints_xy"][idx].to(device)
        support_vis = train_batch["vis"][idx].to(device)
        logger.info("seed %d: support indices %s", seed, idx.tolist())

        # ---------- (1) NO-ADAPT-RANDOM-TOKEN -------------------------------
        torch.manual_seed(seed)
        random_tok = torch.zeros(768, device=device)
        nn.init.normal_(random_tok, std=0.02)
        m = evaluate_with_token(model, val_batches, random_tok, device)
        rows.append({"strategy": "no_adapt_random_token", "n_trainable_per_id": 0,
                     "seed": seed, **m})
        logger.info("[seed=%d] no_adapt_random_token  rmse_norm=%.4f  pck@0.05=%.3f",
                    seed, m["rmse_norm"], m["pck@0.05"])

        # ---------- (2) NO-ADAPT-MEAN-TOKEN ---------------------------------
        mean_tok = id_bank.mean(dim=0)
        m = evaluate_with_token(model, val_batches, mean_tok, device)
        rows.append({"strategy": "no_adapt_mean_token", "n_trainable_per_id": 0,
                     "seed": seed, **m})
        logger.info("[seed=%d] no_adapt_mean_token    rmse_norm=%.4f  pck@0.05=%.3f",
                    seed, m["rmse_norm"], m["pck@0.05"])

        # ---------- (3) NO-ADAPT-INTERP-TOKEN -------------------------------
        with torch.no_grad():
            # interp head expects (B, k, F); use mean-pooled patch features as the per-demo summary
            ex_feat = support_features.mean(dim=1)                           # (k, 768)
            ex_feat_b = ex_feat.unsqueeze(0)                                 # (1, k, 768)
            interp_init, alphas = interp_head(ex_feat_b, id_bank)
            interp_tok = interp_init.squeeze(0)
        m = evaluate_with_token(model, val_batches, interp_tok, device)
        rows.append({"strategy": "no_adapt_interp_token", "n_trainable_per_id": 0,
                     "seed": seed, **m,
                     "alphas": alphas.squeeze(0).cpu().numpy().tolist()})
        logger.info("[seed=%d] no_adapt_interp_token  rmse_norm=%.4f  pck@0.05=%.3f  alphas=%s",
                    seed, m["rmse_norm"], m["pck@0.05"],
                    [f"{a:.2f}" for a in alphas.squeeze(0).cpu().numpy()])

        # ---------- (4) ADAPT-VANILLA-SOFT-PROMPT ---------------------------
        torch.manual_seed(seed + 1000)
        random_init2 = torch.zeros(768, device=device)
        nn.init.normal_(random_init2, std=0.02)
        adapted = adapt_token(model, random_init2, support_features,
                              support_kp, support_vis,
                              n_steps=args.n_adapt_steps, lr=args.adapt_lr)
        m = evaluate_with_token(model, val_batches, adapted, device)
        rows.append({"strategy": "adapt_vanilla_soft_prompt", "n_trainable_per_id": 768,
                     "seed": seed, **m})
        logger.info("[seed=%d] adapt_vanilla_softpr   rmse_norm=%.4f  pck@0.05=%.3f",
                    seed, m["rmse_norm"], m["pck@0.05"])

        # ---------- (5) ADAPT-ITI-FULL --------------------------------------
        adapted_iti = adapt_token(model, interp_tok, support_features,
                                  support_kp, support_vis,
                                  n_steps=args.n_adapt_steps, lr=args.adapt_lr)
        m = evaluate_with_token(model, val_batches, adapted_iti, device)
        rows.append({"strategy": "adapt_iti_full", "n_trainable_per_id": 768,
                     "seed": seed, **m})
        logger.info("[seed=%d] adapt_iti_full         rmse_norm=%.4f  pck@0.05=%.3f",
                    seed, m["rmse_norm"], m["pck@0.05"])

        # Re-restore requires_grad for next iteration  (adapt_token freezes the model)
        for p in model.parameters():
            p.requires_grad_(False)
        # Just leave model frozen; we never update it again here.

    # Write CSV
    cols = ["strategy", "seed", "n_trainable_per_id",
            "pck@0.05", "pck@0.10", "pck@0.20",
            "rmse_norm", "mae_norm",
            "rmse_norm_ci95_lo", "rmse_norm_ci95_hi",
            "n_keypoints_evaluated"]
    with args.out_csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    logger.info("wrote %s", args.out_csv)

    # Print summary table averaged across seeds
    by_strategy: dict[str, list[float]] = {}
    by_strategy_pck: dict[str, list[float]] = {}
    for r in rows:
        by_strategy.setdefault(r["strategy"], []).append(r["rmse_norm"])
        by_strategy_pck.setdefault(r["strategy"], []).append(r["pck@0.05"])
    logger.info("=" * 80)
    logger.info("SUMMARY  held-out: %s   k-shot: %d   n-seeds: %d",
                args.held_out_species, args.k_shot, args.n_seeds)
    logger.info("%-30s  rmse_norm (mean ± sd)   pck@0.05 (mean)", "strategy")
    for strat, vals in by_strategy.items():
        v = np.asarray(vals); pv = np.asarray(by_strategy_pck[strat])
        logger.info("%-30s  %.4f ± %.4f         %.3f",
                    strat, v.mean(), v.std(), pv.mean())

    # Paired bootstrap CI: adapt_iti_full vs no_adapt_mean_token, per-seed differences
    if args.paired_ci and "adapt_iti_full" in by_strategy and "no_adapt_mean_token" in by_strategy:
        adapt = np.asarray(by_strategy["adapt_iti_full"])
        baseline = np.asarray(by_strategy["no_adapt_mean_token"])
        deltas = adapt - baseline
        logger.info("--- PAIRED CI (adapt_iti_full - no_adapt_mean_token) ---")
        logger.info("Per-seed deltas (n=%d): %s", len(deltas),
                    " ".join(f"{d:+.4f}" for d in deltas))
        logger.info("Mean delta: %+.4f  (SE: %.4f)", deltas.mean(), deltas.std()/np.sqrt(len(deltas)))
        # bootstrap the seed-level mean delta
        rng = np.random.default_rng(0)
        n_boot = 5000
        boot_means = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, len(deltas), size=len(deltas))
            boot_means[i] = deltas[idx].mean()
        lo, hi = np.quantile(boot_means, 0.025), np.quantile(boot_means, 0.975)
        logger.info("Paired bootstrap 95%% CI on mean delta: [%+.4f, %+.4f]", lo, hi)
        # signed-test across seeds
        n_neg = int((deltas < 0).sum())
        logger.info("Signed test (deltas < 0): %d / %d seeds favor adaptation", n_neg, len(deltas))


if __name__ == "__main__":
    main()
