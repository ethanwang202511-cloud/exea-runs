"""H5a binding test: interp init vs random init at k=0/1, NO SGD.

The proposal §4 H5a hypothesis: "At k=0 (zero-shot via interpolation
only) and k=1 (one-shot), the learned-interpolation initialization
beats random-init token by >= 15% RMSE on H1's animal-pose benchmark."

This script implements the test. At each k in {0, 1, 2, 5} and for
each held-out species (elephant, chimpanzee, +gorilla if available),
we compare:

  - random_init      : random token, no SGD
  - mean_init        : mean-of-train-tokens (=interp at k=0 with no demos)
  - interp_init_k    : interp head with k demos, no SGD

The H5a binding claim is whether interp_init_k=1 beats random_init by >= 15%.
H5b binding-stretch is >= 25%.

n_seeds=10 by default for statistical rigor.
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
logger = logging.getLogger("h5a_binding")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "results/q2_within_species/iti_xattn_aux01_seed0/model.pt")
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all")
    p.add_argument("--held-out-species", type=str, required=True)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--ks", type=str, default="0,1,2,5")
    p.add_argument("--out-csv", type=Path, default=ROOT / "results/h5a_binding.csv")
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                       if k in ITIConfig.__dataclass_fields__})
    model = ITIModel(cfg).to(args.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    train_full = CachedFeatureDataset(args.train_cache)
    val_full = CachedFeatureDataset(args.val_cache)
    name_to_gid = {r["species_name"]: r["identity_id"] for r in train_full.records}
    gid = name_to_gid[args.held_out_species]
    train_held = CachedFeatureDataset(args.train_cache, identity_subset={gid})
    val_held = CachedFeatureDataset(args.val_cache, identity_subset={gid})
    val_batches = list(DataLoader(val_held, batch_size=32, shuffle=False,
                                  collate_fn=collate_cached))
    train_batch = next(iter(DataLoader(train_held, batch_size=len(train_held),
                                       shuffle=False, collate_fn=collate_cached)))
    train_features = train_batch["feature"].to(args.device)

    id_bank = model.tokens.emb.weight.detach()
    mean_tok = id_bank.mean(dim=0)

    def eval_token(tok: torch.Tensor) -> dict[str, float]:
        dists, diags = [], []
        with torch.no_grad():
            for batch in val_batches:
                fv = batch["feature"].to(args.device)
                Bv = fv.shape[0]
                pred_unit = model.forward_with_token(fv, tok.unsqueeze(0).expand(Bv, -1)).cpu()
                pred_xy = pred_unit * 224
                d = per_keypoint_distance_normalized(
                    pred_xy.numpy(), batch["keypoints_xy"].numpy(),
                    batch["vis"].numpy(), batch["crop_side"].numpy(), out_size=224,
                )
                dists.append(d); diags.append(batch["bbox_diag_orig"].numpy())
        d = np.concatenate(dists); diag = np.concatenate(diags)
        return pck_metrics(d, diag)

    ks = [int(s) for s in args.ks.split(",")]
    rows = []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)

        # random-init baseline (one per seed)
        torch.manual_seed(seed)
        rt = torch.randn(768, device=args.device) * 0.02
        m = eval_token(rt)
        rows.append({"species": args.held_out_species, "init": "random", "k": -1,
                     "seed": seed, **m})

        # mean-of-train-tokens (deterministic across seeds)
        if seed == 0:
            m = eval_token(mean_tok)
            rows.append({"species": args.held_out_species, "init": "mean", "k": 0,
                         "seed": 0, **m})

        # interp_init at each k > 0
        for k in ks:
            if k == 0:
                # k=0 with interp head defined as: query is zero (no demo info) -> uniform alphas
                # But our head needs a query. Provide all-zeros-feature as a "no support" probe.
                with torch.no_grad():
                    zero_feat = torch.zeros(1, 1, 768, device=args.device)
                    init_tok, _ = model.interpolation_head(zero_feat, id_bank)
                    init_tok = init_tok.squeeze(0)
                m = eval_token(init_tok)
                rows.append({"species": args.held_out_species, "init": "interp_zero_query",
                             "k": 0, "seed": seed, **m})
                continue
            n_train = len(train_held)
            if k > n_train:
                continue
            idx = rng.choice(n_train, size=k, replace=False)
            sup_f = train_features[idx]                  # (k, 256, 768)
            with torch.no_grad():
                ex_feat = sup_f.mean(dim=1).unsqueeze(0)  # (1, k, 768)
                init_tok, _ = model.interpolation_head(ex_feat, id_bank)
                init_tok = init_tok.squeeze(0)
            m = eval_token(init_tok)
            rows.append({"species": args.held_out_species, "init": "interp",
                         "k": k, "seed": seed, **m})

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    cols = ["species", "init", "k", "seed", "rmse_norm",
            "pck@0.05", "pck@0.10", "pck@0.20", "n_keypoints_evaluated"]
    with args.out_csv.open("a") as f:
        if write_header:
            f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # Summary
    by = {}
    for r in rows:
        key = (r["init"], r["k"])
        by.setdefault(key, []).append(r["rmse_norm"])
    logger.info("=" * 80)
    logger.info("H5a SUMMARY  held-out: %s   n_seeds: %d", args.held_out_species, args.n_seeds)
    for (init, k), vals in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        v = np.asarray(vals)
        logger.info("  init=%-18s  k=%2d  n=%2d  rmse_norm = %.4f ± %.4f",
                    init, k, len(v), v.mean(), v.std())

    # H5a comparison: interp(k=1) vs random
    interp_k1 = np.asarray(by.get(("interp", 1), []))
    random_vals = np.asarray(by.get(("random", -1), []))
    if len(interp_k1) and len(random_vals):
        # paired comparison (per-seed)
        deltas = interp_k1 - random_vals
        mean_d = deltas.mean()
        rel = mean_d / random_vals.mean()
        logger.info("--- H5a binding test ---")
        logger.info("interp(k=1) mean RMSE: %.4f", interp_k1.mean())
        logger.info("random      mean RMSE: %.4f", random_vals.mean())
        logger.info("Delta (interp - random): %+.4f  (relative: %+.1f%%)",
                    mean_d, 100 * rel)
        logger.info("H5a binding (interp BEATS random by >= 15%%): %s",
                    "PASS" if rel <= -0.15 else "FAIL")
        logger.info("H5b stretch  (interp BEATS random by >= 25%%): %s",
                    "PASS" if rel <= -0.25 else "FAIL")


if __name__ == "__main__":
    main()
