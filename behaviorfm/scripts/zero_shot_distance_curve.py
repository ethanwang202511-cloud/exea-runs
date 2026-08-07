"""Zero-shot RMSE-vs-distance curve on AP-10K val.

If zero-shot RMSE is essentially flat across cos_max-to-training-pool
distance, AP-10K is structurally degenerate for the H1 cross-species
adaptation claim — the floor is too low for any adaptation method to
demonstrate gain. We test by
evaluating the trained Stage-1 model with mean-of-train-tokens on
EVERY species' val crops, then plotting RMSE vs cos_max distance.

If RMSE rises sharply at cos_max < 0.6 (e.g. alouatta, chimpanzee,
hamster), AP-10K can support the H1 binding test with cross-family
species. If RMSE stays ~ 0.08–0.12 across all cos_max levels, AP-10K
is unsalvageable for H1 and we must move to substrate alternatives.
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
from iti.eval.pck import per_keypoint_distance_normalized, pck_metrics, bootstrap_ci
from iti.model.iti import ITIModel, ITIConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("zero_shot_distance_curve")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "results/within_species_pilot/iti_full_seed0/model.pt")
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all")
    p.add_argument("--wasserstein-csv", type=Path,
                   default=ROOT / "results/wasserstein_scout.csv")
    p.add_argument("--out-csv", type=Path,
                   default=ROOT / "results/zero_shot_distance_curve.csv")
    p.add_argument("--device", type=str, default="mps")
    args = p.parse_args()

    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg_dict = state["cfg"]
    cfg = ITIConfig(**{k: v for k, v in cfg_dict.items()
                       if k in ITIConfig.__dataclass_fields__})
    model = ITIModel(cfg).to(args.device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    logger.info("Loaded ckpt %s; train species %s", args.ckpt, state["train_species_names"])

    # Load distance metrics
    import csv
    with args.wasserstein_csv.open() as f:
        dist_rows = list(csv.DictReader(f))
    cos_max_by_species = {r["species"]: float(r["cos_sim_max_to_train_species_mean"])
                          for r in dist_rows}
    in_train_by_species = {r["species"]: r["in_training_pool"] == "True"
                           for r in dist_rows}

    val_full = CachedFeatureDataset(args.val_cache)
    species_idx: dict[str, list[int]] = {}
    for i, r in enumerate(val_full.records):
        species_idx.setdefault(r["species_name"], []).append(i)

    # Build mean-of-trained-tokens (5 species) on device
    train_tokens = model.tokens.emb.weight.detach()           # (5, 768)
    mean_tok = train_tokens.mean(dim=0)

    out_size = 224
    rows = []
    for species, indices in species_idx.items():
        feats, kps, viss, sides, diags = [], [], [], [], []
        for i in indices:
            it = val_full[i]
            feats.append(it["feature"]); kps.append(it["keypoints_xy"])
            viss.append(it["vis"]); sides.append(it["crop_side"])
            diags.append(it["bbox_diag_orig"])
        F = torch.stack(feats).to(args.device)
        with torch.no_grad():
            B = F.shape[0]
            pred_unit = model.forward_with_token(F, mean_tok.unsqueeze(0).expand(B, -1)).cpu()
        pred_xy = pred_unit * out_size
        kp = torch.stack(kps).numpy()
        vis = torch.stack(viss).numpy()
        side = np.asarray(sides); diag = np.asarray(diags)
        d = per_keypoint_distance_normalized(pred_xy.numpy(), kp, vis, side, out_size=out_size)
        m = pck_metrics(d, diag)
        # bootstrap CI on rmse_norm
        flat = (d / diag[:, None]).flatten()
        flat = flat[~np.isnan(flat)]
        if flat.size:
            sq = flat ** 2
            mn, lo, hi = bootstrap_ci(sq, n_resamples=300)
            m["rmse_norm_ci95_lo"] = float(np.sqrt(lo))
            m["rmse_norm_ci95_hi"] = float(np.sqrt(hi))

        cos_max = cos_max_by_species.get(species, float("nan"))
        in_train = in_train_by_species.get(species, False)
        row = {
            "species": species, "n_val": len(indices),
            "cos_max_to_train_pool": cos_max,
            "in_training_pool": in_train,
            **m,
        }
        rows.append(row)

    rows.sort(key=lambda r: -r["cos_max_to_train_pool"])  # descending: closest first

    # Print binned summary
    bins = [(0.85, 1.0, "in-pool ish (cos>=0.85)"),
            (0.70, 0.85, "near (0.70<=cos<0.85)"),
            (0.55, 0.70, "mid  (0.55<=cos<0.70)"),
            (0.0, 0.55, "far  (cos<0.55)")]
    for lo, hi, label in bins:
        sub = [r for r in rows
               if not r["in_training_pool"] and lo <= r["cos_max_to_train_pool"] < hi]
        if not sub:
            continue
        rmses = np.asarray([r["rmse_norm"] for r in sub])
        pcks = np.asarray([r["pck@0.05"] for r in sub])
        logger.info("%s: n=%d  rmse_norm: mean=%.4f sd=%.4f range=[%.4f, %.4f]  pck@0.05: mean=%.3f",
                    label, len(sub), rmses.mean(), rmses.std(),
                    rmses.min(), rmses.max(), pcks.mean())
        for r in sub:
            logger.info("    %-22s  cos_max=%.3f  rmse_norm=%.4f  pck@0.05=%.3f  n_val=%d",
                        r["species"], r["cos_max_to_train_pool"], r["rmse_norm"],
                        r["pck@0.05"], r["n_val"])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["species", "n_val", "cos_max_to_train_pool", "in_training_pool",
            "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20",
            "rmse_norm_ci95_lo", "rmse_norm_ci95_hi", "n_keypoints_evaluated"]
    with args.out_csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    logger.info("wrote %s", args.out_csv)


if __name__ == "__main__":
    main()
