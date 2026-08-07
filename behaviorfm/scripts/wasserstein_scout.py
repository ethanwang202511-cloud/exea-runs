"""Step 3 (proposal §7 attack-order): scout the H0 distance metric's
dynamic range on the 5-species AP-10K subset.

Question (per proposal §7): does the per-backbone Wasserstein
distance between a held-out species' demos and the convex hull of
training-species tokens have enough dynamic range to plausibly
produce a phase boundary? If the distance is constant or pathological,
the H0 design needs revision.

We compute, for each species in the cache, distance metrics from THAT
species' val examples to the convex hull of {top-5 training species}'
train examples, in the FROZEN DINOv2-base feature space. The 5
training species themselves provide the "in-distribution" reference;
giraffe (held-out, never seen by training) provides the "out-of-
distribution" signal.

The scout is purely diagnostic; it does NOT pre-register the H0
threshold (that lock happens in iti_v2.yaml at start of Year 2 with
all 3 substrates).
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset
from iti.eval.wasserstein import WasserstainScout, aggregate_per_species

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wasserstein_scout")


def gather_features_per_species(ds: CachedFeatureDataset) -> dict[int, np.ndarray]:
    """Returns gid -> (n_examples, n_patches, D)."""
    by: dict[int, list[np.ndarray]] = {}
    for i in range(len(ds)):
        item = ds[i]
        by.setdefault(item["global_identity_id"], []).append(item["feature"].numpy())
    return {gid: np.stack(v) for gid, v in by.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache",   type=Path, required=True)
    p.add_argument("--training-species", type=str,
                   default="dog,sheep,cat,antelope,zebra")
    p.add_argument("--out-csv", type=Path, default=ROOT / "results" / "wasserstein_scout.csv")
    args = p.parse_args()

    train_ds = CachedFeatureDataset(args.train_cache)
    val_ds = CachedFeatureDataset(args.val_cache)

    train_names = [s.strip() for s in args.training_species.split(",")]
    name_to_gid = {r["species_name"]: r["identity_id"] for r in train_ds.records}
    training_gids = {name_to_gid[n] for n in train_names if n in name_to_gid}
    logger.info("Training species (global ids): %s", training_gids)

    train_features = gather_features_per_species(train_ds)
    train_pool = {gid: arr for gid, arr in train_features.items() if gid in training_gids}
    val_features = gather_features_per_species(val_ds)

    scout = WasserstainScout(train_pool)

    rows = []
    species_lookup = {r["identity_id"]: r["species_name"] for r in val_ds.records}
    for gid, feats in val_features.items():
        name = species_lookup[gid]
        in_train = gid in training_gids
        sub = feats[: min(16, len(feats))]   # cap demos to keep Sinkhorn cheap
        d_hull = scout.distance_to_hull_mean(sub)
        d_sink = scout.sinkhorn_to_each_train_species(sub)
        row = {
            "species": name, "global_id": int(gid),
            "in_training_pool": bool(in_train),
            "n_val_examples": int(len(feats)),
            **d_hull,
            **{k: v for k, v in d_sink.items() if k != "sinkhorn_w2_per_train_species"},
        }
        rows.append(row)
        logger.info(
            "%-15s in_train=%s  euclid_d_to_centroid=%.3f  cos_max=%.3f  sink_min=%.3f  sink_mean=%.3f",
            name, in_train, row["euclid_d_to_training_centroid"],
            row["cos_sim_max_to_train_species_mean"],
            row["sinkhorn_w2_min"], row["sinkhorn_w2_mean"],
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with args.out_csv.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    logger.info("wrote %s", args.out_csv)

    # Summary: dynamic-range stats for in-vs-out training pool.
    by_in = [r for r in rows if r["in_training_pool"]]
    by_out = [r for r in rows if not r["in_training_pool"]]
    if by_in and by_out:
        for metric in ("euclid_d_to_training_centroid", "sinkhorn_w2_min", "sinkhorn_w2_mean"):
            d_in = np.asarray([r[metric] for r in by_in])
            d_out = np.asarray([r[metric] for r in by_out])
            d_in_clean = d_in[~np.isnan(d_in)]
            d_out_clean = d_out[~np.isnan(d_out)]
            logger.info(
                "%-30s  IN: mean=%.3f sd=%.3f  | OUT: mean=%.3f sd=%.3f  | spread=%.3f",
                metric,
                d_in_clean.mean() if len(d_in_clean) else float("nan"),
                d_in_clean.std() if len(d_in_clean) else float("nan"),
                d_out_clean.mean() if len(d_out_clean) else float("nan"),
                d_out_clean.std() if len(d_out_clean) else float("nan"),
                (d_out_clean.mean() - d_in_clean.mean()) / max(d_in_clean.std(), 1e-6)
                if len(d_in_clean) and len(d_out_clean) else float("nan"),
            )


if __name__ == "__main__":
    main()
