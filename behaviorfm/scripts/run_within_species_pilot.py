"""Run the Year-1 Q1 within-species pose pilot (B1 sanity).

Trains an ITIModel on the top-5 AP-10K species using cached DINOv2-base
features. Reports B1 metrics on the val split. Also runs a "linear-probe"
baseline (decoder only, no per-identity token) for comparison.

Usage::

    PYTHONPATH=src python3 scripts/run_within_species_pilot.py \\
        --train-cache data/cache/dinov2_base_ap10k_train_split1_pilot6 \\
        --val-cache   data/cache/dinov2_base_ap10k_val_split1_pilot6 \\
        --out-dir     results/within_species_pilot \\
        --device mps
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset
from iti.model.iti import ITIConfig, ITIModel
from iti.train.trainer import TrainConfig, train_within_species

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("within_species_pilot")


def list_species_in_cache(ds: CachedFeatureDataset) -> list[str]:
    seen: dict[int, str] = {}
    for r in ds.records:
        seen[r["identity_id"]] = r["species_name"]
    return [seen[k] for k in sorted(seen)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", type=Path, required=True)
    p.add_argument("--val-cache",   type=Path, required=True)
    p.add_argument("--out-dir",     type=Path, required=True)
    p.add_argument("--training-species", type=str,
                   default="dog,sheep,cat,antelope,zebra",
                   help="comma-separated; held-out (giraffe) is excluded")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--n-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-id-token", action="store_true",
                   help="ablation: zero out identity tokens for the linear-probe-style baseline")
    p.add_argument("--token-utility-weight", type=float, default=0.0,
                   help="paper §6 item 4: aux loss forcing token to be informationally necessary")
    p.add_argument("--token-utility-margin", type=float, default=0.0)
    p.add_argument("--interp-head-version", type=str, default="v1", choices=("v1", "v2"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_full = CachedFeatureDataset(args.train_cache)
    val_full = CachedFeatureDataset(args.val_cache)

    # Build species mapping: training species names -> (global_id, local_id)
    train_names = [s.strip() for s in args.training_species.split(",")]
    name_to_global: dict[str, int] = {}
    for r in train_full.records:
        name_to_global.setdefault(r["species_name"], r["identity_id"])
    missing = [n for n in train_names if n not in name_to_global]
    if missing:
        raise SystemExit(f"training species missing in cache: {missing}")
    global_ids_training = [name_to_global[n] for n in train_names]
    remap = {gid: i for i, gid in enumerate(global_ids_training)}
    n_training_identities = len(remap)
    logger.info("Training identities (global -> local): %s",
                {gid: (i, train_names[i]) for gid, i in remap.items()})

    train_ds = CachedFeatureDataset(
        args.train_cache,
        identity_subset=set(global_ids_training),
        identity_remap=remap,
    )
    val_ds = CachedFeatureDataset(
        args.val_cache,
        identity_subset=set(global_ids_training),
        identity_remap=remap,
    )
    logger.info("train N=%d  val N=%d  (training species only, no held-out)",
                len(train_ds), len(val_ds))

    cfg = ITIConfig(n_training_identities=n_training_identities,
                    interp_head_version=getattr(args, "interp_head_version", "v1"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ITIModel(cfg)

    if args.no_id_token:
        # Ablation: zero out tokens AND freeze them so the model is forced to
        # do everything from features.
        with torch.no_grad():
            model.tokens.emb.weight.zero_()
        model.tokens.emb.weight.requires_grad_(False)
        logger.info("Ablation: identity tokens are zeroed and frozen.")

    train_cfg = TrainConfig(
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed,
        token_utility_weight=args.token_utility_weight,
        token_utility_margin=args.token_utility_margin,
    )

    t0 = time.time()
    result = train_within_species(model, train_ds, val_ds, train_cfg)
    elapsed = time.time() - t0
    logger.info("Final: %s   total=%.1fs", result["final"], elapsed)

    summary = {
        "args": vars(args) | {"out_dir": str(args.out_dir),
                               "train_cache": str(args.train_cache),
                               "val_cache": str(args.val_cache)},
        "n_training_identities": n_training_identities,
        "training_species": train_names,
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds),
        "trainable_params_total": result["n_trainable_params"],
        "decoder_params": model.n_decoder_params(),
        "history": result["history"],
        "final": result["final"],
        "best": result["best"],
        "elapsed_s": elapsed,
    }
    out = args.out_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", out)

    ckpt_path = args.out_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "train_species_names": train_names,
        "global_id_to_local": remap,
    }, ckpt_path)
    logger.info("Saved checkpoint -> %s", ckpt_path)

    # CSV row for results/
    csv_row = {
        "experiment": "within_species_pilot",
        "ablation": "no_id_token" if args.no_id_token else "iti_full",
        "seed": args.seed,
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds),
        "n_trainable_params": result["n_trainable_params"],
        "decoder_params": model.n_decoder_params(),
        "rmse_norm": result["final"]["rmse_norm"],
        "pck_at_005": result["final"]["pck@0.05"],
        "pck_at_010": result["final"]["pck@0.10"],
        "pck_at_020": result["final"]["pck@0.20"],
        "rmse_norm_ci95_lo": result["final"].get("rmse_norm_ci95_lo"),
        "rmse_norm_ci95_hi": result["final"].get("rmse_norm_ci95_hi"),
        "elapsed_s": elapsed,
    }
    csv_path = ROOT / "results" / "within_species_pilot.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a") as f:
        if write_header:
            f.write(",".join(csv_row.keys()) + "\n")
        f.write(",".join(str(csv_row[k]) for k in csv_row.keys()) + "\n")
    logger.info("Appended row to %s", csv_path)


if __name__ == "__main__":
    main()
