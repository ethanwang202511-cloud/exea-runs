"""Precompute DINOv2-base patch features for AP-10K crops.

Architecture.md §1.4: backbone features are computed ONCE and cached on
disk; all subsequent training uses the cached features. This is the
design hinge that makes the budget feasible.

Usage::

    PYTHONPATH=src python3 scripts/precompute_features.py \\
        --ap10k-root data/ap10k/ap-10k \\
        --split-file ap10k-train-split1.json \\
        --out-dir data/cache/dinov2_base_ap10k_train_split1 \\
        --species-filter top5    # or "all", or comma-separated list

The cache layout is::

    out_dir/
      meta.json            # list of records {annot_id, identity_id, species, vis, kp, crop_*, image_id, bbox_diag_orig}
      features.bin         # raw float16 buffer (N, 256, 768)
      features.shape.json  # shape + dtype metadata
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.ap10k import AP10KSplit
from iti.data.pose_dataset import AP10KPoseDataset
from iti.model.frozen_backbone import FrozenDINOv2Base, DINOV2_BASE_GRID, DINOV2_BASE_HIDDEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_features")


def collate(batch: list[dict]) -> dict:
    out = {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "identity_id": torch.tensor([b["identity_id"] for b in batch], dtype=torch.long),
        "keypoints_xy": torch.stack([b["keypoints_xy"] for b in batch]),
        "vis": torch.stack([b["vis"] for b in batch]),
    }
    out["meta"] = [
        {
            "annot_id": b["annot_id"], "image_id": b["image_id"],
            "species_name": b["species_name"], "identity_id": b["identity_id"],
            "bbox_diag_orig": b["bbox_diag_orig"],
            "crop_x0": b["crop_x0"], "crop_y0": b["crop_y0"], "crop_side": b["crop_side"],
            "keypoints_xy": b["keypoints_xy"].tolist(),
            "vis": b["vis"].tolist(),
        }
        for b in batch
    ]
    return out


def parse_species_filter(spec: str, split: AP10KSplit) -> set[int]:
    if spec == "all":
        return {i for i in range(54)}     # all 54 AP-10K species
    if spec == "top5":
        ann_by_sp = split.annotations_by_species()
        top = sorted(ann_by_sp.items(), key=lambda kv: -len(kv[1]))[:5]
        return {ann[0].category_id - 1 for _, ann in top}
    if spec == "top10":
        ann_by_sp = split.annotations_by_species()
        top = sorted(ann_by_sp.items(), key=lambda kv: -len(kv[1]))[:10]
        return {ann[0].category_id - 1 for _, ann in top}
    # comma-separated species names
    name_to_id = {meta.name: meta.identity_id for meta in split.identity_meta.values()}
    out: set[int] = set()
    for name in spec.split(","):
        name = name.strip()
        if name not in name_to_id:
            raise SystemExit(f"unknown species: {name}")
        out.add(name_to_id[name])
    return out


def select_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" or (prefer == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ap10k-root", type=Path, default="data/ap10k/ap-10k")
    p.add_argument("--split-file", type=str, default="ap10k-train-split1.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--species-filter", type=str, default="top5")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0, help="limit N items for quick test")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    split = AP10KSplit(args.ap10k_root, args.split_file)
    species_ids = parse_species_filter(args.species_filter, split)
    logger.info("Species filter: %s -> %d species", args.species_filter, len(species_ids))
    ds = AP10KPoseDataset(split, identity_ids_filter=species_ids)
    if args.limit:
        ds.items = ds.items[: args.limit]
    logger.info("Dataset size: %d items", len(ds))

    device = select_device(args.device)
    logger.info("Device: %s", device)
    backbone = FrozenDINOv2Base().to(device)
    backbone.eval()

    n = len(ds)
    n_patches = DINOV2_BASE_GRID * DINOV2_BASE_GRID
    feat = np.lib.format.open_memmap(
        args.out_dir / "features.npy",
        mode="w+", dtype=np.float16, shape=(n, n_patches, DINOV2_BASE_HIDDEN),
    )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    meta_records: list[dict] = []
    written = 0
    t0 = time.time()
    last_log = t0
    for bi, batch in enumerate(loader):
        x = batch["pixel_values"].to(device, non_blocking=True)
        with torch.no_grad():
            patches = backbone.patch_features(x)            # (B, 256, 768)
        patches = patches.to(torch.float16).cpu().numpy()
        b = patches.shape[0]
        feat[written : written + b] = patches
        meta_records.extend(batch["meta"])
        written += b
        if time.time() - last_log > 5.0:
            elapsed = time.time() - t0
            rate = written / elapsed
            eta = (n - written) / max(rate, 1e-6)
            logger.info("batch %4d/%d  written %d/%d  rate %.1f/s  ETA %.0fs",
                        bi + 1, len(loader), written, n, rate, eta)
            last_log = time.time()

    feat.flush()
    del feat
    (args.out_dir / "features.shape.json").write_text(json.dumps({
        "shape": [n, n_patches, DINOV2_BASE_HIDDEN], "dtype": "float16",
        "filename": "features.npy", "n_patches_grid": DINOV2_BASE_GRID,
    }))
    (args.out_dir / "meta.json").write_text(json.dumps({
        "split_file": args.split_file,
        "ap10k_root": str(args.ap10k_root),
        "species_filter": args.species_filter,
        "device": str(device),
        "n_items": n,
        "records": meta_records,
    }))
    elapsed = time.time() - t0
    logger.info("DONE: %d items in %.1fs (%.1f/s) -> %s",
                n, elapsed, n / max(elapsed, 1e-6), args.out_dir)


if __name__ == "__main__":
    main()
