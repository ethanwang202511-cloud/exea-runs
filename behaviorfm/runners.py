"""Per-job unit builders / runners for the ITA experiment suite (P10-P41).

Each `pXX_list_units()` / `pXX_run_unit()` pair backs one `exea.jobs.Job` in
behaviorfm/jobs.py. Every `run_unit` is idempotent via a per-unit JSON
snapshot under `results_dir/behaviorfm/<subdir>/<unit_id>.json` (checked
FIRST, before any computation); after a successful compute the whole
subdir's `summary.csv` (and, where relevant, a paired-bootstrap or
quadratic-fit summary) is rebuilt from ALL currently-present unit JSONs, so
resumed / partially-completed multi-day runs always yield an internally
consistent, up-to-date table (no stale or duplicated rows).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_BFM_ROOT = Path(__file__).resolve().parent
_SRC = str(_BFM_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exea.staging import DataUnavailable                              # noqa: E402
from exea.util import set_seed, atomic_write_json                     # noqa: E402

from iti.model.iti import ITIModel, ITIConfig                          # noqa: E402
from iti.train.trainer import (                                        # noqa: E402
    TrainConfig, evaluate, coord_loss, train_within_species,
)

from behaviorfm import staging, backbones, peft_panel                  # noqa: E402

logger = logging.getLogger("behaviorfm.runners")

# --------------------------------------------------------------------------- #
# Shared constants (copied verbatim from scripts/run_peft_baselines.py so the
# species<->cache mapping matches the vendored panel exactly).
# --------------------------------------------------------------------------- #

SPECIES_8 = [
    "alouatta", "chimpanzee", "elephant", "monkey",
    "gorilla", "rabbit", "fox", "panther",
]

SPECIES_TO_TRAIN_CACHE = {
    "fox":        "dinov2_base_ap10k_train_split1_distance_sweep",
    "monkey":     "dinov2_base_ap10k_train_split1_distance_sweep",
    "rabbit":     "dinov2_base_ap10k_train_split1_distance_sweep",
    "panther":    "dinov2_base_ap10k_train_split1_distance_sweep",
    "alouatta":   "dinov2_base_ap10k_train_split1_alouatta",
    "elephant":   "dinov2_base_ap10k_train_split1_elephant",
    "gorilla":    "dinov2_base_ap10k_train_split1_gorilla",
    "chimpanzee": "dinov2_base_ap10k_train_split1_chimpanzee",
}

DEFAULT_CKPT = _BFM_ROOT / "results" / "q2_within_species" / "iti_xattn_aux01_seed0" / "model.pt"
DEFAULT_VAL_CACHE_NAME = "dinov2_base_ap10k_val_split1_all"

# Training-pool cache for P06 (base checkpoint pretraining): ALL AP-10K
# species from ap10k-train-split1.json EXCEPT the 8 held-out ones above (so
# the base within-species model never trains on the species P10-P30 later
# adapt to zero-shot/few-shot -- same held-out discipline
# scripts/run_within_species_pilot.py's fixed 5-species list was going for,
# generalized to the full non-held-out pool). Built by P05 (priority 5,
# before P06's priority 6) via the SAME staging.cache_dir_for(name) function
# P06 reads with, so the two paths cannot drift (mirrors the
# SPECIES_TO_TRAIN_CACHE <-> get_species_batches() discipline above).
BASE_POOL_CACHE_NAME = "dinov2_base_ap10k_train_split1_basepool"
BASE_POOL_SPECIES_SMOKE = ["dog", "sheep", "cat", "antelope", "zebra"]

K_SHOT = 5
ADAPT_LR = 1.0
N_ADAPT_STEPS_FULL = 50
N_ADAPT_STEPS_SMOKE = 5

BACKBONE_NAMES = ["dinov2", "dino_v1", "mae", "clip", "eva02"]
CROSS_BACKBONE_SPECIES = ["fox", "alouatta", "elephant"]  # bounded subset; heaviest job


def _smoke() -> bool:
    return staging.is_smoke()


# --------------------------------------------------------------------------- #
# Generic per-unit JSON / CSV persistence
# --------------------------------------------------------------------------- #

def _job_dir(results_dir: Path, subdir: str) -> Path:
    return results_dir / "behaviorfm" / subdir


def unit_result_path(results_dir: Path, subdir: str, unit_id: str) -> Path:
    return _job_dir(results_dir, subdir) / f"{unit_id}.json"


def load_existing_unit_result(results_dir: Path, subdir: str, unit_id: str) -> Optional[dict]:
    p = unit_result_path(results_dir, subdir, unit_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        logger.warning("[idempotency] corrupt result at %s; recomputing", p)
        return None


def save_unit_result(results_dir: Path, subdir: str, unit_id: str, row: dict) -> Path:
    p = unit_result_path(results_dir, subdir, unit_id)
    atomic_write_json(p, row)
    return p


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in columns) + "\n")
    os.replace(tmp, path)


def _load_all_unit_rows(results_dir: Path, subdir: str) -> list[dict]:
    dir_path = _job_dir(results_dir, subdir)
    rows = []
    if not dir_path.exists():
        return rows
    for p in sorted(dir_path.glob("*.json")):
        if p.stem.startswith("_"):
            continue
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            continue
    return rows


def rebuild_csv(results_dir: Path, subdir: str, columns: list[str]) -> Path:
    rows = _load_all_unit_rows(results_dir, subdir)
    out_csv = _job_dir(results_dir, subdir) / "summary.csv"
    _write_csv(out_csv, rows, columns)
    return out_csv


def write_paired_bootstrap_summary(
    results_dir: Path, subdir: str, anchor_method: str,
    group_key: str, method_key: str, seed_key: str, metric_key: str,
) -> Path:
    """Reviewer-facing table: for each group (usually species), the paired
    (same-seed) bootstrap CI of metric(method) - metric(anchor). Recomputed
    from scratch on every call from whatever unit JSONs currently exist, so
    it is always internally consistent even mid-resume."""
    rows = _load_all_unit_rows(results_dir, subdir)
    by_group: dict = {}
    for r in rows:
        if group_key not in r or method_key not in r or seed_key not in r or metric_key not in r:
            continue
        g, m, s, v = r[group_key], r[method_key], r[seed_key], r[metric_key]
        by_group.setdefault(g, {}).setdefault(m, {})[s] = v

    out_rows = []
    for g, by_method in by_group.items():
        anchor = by_method.get(anchor_method, {})
        if not anchor:
            continue
        for m, seed_vals in by_method.items():
            if m == anchor_method:
                continue
            common = sorted(set(anchor) & set(seed_vals))
            if len(common) < 2:
                continue
            a = np.array([seed_vals[s] for s in common])
            b = np.array([anchor[s] for s in common])
            mean_diff, lo, hi = peft_panel.paired_bootstrap_diff(a, b, seed=0)
            out_rows.append({
                group_key: g, "method": m, "anchor": anchor_method,
                "n_paired_seeds": len(common),
                f"mean_{metric_key}_diff_vs_anchor": mean_diff,
                "diff_ci95_lo": lo, "diff_ci95_hi": hi,
            })
    cols = [group_key, "method", "anchor", "n_paired_seeds",
            f"mean_{metric_key}_diff_vs_anchor", "diff_ci95_lo", "diff_ci95_hi"]
    out_path = _job_dir(results_dir, subdir) / "paired_bootstrap_summary.csv"
    _write_csv(out_path, out_rows, cols)
    return out_path


# --------------------------------------------------------------------------- #
# Cached-feature batch acquisition (real cache -> synthetic fallback)
# --------------------------------------------------------------------------- #

def get_species_batches(
    species: str, k_shot: int, n_val: int, device, smoke: bool,
) -> tuple[dict, list[dict], int]:
    """Return (train_batch, val_batches, gid) in the collate_cached schema,
    from the real feature cache if present, else (smoke only) synthetic."""
    gid_fallback = SPECIES_8.index(species) if species in SPECIES_8 else 0
    train_cache_name = SPECIES_TO_TRAIN_CACHE.get(species)
    if train_cache_name is not None:
        train_cache_dir = staging.cache_dir_for(train_cache_name)
        val_cache_dir = staging.cache_dir_for(DEFAULT_VAL_CACHE_NAME)
        if staging.cache_is_valid(train_cache_dir) and staging.cache_is_valid(val_cache_dir):
            from iti.data.feature_cache import CachedFeatureDataset, collate_cached
            from torch.utils.data import DataLoader
            train_full = CachedFeatureDataset(train_cache_dir)
            name_to_gid = {r["species_name"]: r["identity_id"] for r in train_full.records}
            gid = name_to_gid.get(species)
            if gid is not None:
                train_held = CachedFeatureDataset(train_cache_dir, identity_subset={gid})
                val_held = CachedFeatureDataset(val_cache_dir, identity_subset={gid})
                if len(train_held) >= k_shot and len(val_held) > 0:
                    train_batch = next(iter(DataLoader(
                        train_held, batch_size=len(train_held), shuffle=False, collate_fn=collate_cached)))
                    val_batches = list(DataLoader(
                        val_held, batch_size=32, shuffle=False, collate_fn=collate_cached))
                    return train_batch, val_batches, gid
    if not smoke:
        raise DataUnavailable(f"feature cache missing/insufficient for species={species!r}")
    seed_base = gid_fallback * 97 + 1
    train_batch = staging.synth_cached_batch(
        n=max(k_shot, 8), gid=gid_fallback, species_name=species, seed=seed_base)
    val_batches = [staging.synth_cached_batch(
        n=n_val, gid=gid_fallback, species_name=species, seed=seed_base + 1)]
    return train_batch, val_batches, gid_fallback


def get_species_dataset_held(
    species: str, cache_dir: Optional[Path], smoke: bool, n_synth: int, seed: int,
):
    """Return (Dataset, local_id) with the held species remapped to local id
    0 (matches scripts/run_decoder_ft_from_init.py's REMAP_LOCAL convention),
    real cache if valid, else (smoke only) SyntheticCachedDataset."""
    if cache_dir is not None and staging.cache_is_valid(cache_dir):
        from iti.data.feature_cache import CachedFeatureDataset
        full = CachedFeatureDataset(cache_dir)
        name_to_gid = {r["species_name"]: r["identity_id"] for r in full.records}
        gid = name_to_gid.get(species)
        if gid is not None:
            held = CachedFeatureDataset(cache_dir, identity_subset={gid}, identity_remap={gid: 0})
            if len(held) > 0:
                return held, 0
    if not smoke:
        raise DataUnavailable(f"feature cache missing/insufficient for species={species!r}")
    return staging.SyntheticCachedDataset(n=n_synth, gid=0, species_name=species, seed=seed), 0


# --------------------------------------------------------------------------- #
# P05: precompute the DINOv2 feature caches P10/P15/P20/P30 require.
#
# ORDERING FIX: P10/P15/P20/P30 (priority 10-30) call get_species_batches() /
# get_species_dataset_held(), which `require` a real DINOv2 cache at
# staging.cache_dir_for(SPECIES_TO_TRAIN_CACHE[species]) /
# staging.cache_dir_for(DEFAULT_VAL_CACHE_NAME) on the server (EXEA_SMOKE
# unset). Nothing at a lower priority number ever built those caches — P40's
# precompute writes to a DIFFERENT, backbone-namespaced layout
# (_cross_backbone_cache_dir), not the plain DINOv2 layout P10-P30 read. So
# on the server P10-P30 would DataUnavailable-skip forever. P05 (priority 5,
# runs first) builds exactly the caches P10-P30 consume, at the EXACT SAME
# paths: both P05 and get_species_batches()/get_species_dataset_held() derive
# their cache directory from the SAME module-level constants
# (SPECIES_TO_TRAIN_CACHE, DEFAULT_VAL_CACHE_NAME) via the SAME
# staging.cache_dir_for(name) function, so the paths cannot drift apart.
# --------------------------------------------------------------------------- #

P05_SUBDIR = "p05_precompute_dinov2"
P05_COLUMNS = ["job", "split", "cache_name", "species", "status", "cache_dir"]


def _invert_train_cache_groups() -> dict[str, list[str]]:
    """cache_name -> sorted list of species that share it, derived from
    SPECIES_TO_TRAIN_CACHE so P05's build targets can never drift from what
    get_species_batches()/get_species_dataset_held() actually read."""
    groups: dict[str, set] = {}
    for species, cache_name in SPECIES_TO_TRAIN_CACHE.items():
        groups.setdefault(cache_name, set()).add(species)
    return {k: sorted(v) for k, v in groups.items()}


def p05_list_units() -> list[dict]:
    units = []
    for cache_name, species_list in sorted(_invert_train_cache_groups().items()):
        units.append({
            "id": f"train__{cache_name}", "split": "train",
            "cache_name": cache_name, "species": species_list,
        })
    units.append({
        "id": f"val__{DEFAULT_VAL_CACHE_NAME}", "split": "val",
        "cache_name": DEFAULT_VAL_CACHE_NAME, "species": None,  # None = all species
    })
    # P06 (priority 6, base checkpoint pretraining) needs a TRAIN-split cache
    # spanning the non-held-out species pool -- build it here (idempotent,
    # skip-if-valid like every other P05 target) rather than in P06 itself,
    # keeping "P05 builds every DINOv2 cache" a single invariant.
    units.append({
        "id": f"train__{BASE_POOL_CACHE_NAME}", "split": "train_pool",
        "cache_name": BASE_POOL_CACHE_NAME, "species": None,  # None = all-minus-excluded
        "exclude_species": SPECIES_8,
    })
    return units


def p05_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P05_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(0)
    device = ctx.device
    cache_name = unit["cache_name"]
    split = unit["split"]
    species_list = unit["species"]  # list[str] or None (== all species)
    # THE path P10/P15/P20/P30 will resolve for this exact cache_name.
    out_dir = staging.cache_dir_for(cache_name)

    if staging.cache_is_valid(out_dir):
        row = {"job": "p05_precompute_dinov2", "split": split, "cache_name": cache_name,
               "species": species_list, "status": "already_valid", "cache_dir": str(out_dir)}
        save_unit_result(ctx.results_dir, P05_SUBDIR, unit["id"], row)
        rebuild_csv(ctx.results_dir, P05_SUBDIR, P05_COLUMNS)
        return row

    if smoke:
        if split == "train_pool":
            species_for_synth = BASE_POOL_SPECIES_SMOKE
        else:
            species_for_synth = species_list if species_list is not None else SPECIES_8
        pairs = [(sp, SPECIES_8.index(sp) if sp in SPECIES_8 else i)
                 for i, sp in enumerate(species_for_synth)]
        n_per_species = 16 if split == "val" else 24
        seed = sum(ord(c) for c in cache_name) % 10_000  # deterministic, no PYTHONHASHSEED dependence
        # smoke synth path writes to the EXACT SAME real cache directory
        # get_species_batches()/get_species_dataset_held() will read, so P10+
        # exercise their real-cache code path even in smoke. Use the static
        # DINOv2 BackboneSpec (patch/grid/hidden) directly rather than
        # instantiating the real 86M-param backbone, keeping smoke instant.
        spec = backbones.BACKBONES["dinov2"]

        class _ShapeOnly:
            grid = spec.grid
            hidden = spec.hidden
            n_patches = spec.grid * spec.grid

        _synthesize_multi_species_cache(out_dir, _ShapeOnly(), pairs, n_per_species=n_per_species, seed=seed)
    else:
        handle = backbones.load_backbone("dinov2", device, offline=False)
        # "train_pool" reads from the SAME ap10k-train-split1.json as "train"
        # (it's just a differently-scoped species filter over that split);
        # only "val" reads the val split file.
        split_file = "ap10k-val-split1.json" if split == "val" else "ap10k-train-split1.json"
        split_kind = "val" if split == "val" else "train"  # for the ap10k_annotations_item() check
        species_filter = set(species_list) if species_list is not None else None
        _precompute_features_for_species_set(
            species_filter, handle, out_dir, device, split_file=split_file, split_kind=split_kind,
            exclude=unit.get("exclude_species"))

    row = {"job": "p05_precompute_dinov2", "split": split, "cache_name": cache_name,
           "species": species_list, "status": "built", "cache_dir": str(out_dir)}
    save_unit_result(ctx.results_dir, P05_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P05_SUBDIR, P05_COLUMNS)
    return row


# --------------------------------------------------------------------------- #
# P06: train the base within-species ITI checkpoint (id-token bank + decoder
# + interpolation head) that P10/P15/P20/P30 read via
# peft_panel.get_base_state_and_cfg(DEFAULT_CKPT, ...). Mirrors
# scripts/run_within_species_pilot.py's training procedure exactly
# (iti.train.trainer.train_within_species over cached DINOv2 features), but
# trains on BASE_POOL_CACHE_NAME (every non-held-out AP-10K species, built by
# P05) instead of a fixed 5-species list, and writes to the EXACT path
# get_base_state_and_cfg reads (DEFAULT_CKPT) so the write side (here) and
# read side (peft_panel.py) cannot drift -- same "derive both ends from one
# shared constant" discipline verified for P05 -> P10 above.
#
# ORDERING: priority 6, i.e. after P05 (5, builds BASE_POOL_CACHE_NAME +
# DEFAULT_VAL_CACHE_NAME) and before P10 (10, the first checkpoint reader).
# --------------------------------------------------------------------------- #

P06_SUBDIR = "p06_train_base_ita"
P06_COLUMNS = [
    "job", "status", "ckpt_path", "n_training_identities",
    "n_train_examples", "n_val_examples", "n_trainable_params",
    "final_rmse_norm", "elapsed_s",
]


def p06_list_units() -> list[dict]:
    # One checkpoint, one unit. DEFAULT_CKPT's own name
    # ("iti_xattn_aux01_seed0") pins this to seed 0.
    return [{"id": "iti_xattn_aux01_seed0", "seed": 0}]


def p06_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P06_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(unit["seed"])
    device = ctx.device
    ckpt_path = DEFAULT_CKPT  # the EXACT path peft_panel.get_base_state_and_cfg() reads

    # Idempotent by artifact, not just by our own per-unit JSON marker: a
    # checkpoint already on disk (e.g. from a prior day's partial run, or
    # placed manually) is trusted as-is and never retrained -- same
    # "trust the artifact over the marker" convention as P05's
    # staging.cache_is_valid() check. staging.checkpoint_item() already
    # existed for exactly this (min_bytes=1000 sanity floor) but was unused
    # until now.
    if staging.checkpoint_item(ckpt_path).present():
        row = {"job": "p06_train_base_ita", "status": "already_valid", "ckpt_path": str(ckpt_path)}
        save_unit_result(ctx.results_dir, P06_SUBDIR, unit["id"], row)
        rebuild_csv(ctx.results_dir, P06_SUBDIR, P06_COLUMNS)
        return row

    train_cache_dir = staging.cache_dir_for(BASE_POOL_CACHE_NAME)
    if not staging.cache_is_valid(train_cache_dir):
        raise DataUnavailable(
            f"base-pool training feature cache missing/invalid at {train_cache_dir}; "
            f"behaviorfm.p05_precompute_dinov2 (priority 5) must build it first")

    from iti.data.feature_cache import CachedFeatureDataset

    train_full = CachedFeatureDataset(train_cache_dir)
    name_to_global: dict[str, int] = {}
    for r in train_full.records:
        name_to_global.setdefault(r["species_name"], r["identity_id"])
    train_species_names = sorted(name_to_global)
    if len(train_species_names) < 2:
        raise DataUnavailable(
            f"base-pool cache {train_cache_dir} has fewer than 2 species "
            f"({train_species_names}); cannot train a multi-identity token bank")
    global_ids_training = [name_to_global[n] for n in train_species_names]
    remap = {gid: i for i, gid in enumerate(global_ids_training)}
    n_training_identities = len(remap)

    train_ds = CachedFeatureDataset(
        train_cache_dir, identity_subset=set(global_ids_training), identity_remap=remap)
    if len(train_ds) == 0:
        raise DataUnavailable(f"base-pool cache {train_cache_dir} has zero usable training examples")

    if smoke:
        # DEFAULT_VAL_CACHE_NAME's smoke synthesis only ever covers SPECIES_8
        # (the held-out grid, see p05_run_unit's smoke branch) -- it never
        # contains BASE_POOL_SPECIES_SMOKE, so there is no real disjoint val
        # split to draw from here in smoke. Reuse the train-pool cache itself
        # for a quick eval pass: smoke only needs to prove the training +
        # evaluation code paths run end-to-end and produce a loadable
        # checkpoint, not held-out rigor (every other smoke fallback in this
        # suite makes the same trade, e.g. p41's k-shot support/eval reuse).
        val_ds = CachedFeatureDataset(
            train_cache_dir, identity_subset=set(global_ids_training), identity_remap=remap)
    else:
        # Real path: AP-10K category ids are a fixed dataset-wide taxonomy,
        # stable across train/val split files, so `remap` (built from the
        # train-pool cache) applies as-is to the val cache too -- no separate
        # name_to_global lookup needed (mirrors
        # scripts/run_within_species_pilot.py).
        val_cache_dir = staging.cache_dir_for(DEFAULT_VAL_CACHE_NAME)
        if not staging.cache_is_valid(val_cache_dir):
            raise DataUnavailable(
                f"val feature cache missing/invalid at {val_cache_dir}; "
                f"behaviorfm.p05_precompute_dinov2 (priority 5) must build it first")
        val_ds = CachedFeatureDataset(
            val_cache_dir, identity_subset=set(global_ids_training), identity_remap=remap)
    if len(val_ds) == 0:
        raise DataUnavailable("base-pool val split has zero usable examples to evaluate on")

    if smoke:
        # Tiny but REAL architecture (matches peft_panel.smoke_iti_config():
        # same hidden=768/grid=16 as real DINOv2 so cached-feature shapes
        # match; small decoder for CPU speed) -- P06 must write an actual
        # checkpoint file even in smoke, so P10+ load it via the real
        # `ckpt_path.exists()` branch of get_base_state_and_cfg(), not the
        # random-init smoke bypass.
        cfg = ITIConfig(n_training_identities=n_training_identities,
                         decoder_layers=1, decoder_dim=96, heatmap_size=64,
                         interp_head_version="v1")
        n_steps, desired_bs, warmup = 20, 8, 4
    else:
        cfg = ITIConfig(n_training_identities=n_training_identities, interp_head_version="v1")
        n_steps, desired_bs, warmup = 800, 32, 50

    # drop_last=True inside train_within_species's DataLoader means a
    # batch_size > len(train_ds) would yield ZERO batches per epoch and spin
    # forever; clamp so at least one batch always exists.
    bs = max(1, min(desired_bs, len(train_ds)))
    warmup = max(1, min(warmup, max(1, n_steps // 4)))
    train_cfg = TrainConfig(n_steps=n_steps, batch_size=bs, lr=3e-4, device=str(device),
                             seed=unit["seed"], warmup_steps=warmup)

    # backbone=nn.Identity(): training runs entirely over cached features
    # (never touches model.backbone), same rationale as every other
    # ITIModel(...) construction in this file.
    model = ITIModel(cfg, backbone=nn.Identity()).to(device)

    t0 = time.time()
    result = train_within_species(model, train_ds, val_ds, train_cfg)
    elapsed = time.time() - t0

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": asdict(cfg),
        "train_species_names": train_species_names,
        "global_id_to_local": remap,
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)  # atomic: never leaves a half-written checkpoint at ckpt_path

    row = {
        "job": "p06_train_base_ita", "status": "trained", "ckpt_path": str(ckpt_path),
        "n_training_identities": n_training_identities,
        "n_train_examples": len(train_ds), "n_val_examples": len(val_ds),
        "n_trainable_params": result["n_trainable_params"],
        "final_rmse_norm": result["final"]["rmse_norm"], "elapsed_s": elapsed,
    }
    save_unit_result(ctx.results_dir, P06_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P06_SUBDIR, P06_COLUMNS)
    return row


# --------------------------------------------------------------------------- #
# P10: PEFT panel Tier A (cached-feature) on the 8-species grid, multi-seed
# --------------------------------------------------------------------------- #

P10_SUBDIR = "p10_peft_panel_tier_a"
P10_COLUMNS = [
    "tier", "job", "species", "method", "seed", "n_trainable_params", "gid",
    "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20", "n_keypoints_evaluated",
    "rmse_norm_ci95_lo", "rmse_norm_ci95_hi",
]


def p10_list_units() -> list[dict]:
    smoke = _smoke()
    species_list = SPECIES_8[:2] if smoke else SPECIES_8
    n_seeds = 2 if smoke else 10
    units = []
    for species in species_list:
        for method in peft_panel.TIER_A_METHODS:
            for seed in range(n_seeds):
                units.append({
                    "id": f"{species}__{method}__seed{seed}",
                    "species": species, "method": method, "seed": seed,
                })
    return units


def p10_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P10_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(unit["seed"])
    device = ctx.device

    state, cfg = peft_panel.get_base_state_and_cfg(DEFAULT_CKPT, device, smoke)
    train_batch, val_batches, gid = get_species_batches(
        unit["species"], k_shot=K_SHOT, n_val=(16 if smoke else 512), device=device, smoke=smoke)

    model, tok, extra_params, n_trainable = peft_panel.build_adapted_model_tier_a(
        state, cfg, device, unit["method"])
    sup_f, sup_kp, sup_vis = peft_panel.sample_support(train_batch, K_SHOT, unit["seed"], device)

    if unit["method"] == "ita":
        init_val = peft_panel.init_token_via_interp_head(model, sup_f, device)
        with torch.no_grad():
            tok.data.copy_(init_val)
        tok.requires_grad_(True)
        n_trainable = tok.numel()

    n_steps = N_ADAPT_STEPS_SMOKE if smoke else N_ADAPT_STEPS_FULL
    peft_panel.adapt_loop(model, tok, extra_params, sup_f, sup_kp, sup_vis,
                           n_adapt_steps=n_steps, adapt_lr=ADAPT_LR)
    metrics = peft_panel.eval_on_batches(model, tok, val_batches, device)

    row = {
        "tier": "A", "job": "p10_peft_panel_tier_a",
        "species": unit["species"], "method": unit["method"], "seed": unit["seed"],
        "n_trainable_params": n_trainable, "gid": gid,
        **metrics,
    }
    save_unit_result(ctx.results_dir, P10_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P10_SUBDIR, P10_COLUMNS)
    write_paired_bootstrap_summary(ctx.results_dir, P10_SUBDIR, anchor_method="ita",
                                    group_key="species", method_key="method",
                                    seed_key="seed", metric_key="rmse_norm")
    return row


# --------------------------------------------------------------------------- #
# P15: per-budget Pareto grid {768, 296K, 1.2M, 8.5M} x 8 species x seeds
# --------------------------------------------------------------------------- #

P15_SUBDIR = "p15_pareto_budget_grid"
P15_COLUMNS = [
    "tier", "job", "species", "variant", "nominal_budget", "n_trainable_params",
    "seed", "gid", "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20",
    "n_keypoints_evaluated", "rmse_norm_ci95_lo", "rmse_norm_ci95_hi",
]


def p15_list_units() -> list[dict]:
    smoke = _smoke()
    species_list = SPECIES_8[:2] if smoke else SPECIES_8
    n_seeds = 2 if smoke else 5
    units = []
    for species in species_list:
        for variant in peft_panel.PARETO_VARIANTS:
            for seed in range(n_seeds):
                units.append({
                    "id": f"{species}__{variant}__seed{seed}",
                    "species": species, "variant": variant, "seed": seed,
                })
    return units


def p15_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P15_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(unit["seed"])
    device = ctx.device

    state, cfg = peft_panel.get_base_state_and_cfg(DEFAULT_CKPT, device, smoke)
    train_batch, val_batches, gid = get_species_batches(
        unit["species"], k_shot=K_SHOT, n_val=(16 if smoke else 512), device=device, smoke=smoke)

    model, tok, extra_params, n_trainable = peft_panel.build_pareto_model(
        state, cfg, device, unit["variant"])
    sup_f, sup_kp, sup_vis = peft_panel.sample_support(train_batch, K_SHOT, unit["seed"], device)

    init_val = peft_panel.init_token_via_interp_head(model, sup_f, device)
    with torch.no_grad():
        tok.data.copy_(init_val)
    tok.requires_grad_(True)

    n_steps = N_ADAPT_STEPS_SMOKE if smoke else N_ADAPT_STEPS_FULL
    peft_panel.adapt_loop(model, tok, extra_params, sup_f, sup_kp, sup_vis,
                           n_adapt_steps=n_steps, adapt_lr=ADAPT_LR)
    metrics = peft_panel.eval_on_batches(model, tok, val_batches, device)

    row = {
        "tier": "A", "job": "p15_pareto_budget_grid",
        "species": unit["species"], "variant": unit["variant"],
        "nominal_budget": peft_panel.PARETO_NOMINAL_BUDGET[unit["variant"]],
        "n_trainable_params": n_trainable, "seed": unit["seed"], "gid": gid,
        **metrics,
    }
    save_unit_result(ctx.results_dir, P15_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P15_SUBDIR, P15_COLUMNS)
    write_paired_bootstrap_summary(ctx.results_dir, P15_SUBDIR, anchor_method="token_only",
                                    group_key="species", method_key="variant",
                                    seed_key="seed", metric_key="rmse_norm")
    return row


# --------------------------------------------------------------------------- #
# P20: decoder-FT comparator sensitivity sweep (lr x steps x init x seed)
# on fox + 2 more species, reusing iti.train.trainer.{evaluate,coord_loss}
# and mirroring scripts/run_decoder_ft_from_init.py's training loop exactly.
# --------------------------------------------------------------------------- #

P20_SUBDIR = "p20_decoder_ft_sensitivity_sweep"
P20_SPECIES = ["fox", "monkey", "rabbit"]
P20_COLUMNS = [
    "job", "species", "lr", "steps", "init", "seed", "n_trainable_params",
    "pre_rmse_norm", "post_rmse_norm", "rmse_reduction",
    "pre_pck@0.05", "post_pck@0.05",
]


def p20_list_units() -> list[dict]:
    smoke = _smoke()
    if smoke:
        lrs, steps_list, inits, seeds, species_list = [1e-4], [20], ["source_init", "random_init"], [0], ["fox"]
    else:
        lrs = [1e-5, 1e-4, 1e-3]
        steps_list = [100, 300, 800]
        inits = ["source_init", "random_init"]
        seeds = list(range(3))
        species_list = P20_SPECIES
    units = []
    for species in species_list:
        for lr in lrs:
            for steps in steps_list:
                for init in inits:
                    for seed in seeds:
                        units.append({
                            "id": f"{species}__lr{lr}__steps{steps}__{init}__seed{seed}",
                            "species": species, "lr": lr, "steps": steps,
                            "init": init, "seed": seed,
                        })
    return units


def p20_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P20_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(unit["seed"])
    device = ctx.device

    state, cfg = peft_panel.get_base_state_and_cfg(DEFAULT_CKPT, device, smoke)

    train_cache_name = SPECIES_TO_TRAIN_CACHE.get(unit["species"])
    train_cache_dir = staging.cache_dir_for(train_cache_name) if train_cache_name else None
    val_cache_dir = staging.cache_dir_for(DEFAULT_VAL_CACHE_NAME)
    n_synth = 48 if smoke else 0
    train_ds, _ = get_species_dataset_held(unit["species"], train_cache_dir, smoke, n_synth, seed=unit["seed"])
    val_ds, _ = get_species_dataset_held(unit["species"], val_cache_dir, smoke, max(16, n_synth // 2), seed=unit["seed"] + 1000)

    from iti.data.feature_cache import collate_cached
    from torch.utils.data import DataLoader
    bs = min(32, len(train_ds))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               collate_fn=collate_cached, drop_last=(len(train_ds) > bs))
    val_loader = DataLoader(val_ds, batch_size=min(32, len(val_ds)), shuffle=False, collate_fn=collate_cached)

    model = ITIModel(cfg, backbone=nn.Identity()).to(device)
    if unit["init"] == "source_init":
        model.load_state_dict(state["model_state_dict"])
    # else "random_init": keep the freshly-constructed random weights.

    with torch.no_grad():
        model.tokens.emb.weight.zero_()
    model.tokens.emb.weight.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    warmup = max(1, min(20, unit["steps"] // 4))
    cfg_train = TrainConfig(n_steps=unit["steps"], batch_size=bs, lr=unit["lr"],
                             device=str(device), seed=unit["seed"], warmup_steps=warmup)

    pre = evaluate(model, val_loader, cfg_train)

    opt = torch.optim.AdamW(trainable, lr=cfg_train.lr, weight_decay=0.01)

    def lr_at(s: int) -> float:
        if s < cfg_train.warmup_steps:
            return cfg_train.lr * (s + 1) / cfg_train.warmup_steps
        prog = (s - cfg_train.warmup_steps) / max(cfg_train.n_steps - cfg_train.warmup_steps, 1)
        return cfg_train.lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * prog)))

    step = 0
    while step < cfg_train.n_steps:
        for batch in train_loader:
            if step >= cfg_train.n_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            feat = batch["feature"].to(device)
            ids = batch["identity_id"].to(device)
            kp = batch["keypoints_xy"].to(device)
            vis = batch["vis"].to(device)
            model.train()
            pred = model.forward_from_features(feat, ids)
            loss = coord_loss(pred, kp, vis, out_size=cfg_train.out_size)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            step += 1

    final = evaluate(model, val_loader, cfg_train)
    pre_rmse = pre["rmse_norm"]
    reduction = (pre_rmse - final["rmse_norm"]) / pre_rmse if pre_rmse else float("nan")

    row = {
        "job": "p20_decoder_ft_sensitivity_sweep", "species": unit["species"],
        "lr": unit["lr"], "steps": unit["steps"], "init": unit["init"], "seed": unit["seed"],
        "n_trainable_params": n_trainable,
        "pre_rmse_norm": pre_rmse, "post_rmse_norm": final["rmse_norm"], "rmse_reduction": reduction,
        "pre_pck@0.05": pre["pck@0.05"], "post_pck@0.05": final["pck@0.05"],
    }
    save_unit_result(ctx.results_dir, P20_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P20_SUBDIR, P20_COLUMNS)
    return row


# --------------------------------------------------------------------------- #
# P30: inverted-U species expansion (3 -> 8+ held-out species). Distance
# metric reuses iti.eval.wasserstein.WasserstainScout exactly; adaptation
# gain reuses the P15 token_only (768-param) ITA path.
# --------------------------------------------------------------------------- #

P30_SUBDIR = "p30_inverted_u_species_expansion"
P30_COLUMNS = [
    "job", "species", "seed", "cos_sim_max_to_train_pool",
    "base_rmse_norm", "adapt_rmse_norm", "rmse_reduction",
]


def p30_list_units() -> list[dict]:
    smoke = _smoke()
    species_list = SPECIES_8[:2] if smoke else SPECIES_8
    n_seeds = 1 if smoke else 3
    units = []
    for species in species_list:
        for seed in range(n_seeds):
            units.append({"id": f"{species}__seed{seed}", "species": species, "seed": seed})
    return units


def _get_train_pool_features(smoke: bool) -> dict[int, np.ndarray]:
    val_cache_dir = staging.cache_dir_for(DEFAULT_VAL_CACHE_NAME)
    if staging.cache_is_valid(val_cache_dir):
        from iti.data.feature_cache import CachedFeatureDataset
        ds = CachedFeatureDataset(val_cache_dir)
        by_gid: dict[int, list[np.ndarray]] = {}
        for i in range(len(ds)):
            it = ds[i]
            if it["species_name"] in SPECIES_8:
                continue  # keep the held-out grid OUT of the reference pool
            by_gid.setdefault(int(it["global_identity_id"]), []).append(it["feature"].numpy())
        pool = {gid: np.stack(v) for gid, v in by_gid.items() if len(v) >= 4}
        if len(pool) >= 2:
            return pool
    if not smoke:
        raise DataUnavailable("no real training-species reference pool available for P30")
    pool = {}
    for i in range(5):
        b = staging.synth_cached_batch(n=8, gid=i, species_name=f"synth_train_{i}", seed=9000 + i)
        pool[i] = b["feature"].numpy()
    return pool


def _aggregate_and_fit_inverted_u(results_dir: Path) -> dict:
    rows = _load_all_unit_rows(results_dir, P30_SUBDIR)
    by_species: dict[str, list[dict]] = {}
    for r in rows:
        by_species.setdefault(r["species"], []).append(r)
    points = []
    for species, rs in by_species.items():
        xs = [r["cos_sim_max_to_train_pool"] for r in rs]
        ys = [r["rmse_reduction"] for r in rs]
        points.append({"species": species, "cos_sim_max": float(np.mean(xs)),
                        "rmse_reduction": float(np.mean(ys)), "n_seeds": len(rs)})
    if len(points) < 4:
        summary = {"n_species": len(points),
                    "note": "need >=4 species with >=1 completed seed each for a stable quadratic fit"}
    else:
        xs = np.array([p["cos_sim_max"] for p in points])
        ys = np.array([p["rmse_reduction"] for p in points])
        fit = peft_panel.quadratic_vs_linear_fit(xs, ys)
        summary = {"n_species": len(points), "per_species_mean": points, **fit}
    atomic_write_json(_job_dir(results_dir, P30_SUBDIR) / "_quadratic_fit_summary.json", summary)
    return summary


def p30_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P30_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(unit["seed"])
    device = ctx.device

    state, cfg = peft_panel.get_base_state_and_cfg(DEFAULT_CKPT, device, smoke)
    train_batch, val_batches, gid = get_species_batches(
        unit["species"], k_shot=K_SHOT, n_val=(16 if smoke else 512), device=device, smoke=smoke)

    from iti.eval.wasserstein import WasserstainScout
    train_pool = _get_train_pool_features(smoke)
    scout = WasserstainScout(train_pool)
    n_demo = min(16, train_batch["feature"].shape[0])
    novel_demo = train_batch["feature"][:n_demo].numpy()
    dist = scout.distance_to_hull_mean(novel_demo)
    cos_max = dist["cos_sim_max_to_train_species_mean"]

    # Zero-shot baseline: mean-of-train-tokens, no adaptation at all
    # (mirrors scripts/zero_shot_distance_curve.py).
    model0, _, _, _ = peft_panel.build_adapted_model_tier_a(state, cfg, device, "ita")
    with torch.no_grad():
        mean_tok = model0.tokens.emb.weight.detach().mean(dim=0)
    base_metrics = peft_panel.eval_on_batches(model0, mean_tok, val_batches, device)

    # ITA-adapted (token_only, matches the P15 768-param budget point).
    model1, tok1, extra1, _ = peft_panel.build_pareto_model(state, cfg, device, "token_only")
    sup_f, sup_kp, sup_vis = peft_panel.sample_support(train_batch, K_SHOT, unit["seed"], device)
    init_val = peft_panel.init_token_via_interp_head(model1, sup_f, device)
    with torch.no_grad():
        tok1.data.copy_(init_val)
    tok1.requires_grad_(True)
    n_steps = N_ADAPT_STEPS_SMOKE if smoke else N_ADAPT_STEPS_FULL
    peft_panel.adapt_loop(model1, tok1, extra1, sup_f, sup_kp, sup_vis,
                           n_adapt_steps=n_steps, adapt_lr=ADAPT_LR)
    adapt_metrics = peft_panel.eval_on_batches(model1, tok1, val_batches, device)

    base_rmse = base_metrics["rmse_norm"]
    adapt_rmse = adapt_metrics["rmse_norm"]
    reduction = (base_rmse - adapt_rmse) / base_rmse if base_rmse else float("nan")

    row = {
        "job": "p30_inverted_u_species_expansion", "species": unit["species"], "seed": unit["seed"],
        "cos_sim_max_to_train_pool": cos_max,
        "base_rmse_norm": base_rmse, "adapt_rmse_norm": adapt_rmse, "rmse_reduction": reduction,
    }
    save_unit_result(ctx.results_dir, P30_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P30_SUBDIR, P30_COLUMNS)
    _aggregate_and_fit_inverted_u(ctx.results_dir)
    return row


# --------------------------------------------------------------------------- #
# P40: cross-backbone feature precompute + ITA-on-cache (Tier A, per backbone)
# --------------------------------------------------------------------------- #

P40_SUBDIR = "p40_cross_backbone_precompute_ita"
P40_COLUMNS = [
    "tier", "job", "backbone", "species", "n_trainable_params",
    "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20", "n_keypoints_evaluated",
]


def p40_list_units() -> list[dict]:
    smoke = _smoke()
    # smoke: dinov2 (locally cached -> exercises the real backbone-forward
    # path) + eva02 (timm not installed locally -> exercises the soft-skip
    # path). Both are worth covering in a quick smoke run.
    backbones_list = ["dinov2", "eva02"] if smoke else BACKBONE_NAMES
    species_list = CROSS_BACKBONE_SPECIES[:1] if smoke else CROSS_BACKBONE_SPECIES
    return [{"id": f"{b}__{s}", "backbone": b, "species": s}
            for b in backbones_list for s in species_list]


def _cross_backbone_cache_dir(backbone: str, species: str) -> Path:
    return staging.cache_dir_for(f"{backbone}_ap10k_train_split1_{species}")


def _synthesize_multi_species_cache(
    out_dir: Path, handle: "backbones.BackboneHandle",
    species_gid_pairs: list[tuple[str, int]], n_per_species: int, seed: int,
) -> None:
    """Write a synthetic feature cache spanning MULTIPLE species into ONE
    cache directory. Mirrors the shape of real multi-species train caches
    (e.g. SPECIES_TO_TRAIN_CACHE's "distance_sweep" group, where several
    species share one cache file and are told apart at read time via
    CachedFeatureDataset(identity_subset={gid}))."""
    if staging.cache_is_valid(out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_per_species * len(species_gid_pairs)
    g = torch.Generator().manual_seed(seed)
    feat = torch.randn(n_total, handle.n_patches, handle.hidden, generator=g).to(torch.float16).numpy()
    np.save(out_dir / "features.npy", feat)
    (out_dir / "features.shape.json").write_text(json.dumps({
        "shape": list(feat.shape), "dtype": "float16",
        "filename": "features.npy", "n_patches_grid": handle.grid,
    }))
    kp = torch.rand(n_total, 17, 2, generator=g) * 224.0
    vis = (torch.rand(n_total, 17, generator=g) > 0.15).float()
    vis[:, 0] = 1.0
    diag = 80.0 + torch.rand(n_total, generator=g) * 220.0
    records = []
    idx = 0
    for species, gid in species_gid_pairs:
        for _ in range(n_per_species):
            records.append({
                "annot_id": idx, "image_id": idx, "species_name": species, "identity_id": gid,
                "keypoints_xy": kp[idx].tolist(), "vis": vis[idx].tolist(),
                "bbox_diag_orig": float(diag[idx]), "crop_side": 224.0, "crop_x0": 0.0, "crop_y0": 0.0,
            })
            idx += 1
    (out_dir / "meta.json").write_text(json.dumps({
        "split_file": "synthetic", "species_filter": [s for s, _ in species_gid_pairs],
        "n_items": n_total, "records": records,
    }))


def _synthesize_backbone_cache(out_dir: Path, handle: "backbones.BackboneHandle",
                                species: str, gid: int, n: int, seed: int) -> None:
    """Single-species convenience wrapper (used by P40)."""
    _synthesize_multi_species_cache(out_dir, handle, [(species, gid)], n_per_species=n, seed=seed)


def _precompute_features_for_species_set(
    species_filter: Optional[set], handle: "backbones.BackboneHandle", out_dir: Path, device,
    split_file: str, split_kind: str, batch_size: int = 16, exclude: Optional[set] = None,
) -> Path:
    """Generalizes scripts/precompute_features.py across backbones AND
    across an arbitrary species subset (or `species_filter=None` for ALL
    species in the split, matching precompute_features.py's
    `--species-filter all`): same crop pipeline (iti.data.ap10k /
    iti.data.pose_dataset), generalized patch-count/hidden-dim, and
    backbone-specific mean/std renormalization (CLIP gotcha). Skip-if-exists
    / resumable, per the task spec. Used by P05 (multi-species DINOv2
    train/val/base-pool caches) and P40 (single-species cross-backbone
    caches). `exclude` (species names) is subtracted AFTER `species_filter`
    is resolved -- used by P05's BASE_POOL_CACHE_NAME unit to build "every
    species except the SPECIES_8 held-out grid" (species_filter=None,
    exclude=SPECIES_8) so P06's base checkpoint never trains on the species
    P10-P30 later adapt to."""
    if staging.cache_is_valid(out_dir):
        return out_dir
    if not (staging.ap10k_images_item().present() and staging.ap10k_annotations_item(split_kind).present()):
        raise DataUnavailable(
            f"AP-10K raw data not staged; cannot precompute {handle.spec.name} ({split_file})")

    from iti.data.ap10k import AP10KSplit
    from iti.data.pose_dataset import AP10KPoseDataset
    from torch.utils.data import DataLoader

    split = AP10KSplit(staging.AP10K_ROOT, split_file)
    name_to_id = {m.name: m.identity_id for m in split.identity_meta.values()}
    if species_filter is None:
        species_ids = set(name_to_id.values())
    else:
        missing = set(species_filter) - set(name_to_id)
        if missing:
            raise DataUnavailable(f"species {sorted(missing)} not present in {split_file}")
        species_ids = {name_to_id[s] for s in species_filter}
    if exclude:
        species_ids = species_ids - {name_to_id[s] for s in exclude if s in name_to_id}
    ds = AP10KPoseDataset(split, identity_ids_filter=species_ids)
    if len(ds) == 0:
        raise DataUnavailable(
            f"no annotated AP-10K instances for species_filter={species_filter} "
            f"(exclude={sorted(exclude) if exclude else None}) in {split_file}")

    def collate(batch):
        return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "meta": batch}

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    n = len(ds)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_mm = np.lib.format.open_memmap(
        out_dir / "features.npy", mode="w+", dtype=np.float16,
        shape=(n, handle.n_patches, handle.hidden))
    records = []
    written = 0
    for batch in loader:
        x = batch["pixel_values"].to(device)
        if handle.spec.mean != backbones.IMAGENET_MEAN:
            x = backbones.renormalize(x, backbones.IMAGENET_MEAN, backbones.IMAGENET_STD,
                                       handle.spec.mean, handle.spec.std)
        patches = handle.patch_features(x, grad=False).to(torch.float16).cpu().numpy()
        b = patches.shape[0]
        feat_mm[written:written + b] = patches
        for item in batch["meta"]:
            records.append({
                "annot_id": item["annot_id"], "image_id": item["image_id"],
                "species_name": item["species_name"], "identity_id": item["identity_id"],
                "keypoints_xy": item["keypoints_xy"].tolist(), "vis": item["vis"].tolist(),
                "bbox_diag_orig": item["bbox_diag_orig"],
                "crop_x0": item["crop_x0"], "crop_y0": item["crop_y0"], "crop_side": item["crop_side"],
            })
        written += b
    feat_mm.flush()
    del feat_mm
    (out_dir / "features.shape.json").write_text(json.dumps({
        "shape": [n, handle.n_patches, handle.hidden], "dtype": "float16",
        "filename": "features.npy", "n_patches_grid": handle.grid,
    }))
    (out_dir / "meta.json").write_text(json.dumps({
        "split_file": split_file,
        "species_filter": sorted(species_filter) if species_filter else "all",
        "excluded_species": sorted(exclude) if exclude else None,
        "n_items": n, "records": records,
    }))
    return out_dir


def _precompute_backbone_features_real(
    species: str, handle: "backbones.BackboneHandle", out_dir: Path, device, batch_size: int = 16,
) -> Path:
    """Single-species convenience wrapper (used by P40)."""
    return _precompute_features_for_species_set(
        {species}, handle, out_dir, device,
        split_file="ap10k-train-split1.json", split_kind="train", batch_size=batch_size)


def p40_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P40_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(0)
    device = ctx.device
    backbone_name, species = unit["backbone"], unit["species"]

    try:
        handle = backbones.load_backbone(backbone_name, device, offline=smoke)
    except backbones.BackboneUnavailable as e:
        raise DataUnavailable(str(e))

    cache_dir = _cross_backbone_cache_dir(backbone_name, species)
    gid = SPECIES_8.index(species) if species in SPECIES_8 else 0
    if not staging.cache_is_valid(cache_dir):
        if smoke:
            _synthesize_backbone_cache(cache_dir, handle, species, gid, n=24, seed=gid + 1)
        else:
            _precompute_backbone_features_real(species, handle, cache_dir, device)

    cfg = ITIConfig(n_training_identities=5, hidden=handle.hidden, grid=handle.grid,
                     n_keypoints=17, decoder_layers=1, decoder_dim=96)
    model = ITIModel(cfg, backbone=nn.Identity()).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    from iti.data.feature_cache import CachedFeatureDataset, collate_cached
    from torch.utils.data import DataLoader
    ds = CachedFeatureDataset(cache_dir)
    if len(ds) < K_SHOT + 2:
        raise DataUnavailable(f"cross-backbone cache {cache_dir} has too few examples")
    train_batch = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_cached)))
    val_batches = [train_batch]  # small cache: reuse as both support pool and eval set

    sup_f, sup_kp, sup_vis = peft_panel.sample_support(train_batch, K_SHOT, seed=0, device=device)
    tok = nn.Parameter(model.tokens.emb.weight.detach().mean(dim=0).clone())
    init_val = peft_panel.init_token_via_interp_head(model, sup_f, device)
    with torch.no_grad():
        tok.data.copy_(init_val)
    tok.requires_grad_(True)

    n_steps = N_ADAPT_STEPS_SMOKE if smoke else N_ADAPT_STEPS_FULL
    peft_panel.adapt_loop(model, tok, [], sup_f, sup_kp, sup_vis, n_adapt_steps=n_steps, adapt_lr=ADAPT_LR)
    metrics = peft_panel.eval_on_batches(model, tok, val_batches, device)

    row = {
        "tier": "A", "job": "p40_cross_backbone_precompute_ita",
        "backbone": backbone_name, "species": species, "n_trainable_params": tok.numel(),
        **metrics,
    }
    save_unit_result(ctx.results_dir, P40_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P40_SUBDIR, P40_COLUMNS)
    return row


# --------------------------------------------------------------------------- #
# P41: cross-backbone Tier-B PEFT (LoRA-on-backbone, AdaptFormer, IA3, VPT) —
# re-runs the ViT forward every step; NEVER touches cached features.
# --------------------------------------------------------------------------- #

P41_SUBDIR = "p41_tier_b_peft"
P41_COLUMNS = [
    "tier", "job", "backbone", "species", "method", "seed", "n_trainable_params",
    "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20", "n_keypoints_evaluated",
]


def p41_list_units() -> list[dict]:
    smoke = _smoke()
    backbones_list = BACKBONE_NAMES[:1] if smoke else BACKBONE_NAMES
    species_list = CROSS_BACKBONE_SPECIES[:1] if smoke else CROSS_BACKBONE_SPECIES
    n_seeds = 1 if smoke else 3
    units = []
    for b in backbones_list:
        for s in species_list:
            for m in peft_panel.TIER_B_METHODS:
                for seed in range(n_seeds):
                    units.append({
                        "id": f"{b}__{s}__{m}__seed{seed}",
                        "backbone": b, "species": s, "method": m, "seed": seed,
                    })
    return units


def _get_species_pixel_batches(
    species: str, k_shot: int, n_val: int, smoke: bool, seed: int,
    spec: Optional["backbones.BackboneSpec"] = None,
) -> tuple[dict, list[dict]]:
    """AP10KPoseDataset (and staging.synth_pixel_batch) always bake in
    ImageNet mean/std normalization (see pose_dataset.py's _MEAN/_STD). Tier-B
    forwards these pixels through whatever backbone `spec` names -- CLIP was
    pretrained on its OWN mean/std (backbones.CLIP_MEAN/STD), not ImageNet's
    -- so, exactly like _precompute_features_for_species_set does for the
    Tier-A/P40 cache-precompute path (guarded by `spec.mean != IMAGENET_MEAN`
    below), pixels must be renormalized per-backbone before being handed to
    the backbone forward. Without this, CLIP silently receives mis-normalized
    inputs (no crash, just degraded/wrong Tier-B numbers for that backbone).
    """
    def _renorm(pixel_values: torch.Tensor) -> torch.Tensor:
        if spec is not None and spec.mean != backbones.IMAGENET_MEAN:
            return backbones.renormalize(
                pixel_values, backbones.IMAGENET_MEAN, backbones.IMAGENET_STD, spec.mean, spec.std)
        return pixel_values

    if not smoke:
        if staging.ap10k_images_item().present() and staging.ap10k_annotations_item("train").present():
            from iti.data.ap10k import AP10KSplit
            from iti.data.pose_dataset import AP10KPoseDataset
            split = AP10KSplit(staging.AP10K_ROOT, "ap10k-train-split1.json")
            name_to_id = {m.name: m.identity_id for m in split.identity_meta.values()}
            if species in name_to_id:
                ds = AP10KPoseDataset(split, identity_ids_filter={name_to_id[species]})
                if len(ds) >= k_shot + 4:
                    rng = np.random.default_rng(seed)
                    idx = rng.permutation(len(ds))
                    sup_idx = idx[:k_shot]
                    val_idx = idx[k_shot:k_shot + n_val]

                    def build(idxs):
                        items = [ds[int(i)] for i in idxs]
                        return {
                            "pixel_values": _renorm(torch.stack([it["pixel_values"] for it in items])),
                            "keypoints_xy": torch.stack([it["keypoints_xy"] for it in items]),
                            "vis": torch.stack([it["vis"] for it in items]),
                            "bbox_diag_orig": torch.tensor([it["bbox_diag_orig"] for it in items]),
                            "crop_side": torch.tensor([it["crop_side"] for it in items]),
                        }
                    return build(sup_idx), [build(val_idx)]
        raise DataUnavailable(f"AP-10K raw images/annotations not staged for Tier-B species={species!r}")
    sup = staging.synth_pixel_batch(n=k_shot, seed=seed)
    val = [staging.synth_pixel_batch(n=n_val, seed=seed + 1)]
    sup["pixel_values"] = _renorm(sup["pixel_values"])
    for v in val:
        v["pixel_values"] = _renorm(v["pixel_values"])
    return sup, val


def p41_run_unit(unit: dict, ctx) -> dict:
    existing = load_existing_unit_result(ctx.results_dir, P41_SUBDIR, unit["id"])
    if existing is not None:
        return existing

    smoke = _smoke()
    set_seed(unit["seed"])
    device = ctx.device

    try:
        handle = backbones.load_backbone(unit["backbone"], device, offline=smoke)
    except backbones.BackboneUnavailable as e:
        raise DataUnavailable(str(e))

    try:
        tb = peft_panel.build_tier_b(handle, unit["method"])
    except (backbones.BackboneUnavailable, peft_panel.DependencyUnavailable) as e:
        raise DataUnavailable(str(e))

    cfg = ITIConfig(n_training_identities=5, hidden=handle.hidden, grid=handle.grid,
                     n_keypoints=17, decoder_layers=1, decoder_dim=96)
    decoder_model = ITIModel(cfg, backbone=nn.Identity()).to(device)
    decoder_model.eval()
    for p in decoder_model.parameters():
        p.requires_grad_(False)
    tok = decoder_model.tokens.emb.weight.detach().mean(dim=0).clone()

    sup_batch, val_pixel_batches = _get_species_pixel_batches(
        unit["species"], k_shot=K_SHOT, n_val=(8 if smoke else 64), smoke=smoke, seed=unit["seed"],
        spec=handle.spec)

    n_steps = 3 if smoke else 20
    metrics = peft_panel.tier_b_adapt_and_eval(
        tb, decoder_model, tok,
        sup_batch["pixel_values"].to(device), sup_batch["keypoints_xy"].to(device),
        sup_batch["vis"].to(device), val_pixel_batches,
        n_adapt_steps=n_steps, adapt_lr=1e-3,
    )
    row = {
        "tier": "B", "job": "p41_tier_b_peft",
        "backbone": unit["backbone"], "species": unit["species"], "method": unit["method"],
        "seed": unit["seed"], "n_trainable_params": tb.n_trainable,
        **metrics,
    }
    save_unit_result(ctx.results_dir, P41_SUBDIR, unit["id"], row)
    rebuild_csv(ctx.results_dir, P41_SUBDIR, P41_COLUMNS)
    return row
