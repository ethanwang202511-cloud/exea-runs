"""BehaviorFM (Identity-Token Adaptation / ITA) job wiring for the exea-runs
unattended harness. `exea.orchestrator` calls `get_jobs()` and drives every
Job's `list_units()` / `run_unit()` fail-soft, time-boxed, in priority order.

Seven jobs, in reviewer-complaint priority order (lower runs first):
  P05 (5)  Precompute the DINOv2 train/val/base-pool feature caches P06 and
           P10/P15/P20/P30 consume, at the exact paths those jobs read (must
           run first).
  P06 (6)  Train the base within-species ITI checkpoint (id-token bank +
           decoder + interpolation head) that P10/P15/P20/P30 adapt from --
           without this, get_base_state_and_cfg() has nothing to load on the
           server (smoke's random-init bypass does not apply there) and
           those four jobs would DataUnavailable-skip forever.
  P10 (10) PEFT panel Tier A (cached-feature): ITA, decoder-FT, LoRA-decoder,
           BitFit-decoder, 8-species grid, multi-seed, paired-bootstrap CIs.
  P15 (15) Per-budget Pareto grid {768, 296K, 1.2M, 8.5M} x 8 species x seeds
           (fills the previously-unreported 1.2M point).
  P20 (20) Decoder-FT comparator sensitivity sweep: lr x steps x init x seed
           on fox + 2 more species (replaces the single 300-step comparator).
  P30 (30) Inverted-U species-expansion: 8 held-out species, distance metric
           (WasserstainScout) vs adaptation-gain, quadratic-vs-linear fit
           with bootstrap CI on the quadratic coefficient.
  P40 (40) / P41 (41): cross-backbone feature precompute + Tier-A ITA, and
           cross-backbone Tier-B PEFT (LoRA-on-backbone, AdaptFormer, IA3,
           VPT) across {DINOv2, DINOv1, MAE, CLIP, EVA-02}. Tier-B always
           re-runs the full backbone forward — never fed cached features.
           This is the heaviest job (real ViT forward passes); EVA-02
           soft-skips wherever timm is unavailable.

Every run_unit is idempotent (skip-if-results-exist, checked first thing),
reseeds via exea.util.set_seed(unit['seed']), uses ctx.device, and raises
exea.staging.DataUnavailable (never a bare exception) for any missing
checkpoint / feature cache / raw AP-10K data / optional heavy dependency
(peft, timm) so the orchestrator soft-skips that unit rather than failing
the whole run.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BFM_ROOT = Path(__file__).resolve().parent
_SRC = str(_BFM_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from exea.jobs import Job  # noqa: E402

from behaviorfm import runners  # noqa: E402


def get_jobs() -> list[Job]:
    return [
        Job(
            job_id="behaviorfm.p05_precompute_dinov2",
            project="behaviorfm",
            priority=5,
            list_units=runners.p05_list_units,
            run_unit=runners.p05_run_unit,
            est_unit_seconds=240.0,
            description=(
                "P05: precompute the DINOv2-base feature caches P06 and "
                "P10/P15/P20/P30 require, at the exact staging.cache_dir_for() "
                "paths those jobs read (SPECIES_TO_TRAIN_CACHE groups, the "
                "shared val-split1-all cache, and the base-pool cache for P06's "
                "pretraining). Must run before them (lower priority number)."
            ),
            tags=["precompute", "prerequisite"],
        ),
        Job(
            job_id="behaviorfm.p06_train_base_ita",
            project="behaviorfm",
            priority=6,
            list_units=runners.p06_list_units,
            run_unit=runners.p06_run_unit,
            est_unit_seconds=600.0,
            description=(
                "P06: train the base within-species ITI checkpoint (id-token "
                "bank + decoder + interpolation head) at DEFAULT_CKPT, over the "
                "P05 base-pool cache (every AP-10K species except the SPECIES_8 "
                "held-out grid). Prerequisite for P10/P15/P20/P30, which load "
                "this exact checkpoint via peft_panel.get_base_state_and_cfg()."
            ),
            tags=["pretrain", "prerequisite"],
        ),
        Job(
            job_id="behaviorfm.p10_peft_panel_tier_a",
            project="behaviorfm",
            priority=10,
            list_units=runners.p10_list_units,
            run_unit=runners.p10_run_unit,
            est_unit_seconds=20.0,
            description=(
                "P10: Tier-A (cached-feature) PEFT panel — ITA / decoder-FT / "
                "LoRA-on-decoder / BitFit-on-decoder — on the 8-species AP-10K "
                "held-out grid, multi-seed, with paired-bootstrap CIs."
            ),
            tags=["tier-a", "peft", "reviewer-priority-1"],
        ),
        Job(
            job_id="behaviorfm.p15_pareto_budget_grid",
            project="behaviorfm",
            priority=15,
            list_units=runners.p15_list_units,
            run_unit=runners.p15_run_unit,
            est_unit_seconds=20.0,
            description=(
                "P15: per-budget Pareto grid {768, 296K, 1.2M, 8.5M} params x "
                "8 species x seeds; fills the previously-unreported 1.2M point."
            ),
            tags=["tier-a", "pareto"],
        ),
        Job(
            job_id="behaviorfm.p20_decoder_ft_sensitivity_sweep",
            project="behaviorfm",
            priority=20,
            list_units=runners.p20_list_units,
            run_unit=runners.p20_run_unit,
            est_unit_seconds=60.0,
            description=(
                "P20: decoder-FT comparator sensitivity sweep (lr x step-budget "
                "x {random-init, source-init}) on fox + 2 more species."
            ),
            tags=["tier-a", "comparator-sweep"],
        ),
        Job(
            job_id="behaviorfm.p30_inverted_u_species_expansion",
            project="behaviorfm",
            priority=30,
            list_units=runners.p30_list_units,
            run_unit=runners.p30_run_unit,
            est_unit_seconds=30.0,
            description=(
                "P30: inverted-U species expansion (3 -> 8 held-out species); "
                "quadratic-vs-linear fit of adaptation gain vs distance-to-"
                "training-pool, with bootstrap CI on the quadratic coefficient."
            ),
            tags=["tier-a", "distance-curve"],
        ),
        Job(
            job_id="behaviorfm.p40_cross_backbone_precompute_ita",
            project="behaviorfm",
            priority=40,
            list_units=runners.p40_list_units,
            run_unit=runners.p40_run_unit,
            est_unit_seconds=180.0,
            description=(
                "P40: cross-backbone feature precompute (DINOv2/DINOv1/MAE/"
                "CLIP/EVA-02) + ITA-on-cache (Tier A) per backbone x species."
            ),
            tags=["cross-backbone", "precompute", "tier-a"],
        ),
        Job(
            job_id="behaviorfm.p41_tier_b_peft",
            project="behaviorfm",
            priority=41,
            list_units=runners.p41_list_units,
            run_unit=runners.p41_run_unit,
            est_unit_seconds=300.0,
            description=(
                "P41: cross-backbone Tier-B PEFT (LoRA-on-backbone(Q,V), "
                "AdaptFormer, IA3, VPT) — re-runs the full frozen ViT forward "
                "every step; structurally cannot be fed cached features. "
                "Heaviest job in the suite."
            ),
            tags=["cross-backbone", "tier-b", "peft", "heaviest"],
        ),
    ]
