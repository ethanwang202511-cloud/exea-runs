"""InterveneFM job definitions for the exea unattended harness.

`get_jobs()` is the only entry point the orchestrator calls
(`intervenefm.jobs.get_jobs()`). Everything else here wires the vendored,
validated `src/` + `scripts/` code (via `intervenefm/runners.py` and
`intervenefm/pertformer.py`) into `exea.jobs.Job` objects.

Jobs (priority = run order, lower first):
  10  p10_cpa_cpg_audit        core CPA+CPG audit, multi-seed, per-dataset,
                                variant in {learned_CPA_baseline, CPG_SVD,
                                CPG_GO, CPG_linear}
  11  p10_cluster_bootstrap    cross-seed cluster-bootstrap CI tables for P10
  20  p20_pertformer_audit     PertFormer (pure nn.TransformerEncoderLayer)
                                attention audit, variant in {A, B}
  30  p30_gears_audit          published-checkpoint GEARS audit (fail-soft
                                optional -- list_units()=[] if torch_geometric
                                or the `gears` package is unavailable)
  40  p40_scgpt_audit          scGPT-Perturb audit (fail-soft optional --
                                list_units()=[] if scgpt is unavailable)

Only torch + numpy + pandas + scanpy are imported at module import time here
(via `runners`/`staging`), matching the harness's hard rule that jobs.py must
import cleanly even when torch_geometric / scgpt / cell_gears are absent.
Those heavy optional imports happen ONLY inside the P30/P40 functions, at the
point of use.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

INTERVENEFM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = INTERVENEFM_ROOT.parent
if str(INTERVENEFM_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERVENEFM_ROOT))

from exea.jobs import Job, RunContext  # noqa: E402
from exea.staging import DataUnavailable  # noqa: E402
from exea.util import set_seed, atomic_write_json, read_json  # noqa: E402

from src.cpa_cpg import compute_gene_identity_table  # noqa: E402
from src.data_norman import PerturbSeqDataset  # noqa: E402

from . import runners, staging  # noqa: E402
from .pertformer import PertFormer, PertFormerConfig  # noqa: E402

log = logging.getLogger("intervenefm.jobs")

DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


def _seeds() -> List[int]:
    return [0, 1] if runners.is_smoke() else [0, 1, 2]


# ============================================================================
# P10: core CPA+CPG audit, multi-seed, per-dataset
# ============================================================================
def _p10_out_dir(results_dir, dataset: str) -> Path:
    d = Path(results_dir) / "intervenefm" / "p10" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d


def p10_list_units() -> List[dict]:
    seeds = _seeds()
    units = []
    for dataset in runners.DATASETS:
        for variant in runners.VARIANTS:
            for seed in seeds:
                units.append({
                    "id": f"{dataset}__{variant}__seed{seed}",
                    "dataset": dataset, "variant": variant, "seed": seed,
                })
    return units


def p10_run_unit(unit: dict, ctx: RunContext) -> dict:
    dataset, variant, seed = unit["dataset"], unit["variant"], unit["seed"]
    set_seed(seed)

    out_dir = _p10_out_dir(ctx.results_dir, dataset)
    csv_path = out_dir / f"{variant}_seed{seed}.csv"
    summary_path = out_dir / f"{variant}_seed{seed}_summary.json"
    _cached = read_json(summary_path)
    if isinstance(_cached, dict):
        return _cached

    smoke = runners.is_smoke()
    cfg = runners.get_config(dataset, smoke)

    need_full_panel = variant != "learned_CPA_baseline"
    bundle = runners.load_bundle(dataset, cfg, seed, need_full_panel)
    if len(bundle["test_items"]) == 0:
        raise DataUnavailable(f"{dataset}/{variant}: no held-out test items produced by the split")

    model, _gene_id_table = runners.build_cpa_or_cpg_model(variant, bundle, cfg, seed)
    bundle.pop("adata_full", None)  # free the full-panel matrix once the identity table is built

    train_ds = PerturbSeqDataset(bundle["adata"], bundle["split"]["train_cells"], bundle["split"]["ctrl_cells"],
                                  bundle["vocab"], max_perts=2, seed=seed)
    if len(train_ds) == 0:
        raise DataUnavailable(f"{dataset}/{variant}: empty training set")
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    ckpt = runners.make_checkpointer(ctx, f"p10::{unit['id']}")
    runners.train_model(model, loader, ctx.device, epochs=cfg["epochs"], lr=cfg["lr"],
                         ckpt=ckpt, ctx=ctx, logger=ctx.logger)

    audit_df = runners.run_manual_audit(model, bundle, bundle["vocab"], seed, cfg["n_per_pair"],
                                         cfg["deg_k"], ctx.device)
    audit_df.to_csv(csv_path, index=False)

    n_boot = min(cfg["n_boot"], 500)  # per-unit (single-seed) stats; full n_boot happens in the aggregate job
    mode_table = runners.cluster_bootstrap_table(audit_df, n_boot=n_boot, seed=seed)
    gap_table = runners.gap_stats_table(audit_df, n_boot=n_boot, seed=seed)

    summary = {
        "dataset": dataset, "variant": variant, "seed": seed,
        "n_test_items": int(audit_df["item"].nunique()) if len(audit_df) else 0,
        "n_train_epochs": cfg["epochs"],
        "mode_means": {r["mode"]: float(r["mean"]) for _, r in mode_table.iterrows()},
        "gap_learned_minus": {r["mode"]: float(r["gap_mean"]) for _, r in gap_table.iterrows()},
        "csv": str(csv_path),
    }
    atomic_write_json(summary_path, summary)
    return summary


# ============================================================================
# P10 (aggregate): cross-seed cluster-bootstrap CI + effect-size tables.
# Reviewers complained results were vague ("around 90%") -- this is the full
# numeric table. Unit id embeds how many seeds are currently on disk, so the
# aggregate automatically recomputes (a fresh unit id) as more seeds land on
# later days, instead of freezing at a stale partial-seed-count forever once
# marked "done" by the manifest.
# ============================================================================
def p10_agg_list_units() -> List[dict]:
    seeds = _seeds()
    units = []
    for dataset in runners.DATASETS:
        for variant in runners.VARIANTS:
            in_dir = DEFAULT_RESULTS_DIR / "intervenefm" / "p10" / dataset
            n_found = sum(1 for s in seeds if (in_dir / f"{variant}_seed{s}_summary.json").exists())
            units.append({
                "id": f"{dataset}__{variant}__n{n_found}",
                "dataset": dataset, "variant": variant, "seeds": seeds,
            })
    return units


def p10_agg_run_unit(unit: dict, ctx: RunContext) -> dict:
    import pandas as pd

    dataset, variant, seeds = unit["dataset"], unit["variant"], unit["seeds"]
    out_dir = Path(ctx.results_dir) / "intervenefm" / "p10_agg" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{variant}_cluster_bootstrap.csv"
    gap_path = out_dir / f"{variant}_gap_stats.csv"
    summary_path = out_dir / f"{variant}_summary.json"
    cached = read_json(summary_path)
    if isinstance(cached, dict):
        if cached.get("n_seeds_used") == sum(
            1 for s in seeds if (Path(ctx.results_dir) / "intervenefm" / "p10" / dataset
                                  / f"{variant}_seed{s}.csv").exists()):
            return cached

    in_dir = Path(ctx.results_dir) / "intervenefm" / "p10" / dataset
    dfs, seeds_used = [], []
    for seed in seeds:
        f = in_dir / f"{variant}_seed{seed}.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["seed"] = seed
            dfs.append(d)
            seeds_used.append(seed)
    if not dfs:
        raise DataUnavailable(f"no completed P10 seeds yet for {dataset}/{variant}")
    big = pd.concat(dfs, ignore_index=True)

    cfg = runners.get_config(dataset, runners.is_smoke())
    mode_table = runners.cluster_bootstrap_table(big, n_boot=cfg["n_boot"], seed=0)
    gap_table = runners.gap_stats_table(big, n_boot=cfg["n_boot"], seed=0)
    mode_table.to_csv(table_path, index=False)
    gap_table.to_csv(gap_path, index=False)

    summary = {
        "dataset": dataset, "variant": variant,
        "n_seeds_used": len(seeds_used), "seeds_used": seeds_used,
        "mode_means": {r["mode"]: float(r["mean"]) for _, r in mode_table.iterrows()},
        "mode_ci": {r["mode"]: [float(r["ci_lower"]), float(r["ci_upper"])] for _, r in mode_table.iterrows()},
        "gaps": {r["mode"]: {"mean": float(r["gap_mean"]), "ci": [float(r["gap_ci_lower"]), float(r["gap_ci_upper"])],
                              "cohens_d": float(r["cohens_d"]), "wilcoxon_p": float(r["wilcoxon_p"])}
                 for _, r in gap_table.iterrows()},
        "cluster_bootstrap_csv": str(table_path),
        "gap_stats_csv": str(gap_path),
    }
    atomic_write_json(summary_path, summary)
    return summary


# ============================================================================
# P20: PertFormer attention audit
# ============================================================================
def p20_list_units() -> List[dict]:
    seeds = _seeds()
    units = []
    for dataset in runners.DATASETS:
        for variant in runners.PERTFORMER_VARIANTS:
            for seed in seeds:
                units.append({
                    "id": f"{dataset}__{variant}__seed{seed}",
                    "dataset": dataset, "variant": variant, "seed": seed,
                })
    return units


def p20_run_unit(unit: dict, ctx: RunContext) -> dict:
    dataset, variant, seed = unit["dataset"], unit["variant"], unit["seed"]
    set_seed(seed)

    out_dir = Path(ctx.results_dir) / "intervenefm" / "p20" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"pertformer_{variant}_seed{seed}.csv"
    dorm_path = out_dir / f"pertformer_{variant}_seed{seed}_dormancy.csv"
    summary_path = out_dir / f"pertformer_{variant}_seed{seed}_summary.json"
    _cached = read_json(summary_path)
    if isinstance(_cached, dict):
        return _cached

    smoke = runners.is_smoke()
    cfg = runners.get_config(dataset, smoke)

    need_full_panel = variant == "B"  # Variant B's q_pert MLP needs the gene-identity table
    bundle = runners.load_bundle(dataset, cfg, seed, need_full_panel)
    if len(bundle["test_items"]) == 0:
        raise DataUnavailable(f"{dataset}/pertformer_{variant}: no held-out test items produced by the split")

    adata, vocab, split = bundle["adata"], bundle["vocab"], bundle["split"]
    pos_lookup = runners.build_pos_lookup(adata, vocab)

    gene_id_table = None
    if variant == "B":
        gene_id_table = compute_gene_identity_table(bundle["adata_full"], vocab, split["ctrl_cells"],
                                                     d_id=cfg["gene_id_dim"], seed=seed)
    bundle.pop("adata_full", None)

    pf_cfg = PertFormerConfig(n_genes=adata.n_vars, n_pert_genes=len(vocab), variant=variant,
                               d_model=cfg["pf_d_model"], nhead=cfg["pf_nhead"], num_layers=cfg["pf_layers"],
                               dim_feedforward=cfg["pf_ff"], gene_id_dim=cfg["gene_id_dim"])
    model = PertFormer(pf_cfg, gene_identity_table=gene_id_table)

    train_ds = PerturbSeqDataset(adata, split["train_cells"], split["ctrl_cells"], vocab, max_perts=2, seed=seed)
    if len(train_ds) == 0:
        raise DataUnavailable(f"{dataset}/pertformer_{variant}: empty training set")
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)

    test_gene_names = set()
    for item in bundle["test_items"]:
        if isinstance(item, tuple):
            test_gene_names.update(item)
        else:
            test_gene_names.add(item)
    test_gene_vocab_indices = sorted({vocab[g] for g in test_gene_names if g in vocab})

    init_snapshot = runners.snapshot_init_qpert(model)

    ckpt = runners.make_checkpointer(ctx, f"p20::{unit['id']}")
    _train_log, grad_norm_sum = runners.train_pertformer(
        model, loader, pos_lookup, ctx.device, epochs=cfg["pf_epochs"], lr=cfg["lr"],
        ckpt=ckpt, ctx=ctx, logger=ctx.logger, track_grad_indices=test_gene_vocab_indices,
    )

    audit_df = runners.run_pertformer_audit(model, bundle, vocab, seed, cfg["n_per_pair"], cfg["deg_k"],
                                             ctx.device, pos_lookup)
    audit_df.to_csv(csv_path, index=False)

    track_index_of = {pidx: i for i, pidx in enumerate(test_gene_vocab_indices)}
    dorm_df = runners.compute_dormancy_diagnostics(model, test_gene_vocab_indices, init_snapshot,
                                                    grad_norm_sum, track_index_of)
    dorm_df.to_csv(dorm_path, index=False)

    n_boot = min(cfg["n_boot"], 500)
    mode_table = runners.cluster_bootstrap_table(audit_df, n_boot=n_boot, seed=seed)
    gap_table = runners.gap_stats_table(audit_df, n_boot=n_boot, seed=seed)

    summary = {
        "dataset": dataset, "model": "PertFormer", "variant": variant, "seed": seed,
        "n_test_items": int(audit_df["item"].nunique()) if len(audit_df) else 0,
        "mode_means": {r["mode"]: float(r["mean"]) for _, r in mode_table.iterrows()},
        "gap_learned_minus": {r["mode"]: float(r["gap_mean"]) for _, r in gap_table.iterrows()},
        "dormancy_mean_movement_from_init": float(dorm_df["movement_from_init"].mean()) if len(dorm_df) else None,
        "dormancy_mean_grad_norm_sum": (float(dorm_df["grad_norm_sum_during_training"].mean())
                                         if variant == "A" and len(dorm_df) else None),
        "csv": str(csv_path), "dormancy_csv": str(dorm_path),
    }
    atomic_write_json(summary_path, summary)
    return summary


# ============================================================================
# P30: GEARS published-checkpoint audit (fail-soft optional, MEDIUM ROCm risk)
# ============================================================================
def _gears_availability() -> tuple[bool, str]:
    try:
        import torch_geometric  # noqa: F401
    except Exception as e:
        return False, f"torch_geometric unavailable ({e!r})"
    try:
        import gears  # noqa: F401  (pip package "cell-gears", import name "gears")
    except Exception as e:
        return False, f"gears (cell-gears) package unavailable ({e!r})"
    return True, "torch_geometric + gears both importable"


def p30_list_units() -> List[dict]:
    ok, reason = _gears_availability()
    log.info("[p30] GEARS availability: %s (%s)", ok, reason)
    if not ok:
        return []
    seeds = [0] if runners.is_smoke() else [1, 2, 3]
    return [{"id": f"norman__gears__seed{s}", "dataset": "norman", "seed": s} for s in seeds]


def p30_run_unit(unit: dict, ctx: RunContext) -> dict:
    import pandas as pd
    import torch

    ok, reason = _gears_availability()
    if not ok:
        raise DataUnavailable(f"GEARS unavailable at run time: {reason}")

    seed = unit["seed"]
    set_seed(seed)
    out_dir = Path(ctx.results_dir) / "intervenefm" / "p30" / "gears"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"gears_norman_seed{seed}_summary.json"
    movement_path = out_dir / f"gears_norman_seed{seed}_embedding_movement.csv"
    _cached = read_json(summary_path)
    if isinstance(_cached, dict):
        return _cached

    try:
        from gears import GEARS, PertData  # noqa: F401
    except Exception as e:
        raise DataUnavailable(f"GEARS import failed at run time: {e!r}")

    norman_path = staging.resolve_norman_path()
    data_dir = norman_path.parent
    ckpt_dir = data_dir / "gears_norman_pretrained"
    if not ckpt_dir.exists():
        raise DataUnavailable(
            f"published GEARS checkpoint not staged at {ckpt_dir}; "
            f"stage it under EXEA_DATA_ROOT/gears_norman_pretrained/ and re-run"
        )

    # Best-effort: the `gears` package's public API (PertData/GEARS, the
    # SGConv-based similarity network at `best_model.sim_layers` over
    # `G_sim`/`G_sim_weight`, and the raw `best_model.pert_emb` embedding) is
    # reproduced here from the published GEARS architecture as closely as
    # possible; this path is not runnable in this environment (no
    # torch_geometric locally) so it could not be executed end-to-end here.
    pert_data = PertData(str(data_dir))
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=32)

    gears_model = GEARS(pert_data, device=str(ctx.device))
    gears_model.load_pretrained(str(ckpt_dir))

    raw_emb = gears_model.best_model.pert_emb.weight.data.clone()
    with torch.no_grad():
        post_sgconv = gears_model.best_model.sim_layers[0](
            raw_emb, gears_model.best_model.G_sim, gears_model.best_model.G_sim_weight,
        )
    test_pert_genes = list(getattr(pert_data, "test_pert_genes", []) or [])[:100]
    node_map = getattr(pert_data, "node_map_pert", {}) or getattr(pert_data, "node_map", {})
    rows = []
    for g in test_pert_genes:
        gi = node_map.get(g)
        if gi is None:
            continue
        pre, post = raw_emb[gi], post_sgconv[gi]
        rows.append({
            "gene": g,
            "pre_sgconv_norm": float(pre.norm()),
            "post_sgconv_norm": float(post.norm()),
            "pre_post_delta_norm": float((post - pre).norm()),
        })
    movement_df = pd.DataFrame(rows)
    movement_df.to_csv(movement_path, index=False)

    summary = {
        "dataset": "norman", "model": "GEARS", "seed": seed,
        "n_test_genes_measured": len(rows),
        "embedding_movement_csv": str(movement_path),
        "note": "GEARS post-processes the raw pert embedding through an SGConv "
                "over the GO similarity graph; held-out rows are GO-neighbor "
                "mixtures, not dormant at random-init -- see pre_post_delta_norm.",
    }
    atomic_write_json(summary_path, summary)
    return summary


# ============================================================================
# P40: scGPT-Perturb audit (fail-soft optional, HIGH ROCm risk)
# ============================================================================
def _scgpt_availability() -> tuple[bool, str]:
    try:
        import scgpt  # noqa: F401
    except Exception as e:
        return False, f"scgpt unavailable ({e!r})"
    return True, "scgpt importable"


def p40_list_units() -> List[dict]:
    ok, reason = _scgpt_availability()
    log.info("[p40] scGPT availability: %s (%s)", ok, reason)
    if not ok:
        return []
    seeds = [0] if runners.is_smoke() else [1, 2, 3]
    return [{"id": f"norman__scgpt__seed{s}", "dataset": "norman", "seed": s} for s in seeds]


def p40_run_unit(unit: dict, ctx: RunContext) -> dict:
    ok, reason = _scgpt_availability()
    if not ok:
        raise DataUnavailable(f"scGPT unavailable at run time: {reason}")

    seed = unit["seed"]
    set_seed(seed)
    out_dir = Path(ctx.results_dir) / "intervenefm" / "p40" / "scgpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"scgpt_norman_seed{seed}_summary.json"
    _cached = read_json(summary_path)
    if isinstance(_cached, dict):
        return _cached

    try:
        from scgpt.model import TransformerModel  # noqa: F401
    except Exception as e:
        raise DataUnavailable(f"scGPT model import failed at run time: {e!r}")

    ckpt_dir = staging.DATA_ROOT / "scgpt_perturb_pretrained"
    if not ckpt_dir.exists():
        raise DataUnavailable(
            f"published scGPT-Perturb checkpoint not staged at {ckpt_dir}; "
            f"stage it under EXEA_DATA_ROOT/scgpt_perturb_pretrained/ and re-run. "
            f"Best-effort only: PertFormer (P20) already covers the reviewers' "
            f"'only additive architectures were tested' complaint, so scGPT "
            f"failing here is acceptable and MUST NOT crash the harness."
        )

    # use_fast_transformer=False per the harness's hard ROCm rule (no
    # flash-attn); AMP intentionally left off here (ctx.device autocast is
    # applied by the caller only if this path is reached on an accelerator).
    raise DataUnavailable(
        "scGPT-Perturb checkpoint loading path is environment-specific and not "
        "implemented beyond staging checks in this fail-soft-optional job."
    )


# ============================================================================
def get_jobs() -> List[Job]:
    return [
        Job(job_id="intervenefm.p10_cpa_cpg_audit", project="intervenefm", priority=10,
            list_units=p10_list_units, run_unit=p10_run_unit,
            est_unit_seconds=1800.0,
            description="Core CPA+CPG intervention-gap audit, multi-seed, per-dataset "
                        "(learned_CPA_baseline / CPG_SVD / CPG_GO / CPG_linear).",
            tags=["cpa", "cpg", "audit", "norman", "replogle"]),
        Job(job_id="intervenefm.p10_cluster_bootstrap", project="intervenefm", priority=11,
            list_units=p10_agg_list_units, run_unit=p10_agg_run_unit,
            est_unit_seconds=120.0,
            description="Cross-seed cluster-bootstrap CI + effect-size tables for P10.",
            tags=["stats", "cluster-bootstrap"]),
        Job(job_id="intervenefm.p20_pertformer_audit", project="intervenefm", priority=20,
            list_units=p20_list_units, run_unit=p20_run_unit,
            est_unit_seconds=2400.0,
            description="PertFormer (torch.nn.TransformerEncoderLayer) attention audit, "
                        "variant A (embedding q_pert) vs B (CPG-attention MLP q_pert), "
                        "with held-out q_pert dormancy diagnostics.",
            tags=["pertformer", "attention", "audit"]),
        Job(job_id="intervenefm.p30_gears_audit", project="intervenefm", priority=30,
            list_units=p30_list_units, run_unit=p30_run_unit,
            est_unit_seconds=3600.0,
            description="Published-checkpoint GEARS audit + pre/post-SGConv held-out gene "
                        "embedding movement. Fail-soft optional: no-ops if torch_geometric "
                        "or the gears package is unavailable.",
            tags=["gears", "optional", "torch_geometric"]),
        Job(job_id="intervenefm.p40_scgpt_audit", project="intervenefm", priority=40,
            list_units=p40_list_units, run_unit=p40_run_unit,
            est_unit_seconds=3600.0,
            description="scGPT-Perturb audit. Fail-soft optional: no-ops if scgpt is "
                        "unavailable; PertFormer (P20) already covers the attention-"
                        "architecture reviewer ask.",
            tags=["scgpt", "optional", "high-risk"]),
    ]
