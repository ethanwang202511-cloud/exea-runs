# exea-runs

Autonomous compute for two computational-biology / ML papers, packaged to run
**unattended** on the Exea Labs AMD MI300X (ROCm) node: the daily runner pulls
this repo, follows [`exea.md`](exea.md), runs a time-boxed queue of experiments,
snapshots the disk, and resumes the next day.

The experiments are chosen to answer the **actual reviewer complaints** that
rejected the ICML-2026-workshop versions of each paper, in preparation for
journal submission.

## Projects

### `intervenefm/` — auditing per-gene perturbation embeddings in cellular FMs
Population-mean dominance + "lookup-row dormancy" + the CPG fix. Reviewer-driven
compute:
- **P10** core CPA + CPG audit, multi-seed × 3 datasets (Norman, Replogle K562,
  Replogle RPE1), with **CPG-linear** and **CPG-GO** ablations and full
  cluster-bootstrap CI tables (reviewers: vague numbers, missing ablations).
- **P20** **PertFormer** — a self-contained attention model (Variant A raw
  embedding vs Variant B feature-grounded) that tests whether dormancy is
  architecture-agnostic. Answers the #1 reviewer ask ("only additive
  architectures tested") **without** scGPT's fragile ROCm toolchain.
- **P30/P40** GEARS / scGPT-Perturb published-checkpoint audits — **fail-soft
  optional** (they may not build on ROCm; PertFormer already covers the point).

### `behaviorfm/` — Identity-Token Adaptation for cross-species animal pose
Reviewer-driven compute:
- **P10** the **PEFT baseline panel** (the #1 unanimous ask), split into the two
  honest tiers — **Tier A** (cached-feature: ITA, decoder-FT, LoRA-on-decoder,
  BitFit-on-decoder) and **Tier B** (backbone-forward: LoRA-on-backbone,
  AdaptFormer, IA³, VPT).
- **P15** per-budget Pareto grid (fills the unreported 1.2M budget).
- **P20** decoder-FT comparator sensitivity sweep (the "tie" rested on one
  under-trained baseline).
- **P30** inverted-U species expansion (3 points → 8+).
- **P40** cross-backbone feature precompute + Tier-B PEFT (DINOv2/DINOv1/MAE/
  CLIP/EVA-02).

## How it runs (the safety model)

See [`exea.md`](exea.md). Five stages: `setup` (install + pre-download,
idempotent), `start`/`resume` (run the orchestrator), `stop` (clean halt),
`save` (push results to git). Guarantees:

- **ROCm-safe** — no CUDA-only kernels; SDPA attention; bf16; no flash-attn /
  xformers / apex / bitsandbytes; `torch_geometric` core only.
- **Resumable** — per-unit status in `state/manifest.json`; finished units are
  never recomputed. InterveneFM's long training units also checkpoint mid-run;
  BehaviorFM units are sized to resume at unit granularity (a killed unit simply
  re-runs next day, with no penalty toward its retry budget).
- **Time-boxed** — self-halts ~2h45m in (under the 3h snapshot), honours SIGTERM
  and `state/STOP`.
- **Fail-soft** — one unit or one optional dependency failing never aborts the
  run.
- **No run-time network assumed** — all data + weights fetched in `setup`.

## Running locally (smoke test)

Every job has a smoke mode that runs on CPU in seconds against tiny synthetic
data — no GPU, no downloads:

```bash
EXEA_SMOKE=1 EXEA_DEVICE=cpu python -m exea.orchestrator
```

Full scale on the server is just `bash start.sh` (then `bash resume.sh` daily).

## Layout

```
exea/            # project-agnostic harness (device, timebox, manifest,
                 # checkpoint, staging, orchestrator, stage_setup)
behaviorfm/      # src/ (vendored, validated) + jobs.py + staging.py + helpers
intervenefm/     # src/ (vendored, validated) + jobs.py + staging.py + helpers
results/         # tabular outputs (tracked in git)
state/           # manifest.json + logs (manifest tracked in git)
setup.sh start.sh resume.sh stop.sh save.sh   # the five Exea stages
```

## Security note (for whoever provisions the runner)

Give the runner a **fine-grained GitHub token scoped to THIS repository only**
(contents: read/write for `save.sh` to push results). Never a classic or
all-repository token.
