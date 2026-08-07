# RUNBOOK — operating exea-runs on the Exea Labs node

Practical guide for whoever provisions the runner and for debugging via the
dashboard. For the *what/why*, see [`README.md`](README.md); for the stage
contract, see [`exea.md`](exea.md).

## Operator checklist (once)

1. **Repo**: push this repo to a GitHub repository the runner can clone.
2. **Token**: provide a **fine-grained PAT scoped to THIS repo only**, contents
   read/write (write is only for `save.sh` to push results). Never a classic /
   all-repo token.
3. **Base image**: a working **ROCm PyTorch** image (e.g. `rocm/pytorch:latest`,
   ROCm 6.3–6.4 + torch 2.5–2.8). `setup.sh` installs only pure-Python deps on
   top; it does NOT reinstall torch.
4. **Network in `setup`**: required (~11 GB of datasets + weights are fetched
   there). Not required during the timed run window.
5. **Persistent disk** across the daily snapshot, OR rely on `save.sh` pushing
   results to git. Ideally both.

## The five stages (runner invokes these)

```
setup  -> bash setup.sh    # once/session, with net: deps + pre-download
start  -> bash start.sh    # first run
resume -> bash resume.sh   # every following day
stop   -> bash stop.sh     # request a clean halt
save   -> bash save.sh     # push results/manifest to git
```

`resume.sh` runs the orchestrator once per project (isolated processes sharing
one absolute deadline), alternating order daily. It self-halts ~2h45m in.

## What runs, in priority order

**intervenefm** (pure-PyTorch core is LOW ROCm risk):
- P10 CPA+CPG audit (CPG-SVD / CPG-GO / CPG-linear) × {Norman, Replogle K562,
  RPE1} × seeds → cluster-bootstrap CI tables.
- P20 **PertFormer** attention audit (Variant A vs B) + dormancy diagnostics.
- P30 GEARS / P40 scGPT — **fail-soft optional** (see risk matrix).

**behaviorfm**:
- P05 DINOv2 feature precompute (prereq for P10–P30).
- P10 PEFT panel Tier A (cached-feature): ITA / decoder-FT / LoRA-dec / BitFit-dec.
- P15 per-budget Pareto grid (incl. the previously-missing 1.2M budget).
- P20 decoder-FT sensitivity sweep. P30 inverted-U species expansion.
- P40 cross-backbone precompute + P41 Tier B (backbone-forward: LoRA-bb /
  AdaptFormer / IA³ / VPT).

## Risk / fail-soft matrix

| Component | ROCm risk | If it fails |
|---|---|---|
| CPA/CPG/PertFormer, ITA/decoder-FT, DINOv2/DINOv1/MAE/CLIP precompute | LOW | should just work |
| EVA-02 (timm) | LOW–MED | that backbone's units `skipped`, retried daily |
| LoRA/IA³ via `peft` | LOW | if `peft` absent, those units `skipped` |
| GEARS (P30, `torch_geometric`+`cell-gears`) | MED | job contributes 0 units, logged reason; nothing else affected |
| scGPT (P40) | HIGH | job contributes 0 units; PertFormer already covers the attention result |

Nothing in the HIGH/MED rows can crash the run. `skipped` units retry every day
(the dep may get installed); `gaveup` only after repeated genuine failures (a
timed-out unit is NOT a failure and retries freely).

**P30 (GEARS) and P40 (scGPT) are inert on a fresh clone** — they need their
optional deps installed AND published checkpoints staged to
`$EXEA_DATA_ROOT/{gears_norman_pretrained,scgpt_perturb_pretrained}/`. The
runtime estimate excludes them. PertFormer (P20) already delivers the
attention-architecture result, so leaving P30/P40 inert loses no headline
finding.

## Debugging from the dashboard

- Per-unit status: `state/manifest.json` (`done` / `skipped` / `failed` /
  `gaveup`). `failed` units store the traceback tail.
- Run summaries: `state/last_run_summary.json`; full log: `state/logs/run.log`.
- Results: `results/<project>/<job>/...` (per-unit JSON + rolled-up CSV).

## Manual full-scale invocation (outside the runner)

```bash
# one-time staging (with network)
bash setup.sh
# a day's work (both projects, timeboxed, resumable)
bash resume.sh
# or a single project / unbounded budget:
EXEA_DEVICE=cuda EXEA_RUN_BUDGET_SECONDS=9000 \
  PYTHONPATH=.:intervenefm python -m exea.orchestrator --project intervenefm
```

## Optional heavy deps (only if you want P30/P40)

Install on the server AFTER the ROCm torch is in place:
```bash
pip install torch_geometric         # core only; NOT torch-scatter/sparse
pip install cell-gears              # GEARS (P30)
# scGPT (P40): install WITHOUT flash-attn; keep use_fast_transformer=False
```
Stage the published checkpoints to `$EXEA_DATA_ROOT/gears_norman_pretrained/`
and `$EXEA_DATA_ROOT/scgpt_perturb_pretrained/` (P30/P40 raise DataUnavailable
pointing at those paths if missing).
