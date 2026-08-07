---
exea_version: 1
owner_email: YOUR_EMAIL@example.com          # <-- fill in before submitting
github_username: ethanwang202511-cloud       # <-- confirm this is correct
github_pat: ${EXEA_GITHUB_PAT}               # injected by the runner; do not hardcode a real token
checkpoint_dir: .                             # persist the WHOLE working dir across the daily snapshot:
                                              # state/manifest.json + checkpoints, the trained base model,
                                              # AND the ~8GB staged datasets/feature caches (data/). If only
                                              # a subdir were snapshotted, those caches would be lost and
                                              # every day would waste its slot re-downloading them.
---

# exea-runs

Autonomous experiment queue for two computational-biology / ML papers. The
runner clones this repo, runs **SETUP** once, then **START**/**RESUME** each day
inside the 10:00–13:00 PST window, is stopped at 13:00, and resumes the next
morning. Progress is written to `./state` so finished work is never repeated.

Each stage below is one command under a heading; the runner executes the command
under each heading.

## SETUP

Install dependencies and pre-download all datasets + model weights (with
network; idempotent — safe to re-run). Does NOT reinstall torch (assumes a ROCm
PyTorch base image).

```bash
bash setup.sh
```

## START

Begin the experiment queue (first run of the session).

```bash
bash start.sh
```

## RESUME

Continue where the previous day left off. Skips units already marked done in
`state/manifest.json`; runs pending units in priority order until the deadline
approaches, then stops cleanly before the 13:00 snapshot.

```bash
bash resume.sh
```

## STOP

Request a clean halt (has a hard ~120s budget to flush): the running job
finishes its current unit, checkpoints, and exits at the next safe point.

```bash
bash stop.sh
```

## SAVE

Commit and push results + manifest back to the repo (best-effort; uses the
runner-injected token if present). The disk snapshot is the primary
persistence; this makes results visible on GitHub too.

```bash
bash save.sh
```

---

## Why this finishes within the daily window (checkpointing contract)

- **Self-halting.** `resume.sh` respects an absolute deadline (`EXEA_DEADLINE_UTC`
  if the runner injects one, else its own ~2h45m budget) and stops with margin
  before 13:00, having flushed the manifest. Also honours SIGTERM and a
  `state/STOP` sentinel.
- **Resumable across days.** Per-unit status lives in `state/manifest.json`
  (captured by the snapshot + pushed by SAVE). Finished units are skipped; a
  killed unit is retried without penalty next day.
- **ROCm-safe & fail-soft.** No CUDA-only kernels (SDPA, bf16; no flash-attn /
  xformers / apex / bitsandbytes). One unit or one optional dependency failing
  never aborts the run.
- **No run-time network assumed.** All data + weights are fetched in SETUP.

## What the runner must provide

- Base image with a working **ROCm PyTorch** (e.g. `rocm/pytorch:latest`).
- **Network during SETUP** (deps + ~8 GB of public datasets/weights). Not needed
  during the run window.
- The **fine-grained token scoped to this repository only** (Exea requires one
  for clone/push) — used by SAVE.
