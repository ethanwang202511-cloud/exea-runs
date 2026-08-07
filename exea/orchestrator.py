"""Daily entrypoint: run pending units in priority order, fail-soft, time-boxed.

Invoked by the `start`/`resume` stages (`python -m exea.orchestrator`). It:
  1. detects the device (ROCm-safe) and logs an environment report,
  2. loads the manifest (yesterday's progress, restored from the snapshot),
  3. gathers jobs from both projects, sorts by (priority, job_id),
  4. iterates units: skip if done/gaveup; stop if out of time or signalled;
     else run one unit fail-soft and record the outcome,
  5. writes a run summary.

Nothing here is project-specific. Projects contribute jobs via
``behaviorfm.jobs.get_jobs()`` and ``intervenefm.jobs.get_jobs()``. Missing
project modules or import errors in one project never stop the other.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

from . import device as devmod
from .jobs import Job, RunContext, UnitPaused
from .manifest import Manifest
from .staging import DataUnavailable
from .timebox import TimeBox
from .util import atomic_write_json, set_seed, utcnow_iso

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
STATE_DIR = REPO_ROOT / "state"
LOG_DIR = STATE_DIR / "logs"


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exea")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "run.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _gather_jobs(logger: logging.Logger, only_project: str | None) -> list[Job]:
    jobs: list[Job] = []
    projects = ["behaviorfm", "intervenefm"]
    if only_project:
        projects = [only_project]
    for proj in projects:
        try:
            mod = __import__(f"{proj}.jobs", fromlist=["get_jobs"])
            proj_jobs = mod.get_jobs()
            logger.info("[jobs] %s contributed %d jobs", proj, len(proj_jobs))
            jobs.extend(proj_jobs)
        except Exception as e:
            # One project's import failure must not stop the other.
            logger.error("[jobs] could not load jobs from %s: %s", proj, e)
            logger.debug(traceback.format_exc())
    jobs.sort(key=lambda j: (j.priority, j.job_id))
    return jobs


def run(only_project: str | None = None, max_units: int | None = None) -> dict:
    logger = _setup_logging()
    set_seed(0)  # global default; each unit reseeds from its own seed
    dev = devmod.get_device()
    report = devmod.device_report()
    logger.info("[env] device report: %s", report)

    timebox = TimeBox(stop_file=STATE_DIR / "STOP")
    manifest = Manifest(STATE_DIR / "manifest.json")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        results_dir=RESULTS_DIR,
        state_dir=STATE_DIR,
        timebox=timebox,
        logger=logger,
        device=dev,
    )

    smoke = os.environ.get("EXEA_SMOKE", "") not in ("", "0", "false", "False")
    # Headroom reserved before starting a unit. Capped so we never idle more
    # than a few minutes waiting for the snapshot: long units checkpoint and
    # resume, so starting one we can't finish is recoverable, not catastrophic.
    # Zero in smoke so every tiny unit runs during verification.
    headroom_cap = 0.0 if smoke else float(os.environ.get("EXEA_HEADROOM_CAP_SECONDS", "300"))

    jobs = _gather_jobs(logger, only_project)
    logger.info("[run] %d jobs, budget %.0fs, device=%s, smoke=%s",
                len(jobs), timebox.budget, dev, smoke)

    ran = done = failed = skipped = 0
    stopped_early = False

    for job in jobs:
        if timebox.should_stop():
            stopped_early = True
            break
        try:
            units = job.list_units()
        except DataUnavailable as e:
            logger.warning("[job %s] data unavailable, skipping job: %s", job.job_id, e)
            continue
        except Exception as e:
            logger.error("[job %s] list_units failed: %s", job.job_id, e)
            logger.debug(traceback.format_exc())
            continue

        logger.info("[job %s] %d units (priority %d)", job.job_id, len(units), job.priority)
        ctx.est_unit_seconds = job.est_unit_seconds

        for unit in units:
            uid = job.unit_id(unit)
            if manifest.should_skip(uid):
                skipped += 1
                continue
            # Reserve headroom so we don't start a unit we can't finish cleanly,
            # but cap it (checkpointed units resume next day) and zero it in smoke.
            need = min(job.est_unit_seconds, headroom_cap)
            if timebox.should_stop(need_seconds=need):
                logger.info("[run] stopping before %s (%.0fs left, reserve ~%.0fs)",
                            uid, timebox.remaining(), need)
                stopped_early = True
                break

            manifest.mark_running(uid, meta={"project": job.project})
            logger.info("[unit] START %s", uid)
            try:
                result = job.run_unit(unit, ctx)
                manifest.mark_done(uid, result=result if isinstance(result, dict) else None)
                done += 1
                logger.info("[unit] DONE  %s", uid)
            except UnitPaused as e:
                # Cooperative mid-unit stop (checkpointed). No penalty; the pause
                # means time is up, so end the run cleanly for a next-day resume.
                manifest.mark_paused(uid, str(e))
                logger.info("[unit] PAUSE %s (%s)", uid, e)
                stopped_early = True
                break
            except DataUnavailable as e:
                # Soft: data/dep missing -> 'skipped' (retried daily, no attempt spent).
                manifest.mark_skipped(uid, str(e))
                skipped += 1
                logger.warning("[unit] SKIP  %s (data unavailable: %s)", uid, e)
            except Exception as e:
                tb = traceback.format_exc()
                manifest.mark_failed(uid, tb)
                failed += 1
                logger.error("[unit] FAIL  %s: %s", uid, e)
                logger.debug(tb)
            ran += 1
            if max_units is not None and ran >= max_units:
                logger.info("[run] hit max_units=%d, stopping", max_units)
                stopped_early = True
                break
        if stopped_early:
            break

    summary = {
        "finished_at": utcnow_iso(),
        "device": report,
        "counts": {"ran": ran, "done": done, "failed": failed, "skipped": skipped},
        "manifest_summary": manifest.summary(),
        "timebox": timebox.status(),
        "stopped_early": stopped_early,
    }
    atomic_write_json(STATE_DIR / "last_run_summary.json", summary)
    logger.info("[run] summary: %s", summary["counts"])
    logger.info("[run] manifest: %s", summary["manifest_summary"])
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="exea-runs orchestrator")
    ap.add_argument("--project", default=os.environ.get("EXEA_ONLY_PROJECT") or None,
                    choices=["behaviorfm", "intervenefm"],
                    help="run only one project's jobs")
    ap.add_argument("--max-units", type=int,
                    default=int(os.environ["EXEA_MAX_UNITS"]) if os.environ.get("EXEA_MAX_UNITS") else None,
                    help="stop after N units (smoke testing)")
    args = ap.parse_args()
    run(only_project=args.project, max_units=args.max_units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
