"""exea: unattended-run harness for the exea-runs monorepo.

This package contains the project-agnostic machinery that lets BehaviorFM and
InterveneFM experiments run autonomously on Exea Labs compute (AMD MI300X /
ROCm), 3 hours per day, with snapshot/resume across days.

Design commitments (every one is a correctness safeguard for UNATTENDED runs):
  * ROCm-safe: no hardcoded .cuda(); device chosen at runtime (exea.device).
  * Time-boxed: the run self-halts before the daily kill (exea.timebox).
  * Idempotent + resumable: per-unit status in a manifest (exea.manifest);
    finished units are skipped on the next day's run.
  * Fail-soft: one unit crashing never aborts the run (exea.orchestrator).
  * Deterministic: fixed seeds everywhere.
  * Data-staging aware: datasets are fetched-or-found, never assumed
    (exea.staging); jobs whose data is missing fail soft, not hard.
"""
__all__ = ["__version__"]
__version__ = "0.1.0"
