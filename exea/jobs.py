"""Job / Unit interface shared by both projects.

A **Job** is a named body of work (e.g. "behaviorfm.peft_panel_decoder"). It
exposes a flat list of **units** — the smallest independently-resumable pieces
(usually one seed × one species × one config). The orchestrator iterates units,
skipping those the manifest already marks done, and calls ``run_unit`` for the
rest, one at a time, fail-soft, within the day's time budget.

Contract for ``run_unit`` implementations (critical for unattended safety):
  * MUST be idempotent: if its output already exists on disk, return it rather
    than recompute (cheap resume across days).
  * MUST be self-contained per unit: no dependence on in-memory state from a
    previous unit (a previous unit may have run on a different day).
  * SHOULD consult ``ctx.should_stop()`` inside any long inner loop and
    checkpoint, so a mid-unit kill loses little.
  * MUST raise on real failure (the orchestrator catches, records, and moves
    on). Raise ``exea.staging.DataUnavailable`` for missing data — that is
    treated as a soft skip, not an error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .timebox import TimeBox


class UnitPaused(Exception):
    """Raised by a run_unit (or translated from a project's cooperative-stop
    exception) when the unit stopped itself on ctx.should_stop() AFTER writing a
    checkpoint. It means "resume me next day without penalty" — the orchestrator
    rolls back the attempt so a repeatedly-paused heavy unit never reaches
    'gaveup'. Distinct from a genuine crash (which keeps its attempt) and from
    DataUnavailable (missing data/dep)."""


@dataclass
class RunContext:
    results_dir: Path
    state_dir: Path
    timebox: TimeBox
    logger: logging.Logger
    device: Any = None                 # torch.device, filled by orchestrator
    est_unit_seconds: float = 120.0    # default headroom reserved per unit

    def should_stop(self, need_seconds: float | None = None) -> bool:
        return self.timebox.should_stop(
            need_seconds if need_seconds is not None else 0.0
        )


@dataclass
class Job:
    job_id: str
    project: str                       # "behaviorfm" | "intervenefm"
    priority: int                      # lower runs first
    list_units: Callable[[], list[dict]]
    run_unit: Callable[[dict, RunContext], dict]
    # rough per-unit cost estimate, used to reserve time headroom before starting
    est_unit_seconds: float = 120.0
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def unit_id(self, unit: dict) -> str:
        """Globally-unique manifest key for a unit."""
        return f"{self.job_id}::{unit['id']}"
