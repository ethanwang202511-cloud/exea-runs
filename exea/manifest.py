"""Idempotent per-unit progress manifest — the core of cross-day resume.

A *unit* is the smallest independently-checkpointable piece of work, e.g.
"intervenefm.cpg_replogle_k562 seed=2" or "behaviorfm.peft_lora fox seed=7".
The orchestrator asks the manifest whether a unit is already done (skip) and
records the outcome of every unit it runs. The manifest is the single source of
truth restored from yesterday's snapshot, so a multi-day run makes monotonic
progress and never repeats finished work.

Status values:
  pending  — seen but not yet completed
  done     — completed successfully; skipped forever after
  failed   — errored; retried up to max_attempts, then given up (so one broken
             unit cannot consume every day's slot)
  gaveup   — failed max_attempts times; never retried again
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .util import atomic_write_json, read_json, utcnow_iso


class Manifest:
    def __init__(self, path: str | Path, max_attempts: int = 3):
        self.path = Path(path)
        self.max_attempts = max_attempts
        data = read_json(self.path, default=None)
        if not isinstance(data, dict) or "units" not in data:
            data = {"units": {}, "created": utcnow_iso()}
        self.data: dict[str, Any] = data

    # ------------------------------------------------------------------ query
    def get(self, unit_id: str) -> dict:
        return self.data["units"].get(unit_id, {})

    def is_done(self, unit_id: str) -> bool:
        return self.get(unit_id).get("status") == "done"

    def should_skip(self, unit_id: str) -> bool:
        """Skip if done, or if it has permanently given up.

        Also skip a unit stuck in 'running' after max_attempts starts: that means
        it was started that many times but never reached a Python outcome — the
        signature of a HARD crash (ROCm illegal-access, OOM-kill, driver panic)
        that kills the interpreter before mark_failed can run. Giving it up stops
        it from blocking every later unit's slot forever.

        NOT skipped: 'skipped' (data/dep unavailable) — those retry every day
        because the dependency may appear later (e.g. timm/peft installed on the
        server, a dataset staged tomorrow).
        """
        u = self.get(unit_id)
        st = u.get("status")
        if st in ("done", "gaveup"):
            return True
        if st == "running" and int(u.get("attempts", 0)) >= self.max_attempts:
            return True
        return False

    def attempts(self, unit_id: str) -> int:
        return int(self.get(unit_id).get("attempts", 0))

    # ------------------------------------------------------------------ mutate
    def mark_running(self, unit_id: str, meta: Optional[dict] = None) -> None:
        # Charge the attempt HERE (persisted before run_unit) so that even a hard
        # crash that never reaches mark_failed is counted — otherwise a unit that
        # segfaults the interpreter every time would retry forever and block the
        # slot. A clean 13:00 boundary-kill also lands here, charging at most one
        # attempt to the single in-flight unit; it completes next day (staying at
        # 1 attempt). Only a unit that can NEVER finish climbs to gaveup.
        # DataUnavailable soft-skips roll this back (see mark_skipped).
        u = self.data["units"].setdefault(unit_id, {})
        u["status"] = "running"
        u["attempts"] = int(u.get("attempts", 0)) + 1
        u["started"] = utcnow_iso()
        if meta:
            u.setdefault("meta", {}).update(meta)
        self.save()

    def mark_done(self, unit_id: str, result: Optional[dict] = None) -> None:
        u = self.data["units"].setdefault(unit_id, {})
        u["status"] = "done"
        u["finished"] = utcnow_iso()
        u.pop("error", None)
        if result:
            u["result"] = result
        self.save()

    def mark_skipped(self, unit_id: str, reason: str) -> None:
        """Soft skip: required data/dependency unavailable, detected by a cheap
        pre-check before any real work. Roll back the attempt that mark_running
        charged (this was not a real attempt) and retry every day — the
        dependency may appear later (timm/peft installed, a dataset staged)."""
        u = self.data["units"].setdefault(unit_id, {})
        u["attempts"] = max(0, int(u.get("attempts", 1)) - 1)
        u["status"] = "skipped"
        u["finished"] = utcnow_iso()
        u["skip_reason"] = reason[-500:]
        self.save()

    def mark_paused(self, unit_id: str, reason: str) -> None:
        """Cooperative mid-unit stop that already checkpointed. Roll back the
        attempt mark_running charged and set a retry-eligible status, so a heavy
        unit paused near the daily boundary resumes from its checkpoint next day
        without ever accumulating toward 'gaveup'."""
        u = self.data["units"].setdefault(unit_id, {})
        u["attempts"] = max(0, int(u.get("attempts", 1)) - 1)
        u["status"] = "paused"
        u["finished"] = utcnow_iso()
        u["pause_reason"] = reason[-300:]
        self.save()

    def mark_failed(self, unit_id: str, error: str) -> None:
        # attempts was already charged in mark_running; do not double-count.
        # After max_attempts genuine failures the unit is given up so a
        # truly-broken unit can't consume the slot forever.
        u = self.data["units"].setdefault(unit_id, {})
        attempts = int(u.get("attempts", 0))
        u["status"] = "gaveup" if attempts >= self.max_attempts else "failed"
        u["finished"] = utcnow_iso()
        u["error"] = error[-4000:]  # cap stored traceback
        self.save()

    def save(self) -> None:
        self.data["updated"] = utcnow_iso()
        atomic_write_json(self.path, self.data)

    # ------------------------------------------------------------------ report
    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for u in self.data["units"].values():
            counts[u.get("status", "unknown")] = counts.get(u.get("status", "unknown"), 0) + 1
        return counts
