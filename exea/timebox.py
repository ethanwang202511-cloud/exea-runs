"""Wall-clock budget guard for the daily 3-hour (or shorter) Exea slot.

Exea runs a shared, sequential queue across ~200 students inside a 10:00-13:00
window, then hard-snapshots. We therefore CANNOT assume we get the full three
hours. Three independent stop signals, checked cooperatively between units:

  1. Budget deadline: process_start + EXEA_RUN_BUDGET_SECONDS (default 9900s
     ~= 2h45m, a margin under the 3h kill). Tunable via env.
  2. SIGTERM/SIGINT: the runner (or `stop` stage) may signal us; we set a flag
     and stop at the next safe point, having checkpointed.
  3. A cooperative STOP sentinel file (state/STOP), which `stop.sh` can touch.

Because every training loop checkpoints frequently (see exea.checkpoint), a
hard kill at 13:00 loses at most a few steps — the budget guard exists to make
clean stops the common case, not to be the only safety net.
"""
from __future__ import annotations

import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BUDGET_SECONDS = 9900  # 2h45m; under the 3h hard snapshot with margin


def _seconds_until_iso(iso_str: str) -> float | None:
    """Seconds from now until an ISO-8601 UTC timestamp, or None if unparseable.

    Accepts a trailing 'Z' (normalised to +00:00). Any parse failure returns
    None so the caller falls back to the relative budget rather than crashing —
    a bad env value must never abort the run.
    """
    try:
        s = iso_str.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (dt - datetime.now(timezone.utc)).total_seconds()
    except (ValueError, TypeError):
        return None


class TimeBox:
    def __init__(self, budget_seconds: float | None = None, stop_file: str | Path | None = None):
        self.start = time.monotonic()
        # An ABSOLUTE deadline (unix epoch) lets several sequential processes in
        # one daily slot (e.g. behaviorfm then intervenefm) share the same wall
        # clock. It takes precedence over the relative budget when set.
        env_budget = os.environ.get("EXEA_RUN_BUDGET_SECONDS")
        if budget_seconds is not None:
            self.budget = float(budget_seconds)
        elif env_budget:
            self.budget = float(env_budget)
        else:
            self.budget = float(DEFAULT_BUDGET_SECONDS)
        # If the runner (or resume.sh) hands us an ABSOLUTE deadline, honour the
        # EARLIEST of it and our relative budget — never overshoot a real 13:00
        # kill when the shared queue gives us a short slot. We accept both an
        # epoch form (set by resume.sh) and an ISO-8601 UTC form (a name the
        # Exea runner may inject); whichever is present wins if it is sooner.
        abs_deadlines: list[float] = []
        epoch_env = os.environ.get("EXEA_DEADLINE_EPOCH")
        if epoch_env:
            try:
                abs_deadlines.append(float(epoch_env) - time.time())
            except ValueError:
                pass
        iso_env = os.environ.get("EXEA_DEADLINE_UTC")
        if iso_env:
            secs = _seconds_until_iso(iso_env)
            if secs is not None:
                abs_deadlines.append(secs)
        for rem in abs_deadlines:
            self.budget = min(self.budget, max(0.0, rem))
        self.deadline = self.start + self.budget
        self.stop_file = Path(stop_file) if stop_file else None
        self._signalled = False
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame):  # noqa: ARG001
            self._signalled = True
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not in main thread (e.g. under some runners) — skip silently.
                pass

    def elapsed(self) -> float:
        return time.monotonic() - self.start

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def signalled(self) -> bool:
        if self._signalled:
            return True
        if self.stop_file is not None and self.stop_file.exists():
            self._signalled = True
            return True
        return False

    def should_stop(self, need_seconds: float = 0.0) -> bool:
        """True if we should stop before starting the next chunk.

        ``need_seconds`` lets a caller reserve headroom for a unit it is about
        to start (so we don't begin a 10-minute unit with 3 minutes left).
        """
        if self.signalled():
            return True
        return self.remaining() <= need_seconds

    def status(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed(), 1),
            "remaining_s": round(self.remaining(), 1),
            "budget_s": self.budget,
            "signalled": self.signalled(),
        }
