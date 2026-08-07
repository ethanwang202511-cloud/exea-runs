"""Data-staging layer: fetch-or-find, never assume.

Datasets and model weights are NOT committed to git (too large). They are
acquired once in the `setup` stage and cached on the persistent (snapshotted)
disk. Two guarantees for unattended safety:

  * ensure(): idempotent acquisition. If the target already exists (and passes
    an optional check), it is a no-op — so re-running setup across days is
    cheap and safe.
  * require(): a job calls this at run time; if the data is missing it raises
    DataUnavailable, which the orchestrator converts into a *fail-soft* skip of
    that job only. A missing dataset never crashes the whole run.

Downloaders are plain callables so each project owns its own fetch logic
(reusing the existing download_*.py scripts). We assume network is available in
`setup`; if it is not, ensure() fails loudly *there* (before the timed run),
which is the right place to fail.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("exea.staging")


class DataUnavailable(Exception):
    """Raised when a job's required data is not present at run time."""


@dataclass
class StagedItem:
    name: str
    path: Path                      # final location on disk
    fetch: Optional[Callable[[Path], None]] = None   # downloader(dest) -> None
    min_bytes: int = 0              # sanity floor; 0 disables the check
    is_dir: bool = False

    def present(self) -> bool:
        p = self.path
        if not p.exists():
            return False
        if self.is_dir:
            return any(p.iterdir())
        if self.min_bytes and p.stat().st_size < self.min_bytes:
            return False
        return True

    def ensure(self) -> bool:
        """Acquire if missing. Returns True if present after the call.

        Raises the underlying fetch error (in setup, loudly) if download fails.
        """
        if self.present():
            logger.info("[stage] %s already present at %s", self.name, self.path)
            return True
        if self.fetch is None:
            logger.warning("[stage] %s missing and no fetcher provided: %s", self.name, self.path)
            return False
        logger.info("[stage] fetching %s -> %s", self.name, self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fetch(self.path)
        ok = self.present()
        if not ok:
            logger.error("[stage] fetch of %s did not produce a valid file at %s", self.name, self.path)
        return ok


def require(item: StagedItem) -> Path:
    """Run-time guard: return the path or raise DataUnavailable (fail-soft)."""
    if not item.present():
        raise DataUnavailable(f"{item.name} not staged at {item.path}")
    return item.path


def ensure_all(items: list[StagedItem]) -> dict[str, bool]:
    """Setup-stage helper: acquire everything, report per-item status."""
    status: dict[str, bool] = {}
    for it in items:
        try:
            status[it.name] = it.ensure()
        except Exception as e:  # setup should surface, but keep going per-item
            logger.exception("[stage] failed to ensure %s: %s", it.name, e)
            status[it.name] = False
    return status
