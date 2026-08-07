"""Intra-unit checkpointing for long training runs.

Most units here are short (small MLPs / small pose decoders on cached features),
so unit-level resume via the manifest is enough. A few units are long — Tier-B
PEFT (ViT forward per step), cross-backbone feature precompute, PertFormer /
GEARS training. For those, save a checkpoint every N seconds so a 13:00 kill
loses minutes, not the whole unit. On the next day the same unit re-runs and
``load`` restores its state, continuing from the last saved step.

Checkpoints are written atomically (temp file + os.replace) to survive a kill
mid-write. torch.save is used for tensors; a sidecar JSON holds step/epoch/rng.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from .util import atomic_write_json, read_json


class Checkpointer:
    def __init__(self, path: str | Path, every_seconds: float = 120.0):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".meta.json")
        self.every = every_seconds
        self._last = 0.0

    def exists(self) -> bool:
        return self.path.exists() and self.meta_path.exists()

    def load(self) -> Optional[dict]:
        """Return {'state': <torch state dict>, 'meta': <json>} or None."""
        if not self.exists():
            return None
        try:
            import torch
            state = torch.load(self.path, map_location="cpu")
            meta = read_json(self.meta_path, default={})
            return {"state": state, "meta": meta}
        except Exception:
            return None  # corrupt checkpoint -> start fresh (fail-soft)

    def due(self) -> bool:
        return (time.monotonic() - self._last) >= self.every

    def save(self, state: dict[str, Any], meta: dict[str, Any], force: bool = False) -> None:
        if not force and not self.due():
            return
        import torch
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".pt.tmp")
        os.close(fd)
        try:
            torch.save(state, tmp)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        atomic_write_json(self.meta_path, meta)
        self._last = time.monotonic()

    def clear(self) -> None:
        for p in (self.path, self.meta_path):
            try:
                p.unlink()
            except (OSError, FileNotFoundError):
                pass
