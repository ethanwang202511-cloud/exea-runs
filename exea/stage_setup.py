"""SETUP-stage entrypoint: pre-download all datasets + model weights to disk.

Run once per session (see setup.sh) while the network is available, because the
timed run window may have no network. Idempotent: each project's staging layer
skips items already present.

Convention (each project may implement any subset; all optional, all fail-soft):
  <project>.staging.get_staged_items() -> list[exea.staging.StagedItem]
  <project>.staging.prefetch()         -> None   # e.g. warm the HF weight cache

A project raising/mis-importing here must not stop the other project.
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("exea.stage_setup")


def _stage_project(proj: str) -> dict:
    out: dict = {"project": proj}
    try:
        mod = __import__(f"{proj}.staging", fromlist=["get_staged_items", "prefetch"])
    except Exception as e:
        logger.warning("[setup] %s.staging not available: %s", proj, e)
        out["staging_import"] = f"unavailable: {e}"
        return out

    # 1. datasets — try the project's aggregate entrypoint, several names.
    try:
        from .staging import ensure_all
        if hasattr(mod, "ensure_all_data"):
            out["data"] = mod.ensure_all_data()
        elif hasattr(mod, "get_staged_items"):
            out["data"] = ensure_all(mod.get_staged_items())
        else:
            out["data"] = "no aggregate staging entrypoint (ensure_all_data/get_staged_items)"
            logger.warning("[setup] %s.staging has no ensure_all_data()/get_staged_items()", proj)
    except Exception as e:
        logger.exception("[setup] %s dataset staging failed: %s", proj, e)
        out["data_error"] = str(e)

    # 2. model weights (optional warm-up of caches)
    try:
        if hasattr(mod, "prefetch"):
            mod.prefetch()
            out["prefetch"] = "ok"
    except Exception as e:
        logger.warning("[setup] %s prefetch failed (non-fatal): %s", proj, e)
        out["prefetch_error"] = str(e)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", choices=["behaviorfm", "intervenefm"], default=None,
                    help="stage only this project (run once per project with its path)")
    args = ap.parse_args()
    projects = [args.project] if args.project else ["behaviorfm", "intervenefm"]
    results = []
    for proj in projects:
        logger.info("[setup] staging %s ...", proj)
        results.append(_stage_project(proj))
    logger.info("[setup] staging summary: %s", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
