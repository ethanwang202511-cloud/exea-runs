#!/usr/bin/env bash
# ============================================================================
# Exea stage: START  (first run of the day). Identical to RESUME because the
# orchestrator is idempotent + resumable: it skips units already marked done in
# state/manifest.json (restored from yesterday's snapshot).
# ============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"
exec bash "$REPO/resume.sh"
