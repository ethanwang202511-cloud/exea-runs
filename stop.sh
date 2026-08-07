#!/usr/bin/env bash
# ============================================================================
# Exea stage: STOP  (cooperative halt request). Touch the STOP sentinel so a
# running orchestrator finishes its current unit, checkpoints, and exits
# cleanly at the next safe point. The runner's own 13:00 disk snapshot is the
# hard stop; this makes a *clean* stop the common case.
# ============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$REPO/state"
touch "$REPO/state/STOP"
echo "[stop] wrote $REPO/state/STOP — orchestrator will halt at next safe point."
