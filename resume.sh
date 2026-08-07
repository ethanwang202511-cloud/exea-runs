#!/usr/bin/env bash
# ============================================================================
# Exea stage: RESUME  (== START). Runs the orchestrator once PER PROJECT, in
# separate processes, so the two vendored `src`/`scripts` packages never
# collide and one project's import error cannot touch the other. Both processes
# share ONE absolute wall-clock deadline (EXEA_DEADLINE_EPOCH) so together they
# stay inside the daily slot. Project order alternates each day for fairness.
# The orchestrator is idempotent + resumable via state/manifest.json.
# ============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="${EXEA_PYTHON:-python3}"
export EXEA_RUN_BUDGET_SECONDS="${EXEA_RUN_BUDGET_SECONDS:-9900}"   # 2h45m default
mkdir -p state state/logs results

# One absolute deadline shared by both project subprocesses.
export EXEA_DEADLINE_EPOCH="$($PY -c 'import time,os;print(time.time()+float(os.environ["EXEA_RUN_BUDGET_SECONDS"]))')"

# Clear stale cooperative-stop sentinel.
rm -f "$REPO/state/STOP" 2>/dev/null || true

# Fair alternation: even days behaviorfm-first, odd days intervenefm-first.
CNT_FILE="$REPO/state/day_counter"
CNT=0; [ -f "$CNT_FILE" ] && CNT="$(cat "$CNT_FILE" 2>/dev/null || echo 0)"
echo $((CNT + 1)) > "$CNT_FILE"
if [ $((CNT % 2)) -eq 0 ]; then ORDER="behaviorfm intervenefm"; else ORDER="intervenefm behaviorfm"; fi
ORDER="${EXEA_PROJECT_ORDER:-$ORDER}"
echo "[resume] deadline_epoch=$EXEA_DEADLINE_EPOCH  order=$ORDER"

for proj in $ORDER; do
  echo "[resume] === project: $proj ==="
  PYTHONPATH="$REPO:$REPO/$proj:${PYTHONPATH:-}" \
    $PY -m exea.orchestrator --project "$proj" || echo "[resume] $proj orchestrator exited nonzero (continuing)"
done

# Persist results (best-effort) even if the disk snapshot is lost.
bash "$REPO/save.sh" || true
