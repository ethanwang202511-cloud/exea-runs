#!/usr/bin/env bash
# ============================================================================
# Exea stage: SETUP  (runs once, with network; before the timed run window)
# ----------------------------------------------------------------------------
# 1. Install core deps (must succeed). torch is NOT reinstalled — the base
#    image is expected to be rocm/pytorch with a working ROCm PyTorch.
# 2. Install optional deps best-effort (|| true) — GEARS/PyG etc. may not build
#    on ROCm; the run proceeds without them (dependent jobs soft-skip).
# 3. Pre-download ALL datasets + model weights to the persistent disk, because
#    network is NOT assumed during the timed run window.
# Idempotent: safe to re-run every day.
# ============================================================================
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY="${EXEA_PYTHON:-python3}"
echo "[setup] repo=$REPO  python=$($PY --version 2>&1)"

echo "[setup] === core dependencies ==="
$PY -m pip install --no-input -r requirements-core.txt || {
  echo "[setup] FATAL: core dependency install failed"; exit 1;
}

echo "[setup] === optional dependencies (best-effort, ROCm may skip) ==="
# torch_geometric core only; never torch-scatter/sparse/pyg-lib (fragile on ROCm)
$PY -m pip install --no-input -r requirements-optional.txt || \
  echo "[setup] WARN: some optional deps failed to install (jobs will soft-skip)"

echo "[setup] === torch / device report ==="
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "| cuda(is HIP on ROCm):", torch.cuda.is_available(),
      "| hip:", getattr(torch.version, "hip", None))
try:
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
except Exception as e:
    print("gpu query error:", e)
PYEOF

mkdir -p state state/logs results

echo "[setup] === pre-download datasets + weights (idempotent, per-project) ==="
# Per-project path isolation mirrors resume.sh so each project's vendored
# `src`/`scripts` imports resolve within that project (no cross-project clash).
for proj in behaviorfm intervenefm; do
  echo "[setup] staging $proj ..."
  PYTHONPATH="$REPO:$REPO/$proj:${PYTHONPATH:-}" \
    $PY -m exea.stage_setup --project "$proj" || \
    echo "[setup] WARN: staging $proj had failures; jobs needing that data will soft-skip"
done

echo "[setup] done."
