"""Download Replogle RPE1 essential Perturb-seq atlas from scperturb.

This kills the "K562-only" reviewer critique. Replogle 2022 audited two cell
lines (K562 leukemia, RPE1 retinal pigment epithelium); RPE1 has different
proliferation dynamics and stress response. If our audit's pop_mean dominance
holds on RPE1, the result is not a K562 artifact.
"""
import os, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
out = DATA / "replogle_rpe1_essential.h5ad"
if out.exists() and out.stat().st_size > 500_000_000:
    print(f"[download_rpe1] {out} already exists; size={out.stat().st_size/1e9:.2f}GB; skipping")
    sys.exit(0)
URL = "https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1"
cmd = ["curl", "-fL", "--proto", "=https", "-o", str(out), URL]
print(" ".join(cmd))
res = subprocess.run(cmd)
print(f"\n[download_rpe1] downloaded {out.stat().st_size/1e9:.2f}GB")
sys.exit(res.returncode)
