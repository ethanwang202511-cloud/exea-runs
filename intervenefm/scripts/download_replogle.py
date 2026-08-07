"""Download Replogle K562 essential Perturb-seq atlas from scperturb."""
import os, sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
out = DATA / "replogle_k562_essential.h5ad"
if out.exists() and out.stat().st_size > 1_000_000_000:
    print(f"[download_replogle] {out} already exists; skipping")
    sys.exit(0)
URL = "https://zenodo.org/records/13350497/files/ReplogleWeissman2022_K562_essential.h5ad?download=1"
cmd = ["curl", "-fL", "--proto", "=https", "-o", str(out), URL]
print(" ".join(cmd))
res = subprocess.run(cmd)
sys.exit(res.returncode)
