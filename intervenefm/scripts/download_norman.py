"""
Download Norman et al. 2019 Perturb-seq data for the intervention-gap audit.

Source: Roohani et al. (GEARS) preprocessing — Norman 2019 is hosted as a
preprocessed AnnData on the GEARS data Zenodo / Dataverse, but most
practical downloads come from the Replogle/Norman scperturb collection.

We try multiple sources in order:
1. scperturb processed AnnData on FigShare/Zenodo
2. GEARS pertdata download script
3. Direct GEO GSE133344 raw counts (last resort)

Output: data/norman_2019.h5ad
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT = DATA_DIR / "norman_2019.h5ad"

if OUT.exists() and OUT.stat().st_size > 1_000_000:
    print(f"[download_norman] {OUT} already exists ({OUT.stat().st_size/1e6:.1f} MB). Skipping.")
    sys.exit(0)

# Source 1: scperturb processed AnnData (most reliable; ~600 MB)
# https://zenodo.org/records/10044268 (NormanWeissman2019)
SCPERTURB_URL = (
    "https://zenodo.org/records/10044268/files/NormanWeissman2019_filtered.h5ad?download=1"
)

print(f"[download_norman] Downloading from scperturb Zenodo to {OUT} ...")
try:
    def progress(blocks, blocksize, total):
        if total > 0:
            pct = 100 * blocks * blocksize / total
            if blocks % 100 == 0:
                print(f"  {pct:.1f}% ({blocks * blocksize / 1e6:.1f}/{total/1e6:.1f} MB)", flush=True)
    urllib.request.urlretrieve(SCPERTURB_URL, OUT, reporthook=progress)
    print(f"[download_norman] Done. {OUT} = {OUT.stat().st_size/1e6:.1f} MB")
except Exception as e:
    print(f"[download_norman] FAILED: {e}", file=sys.stderr)
    sys.exit(1)
