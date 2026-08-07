"""Data-staging specs for InterveneFM, on top of exea.staging's generic
StagedItem/require()/ensure_all() machinery.

Real datasets (Norman 2019, Replogle K562, Replogle RPE1, GO GAF) are fetched
once in the harness's `setup` stage and cached under EXEA_DATA_ROOT (default
`intervenefm/data/`, matching the path the vendored `src/data_*.py` loaders
already hardcode). Jobs call `require(...)` at run time; if a real dataset is
missing, that raises `DataUnavailable` and the harness soft-skips the unit —
*except* in smoke mode (`EXEA_SMOKE=1`), where we synthesize a tiny AnnData
(and a tiny GAF) locally so the full code path (load -> HVG -> split -> train
-> audit) executes without any download or GPU.

Fetchers REUSE the existing `scripts/download_*.py` logic by invoking those
scripts as subprocesses (they are self-contained, idempotent, and already
validated) rather than re-implementing the download logic here. There is no
existing GAF downloader in scripts/, so `fetch_gaf` is new (small, plain
urllib).
"""
from __future__ import annotations

import gzip
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from exea.staging import DataUnavailable, StagedItem, ensure_all, require

logger = logging.getLogger("intervenefm.staging")

# --------------------------------------------------------------------------- paths
INTERVENEFM_ROOT = Path(__file__).resolve().parent          # .../intervenefm
SCRIPTS_DIR = INTERVENEFM_ROOT / "scripts"
SRC_DIR = INTERVENEFM_ROOT / "src"

if str(INTERVENEFM_ROOT) not in sys.path:
    # The vendored scripts/src use `from src.xxx import yyy`, which requires
    # the intervenefm/ directory itself (not just the repo root) on sys.path.
    # We insert it defensively here so our helper modules work regardless of
    # how PYTHONPATH was set by the caller.
    sys.path.insert(0, str(INTERVENEFM_ROOT))

DATA_ROOT = Path(os.environ.get("EXEA_DATA_ROOT", str(INTERVENEFM_ROOT / "data")))

NORMAN_PATH = DATA_ROOT / "norman_2019.h5ad"
REPLOGLE_K562_PATH = DATA_ROOT / "replogle_k562_essential.h5ad"
REPLOGLE_RPE1_PATH = DATA_ROOT / "replogle_rpe1_essential.h5ad"
GAF_PATH = DATA_ROOT / "goa_human.gaf.gz"

SMOKE_DATA_DIR = DATA_ROOT / "_smoke"


def _is_smoke() -> bool:
    return os.environ.get("EXEA_SMOKE", "").strip() not in ("", "0", "false", "False")


# --------------------------------------------------------------------------- fetchers
def _run_download_script(script_name: str, produced_path: Path, dest: Path) -> None:
    """Run one of the existing vendored `download_*.py` scripts (they already
    know how to fetch + verify their own dataset) and make the result show up
    at `dest` (which may differ from the script's hardcoded output path if
    EXEA_DATA_ROOT points elsewhere)."""
    script = SCRIPTS_DIR / script_name
    if not script.exists():
        raise DataUnavailable(f"downloader script missing: {script}")
    logger.info("[stage] running %s (cwd=%s) to populate %s", script_name, INTERVENEFM_ROOT, produced_path)
    result = subprocess.run(
        [sys.executable, str(script)], cwd=str(INTERVENEFM_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} exited with code {result.returncode}")
    if not produced_path.exists():
        raise RuntimeError(f"{script_name} did not produce {produced_path}")
    if produced_path.resolve() != dest.resolve():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.symlink_to(produced_path.resolve())
        except OSError:
            import shutil
            shutil.copy2(produced_path, dest)


def fetch_norman(dest: Path) -> None:
    _run_download_script("download_norman.py", INTERVENEFM_ROOT / "data" / "norman_2019.h5ad", dest)


def fetch_replogle_k562(dest: Path) -> None:
    _run_download_script("download_replogle.py", INTERVENEFM_ROOT / "data" / "replogle_k562_essential.h5ad", dest)


def fetch_replogle_rpe1(dest: Path) -> None:
    _run_download_script("download_replogle_rpe1.py", INTERVENEFM_ROOT / "data" / "replogle_rpe1_essential.h5ad", dest)


def fetch_gaf(dest: Path) -> None:
    """GO annotation file for humans (goa_human.gaf.gz). No existing
    download_*.py script covers this, so this is a minimal, self-contained
    fetcher matching the style of the other download_*.py scripts."""
    import urllib.request

    url = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("[stage] downloading GO GAF from %s -> %s", url, dest)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)


NORMAN = StagedItem(name="norman_2019", path=NORMAN_PATH, fetch=fetch_norman, min_bytes=100_000_000)
REPLOGLE_K562 = StagedItem(name="replogle_k562_essential", path=REPLOGLE_K562_PATH,
                            fetch=fetch_replogle_k562, min_bytes=500_000_000)
REPLOGLE_RPE1 = StagedItem(name="replogle_rpe1_essential", path=REPLOGLE_RPE1_PATH,
                            fetch=fetch_replogle_rpe1, min_bytes=300_000_000)
GAF = StagedItem(name="goa_human_gaf", path=GAF_PATH, fetch=fetch_gaf, min_bytes=500_000)

ALL_ITEMS: List[StagedItem] = [NORMAN, REPLOGLE_K562, REPLOGLE_RPE1, GAF]


def ensure_all_data() -> Dict[str, bool]:
    """Setup-stage entry point: acquire every real dataset once."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return ensure_all(ALL_ITEMS)


# --------------------------------------------------------------------------- synthetic (smoke) data
def _make_synthetic_norman(seed: int = 0, n_genes: int = 200, pert_genes: Tuple[str, ...] | None = None,
                            n_ctrl: int = 90, n_single_per_gene: int = 16, n_double_per_pair: int = 16):
    """Tiny synthetic Norman-2019-shaped AnnData: control + singles + a handful
    of double perturbations, so build_split's 0/2 logic has real pairs to hold
    out and >=5 cells/pair for the audit's DEG filter."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    pert_genes = pert_genes or ("PG1", "PG2", "PG3", "PG4", "PG5", "PG6")
    pairs = [(pert_genes[i], pert_genes[j])
             for i in range(len(pert_genes)) for j in range(i + 1, len(pert_genes))]

    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    # Make sure the perturbable genes are actually present in var_names
    # (they occupy the first slots) so gene-identity computation resolves them.
    for gi, g in enumerate(pert_genes):
        gene_names[gi] = g

    rows = []  # list of (label, nperts, pert_genes_list)
    rows.append(("ctrl", 0, [], n_ctrl))
    for g in pert_genes:
        rows.append((g, 1, [g], n_single_per_gene))
    for (g1, g2) in pairs:
        rows.append((f"{g1}_{g2}", 2, [g1, g2], n_double_per_pair))

    obs_pert, obs_nperts, cell_pert_lists = [], [], []
    n_cells = sum(r[3] for r in rows)
    X = rng.lognormal(mean=1.0, sigma=0.6, size=(n_cells, n_genes)).astype(np.float32)
    # Give perturbed conditions a mild, gene-specific expression shift so DEGs
    # are non-degenerate (real signal for the DE-Spearman audit to detect).
    cursor = 0
    effect_rng = np.random.default_rng(seed + 1)
    for label, nperts, plist, n in rows:
        sl = slice(cursor, cursor + n)
        if nperts > 0:
            shifted = effect_rng.choice(n_genes, size=min(20, n_genes), replace=False)
            X[sl][:, shifted] *= effect_rng.uniform(1.3, 2.2, size=len(shifted))
        obs_pert.extend([label] * n)
        obs_nperts.extend([nperts] * n)
        cell_pert_lists.extend([plist] * n)
        cursor += n

    obs = pd.DataFrame({
        "perturbation": obs_pert,
        "nperts": obs_nperts,
    }, index=[f"cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    return adata


def _make_synthetic_replogle(seed: int = 0, n_genes: int = 200, n_pert_genes: int = 6,
                              n_ctrl: int = 90, n_per_gene: int = 20):
    """Tiny synthetic Replogle-shaped AnnData: control + N single-gene
    perturbations, each with enough cells to survive a small min_cells_per_gene
    threshold in build_split_replogle."""
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]
    pert_genes = [f"RG{i}" for i in range(n_pert_genes)]
    for gi, g in enumerate(pert_genes):
        gene_names[gi] = g

    n_cells = n_ctrl + n_pert_genes * n_per_gene
    X = rng.lognormal(mean=1.0, sigma=0.6, size=(n_cells, n_genes)).astype(np.float32)
    obs_pert = ["control"] * n_ctrl
    effect_rng = np.random.default_rng(seed + 1)
    cursor = n_ctrl
    for g in pert_genes:
        sl = slice(cursor, cursor + n_per_gene)
        shifted = effect_rng.choice(n_genes, size=min(20, n_genes), replace=False)
        X[sl][:, shifted] *= effect_rng.uniform(1.3, 2.2, size=len(shifted))
        obs_pert.extend([g] * n_per_gene)
        cursor += n_per_gene

    obs = pd.DataFrame({
        "perturbation": obs_pert,
        "nperts": [0 if p == "control" else 1 for p in obs_pert],
    }, index=[f"cell{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    return adata


def _make_synthetic_gaf(dest: Path, genes: List[str], seed: int = 0, n_terms: int = 12) -> None:
    """Tiny synthetic GAF matching the real GAF column layout that
    compute_gene_identity_table_go parses: col[2]=symbol, col[4]=GO term,
    >=15 tab-separated columns, comment lines start with '!'.

    NB: the parser does `line.strip().split('\\t')`, and `str.strip()` eats
    trailing TABS too (they're whitespace) -- so trailing empty columns must
    be avoided, or a line's real column count silently collapses below the
    `len(parts) < 15` floor and the whole line is dropped. Every column here
    is therefore filled with a harmless non-empty placeholder.
    """
    rng = np.random.default_rng(seed)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt") as f:
        f.write("!gaf-version: 2.2 (synthetic smoke fixture)\n")
        for g in genes:
            n_annot = int(rng.integers(1, 4))
            terms = rng.choice(n_terms, size=n_annot, replace=False)
            for t in terms:
                cols = [
                    "UniProtKB", f"SYN{g}", g, "involved_in", f"GO:{7000000 + int(t):07d}",
                    "GO_REF:0000synth", "IEA", "-", "P", g, f"{g}_synonym",
                    "protein", "taxon:9606", "20260101", "SYNTH",
                ]
                f.write("\t".join(cols) + "\n")


def ensure_smoke_norman(seed: int = 0) -> Path:
    path = SMOKE_DATA_DIR / "norman_2019.h5ad"
    if not path.exists():
        adata = _make_synthetic_norman(seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
        logger.info("[smoke] synthesized Norman fixture: %s (%s)", path, adata.shape)
    return path


def ensure_smoke_replogle(which: str = "k562", seed: int = 0) -> Path:
    fname = "replogle_k562_essential.h5ad" if which == "k562" else "replogle_rpe1_essential.h5ad"
    path = SMOKE_DATA_DIR / fname
    if not path.exists():
        adata = _make_synthetic_replogle(seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(path)
        logger.info("[smoke] synthesized Replogle(%s) fixture: %s (%s)", which, path, adata.shape)
    return path


def ensure_smoke_gaf(pert_genes: List[str] | None = None, seed: int = 0) -> Path:
    """One shared synthetic GAF fixture covers the union of ALL smoke
    datasets' perturbable gene names (Norman's PG* and Replogle's RG*), not
    just whichever dataset asks first -- otherwise the first caller "wins"
    the cache and every other dataset's CPG_GO variant would resolve to
    all-zero identity vectors (not a crash, just a silently-weak smoke test)."""
    path = SMOKE_DATA_DIR / "goa_human.gaf.gz"
    if not path.exists():
        union = sorted(set(pert_genes or []) | {f"PG{i}" for i in range(1, 7)} | {f"RG{i}" for i in range(6)})
        _make_synthetic_gaf(path, union, seed=seed)
        logger.info("[smoke] synthesized GAF fixture covering %d genes: %s", len(union), path)
    return path


# --------------------------------------------------------------------------- unified accessors
def resolve_norman_path() -> Path:
    """Return a usable Norman h5ad path: staged real data, or (smoke mode
    only) a synthesized tiny fixture. Raises DataUnavailable otherwise."""
    if _is_smoke():
        return ensure_smoke_norman()
    return require(NORMAN)


def resolve_replogle_path(which: str) -> Path:
    if _is_smoke():
        return ensure_smoke_replogle(which)
    return require(REPLOGLE_K562 if which == "k562" else REPLOGLE_RPE1)


def resolve_gaf_path(pert_genes: List[str] | None = None) -> Path:
    if _is_smoke():
        return ensure_smoke_gaf(pert_genes or [f"PG{i}" for i in range(1, 5)])
    return require(GAF)
