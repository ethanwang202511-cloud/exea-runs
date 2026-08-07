"""Data staging for BehaviorFM: AP-10K raw data + precomputed DINOv2 (and
cross-backbone) feature caches, plus SMOKE-mode synthetic data generation.

Real data layout (under EXEA_DATA_ROOT, default ``behaviorfm/data``):
    {DATA_ROOT}/ap10k/ap-10k/data/*.jpg
    {DATA_ROOT}/ap10k/ap-10k/annotations/ap10k-{split}-split1.json
    {DATA_ROOT}/cache/{cache_name}/{meta.json, features.npy, features.shape.json}

This mirrors exactly the layout the vendored scripts (precompute_features.py,
run_peft_baselines.py, ...) already assume, so real caches built by those
scripts (or by behaviorfm.runners.precompute_backbone_features, which reuses
their logic) are picked up with no path surgery.

SMOKE mode (EXEA_SMOKE=1): when the real AP-10K images/annotations or a
species' feature cache are absent, jobs call the synth_* helpers here to
generate tiny random tensors of the RIGHT SHAPE so the adaptation / PEFT /
decoder-FT code paths execute on CPU in seconds, with no download and no GPU.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch

from exea.staging import DataUnavailable, StagedItem

logger = logging.getLogger("behaviorfm.staging")

BEHAVIORFM_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("EXEA_DATA_ROOT", str(BEHAVIORFM_ROOT / "data")))
AP10K_ROOT = DATA_ROOT / "ap10k" / "ap-10k"
CACHE_ROOT = DATA_ROOT / "cache"

# Official AP-10K (Yu et al., NeurIPS 2021) distribution: the "10,015
# annotated images" bundle (images + COCO-style annotations for all splits)
# from https://github.com/AlexTheBad/AP-10K#download. Public, no auth.
# Overridable via EXEA_AP10K_URL (a direct http(s) zip URL, e.g. an internal
# mirror) or EXEA_AP10K_GDRIVE_ID, since scripted Google Drive downloads of
# large files are notoriously flaky in unattended settings.
AP10K_GDRIVE_FILE_ID_DEFAULT = "1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad"


def is_smoke() -> bool:
    # EXACT same expression as exea.orchestrator and intervenefm.staging, so
    # EXEA_SMOKE can't desync the orchestrator's headroom policy (which reads
    # smoke) from behaviorfm's actual run mode.
    return os.environ.get("EXEA_SMOKE", "") not in ("", "0", "false", "False")


# --------------------------------------------------------------------------- #
# Real-data staged items
# --------------------------------------------------------------------------- #

def ap10k_images_item() -> StagedItem:
    return StagedItem(name="ap10k_images", path=AP10K_ROOT / "data", is_dir=True)


def ap10k_annotations_item(split: str = "train", split_num: int = 1) -> StagedItem:
    fname = f"ap10k-{split}-split{split_num}.json"
    return StagedItem(name=f"ap10k_annotations_{split}{split_num}",
                       path=AP10K_ROOT / "annotations" / fname, min_bytes=1000)


def feature_cache_item(cache_name: str) -> StagedItem:
    cache_dir = CACHE_ROOT / cache_name
    return StagedItem(name=f"feature_cache_{cache_name}", path=cache_dir, is_dir=True)


def cache_dir_for(cache_name: str) -> Path:
    return CACHE_ROOT / cache_name


def cache_is_valid(cache_dir: Path) -> bool:
    """Stricter than StagedItem.present(): requires the actual files a
    CachedFeatureDataset needs, not just a non-empty directory."""
    return (cache_dir.exists()
            and (cache_dir / "meta.json").exists()
            and (cache_dir / "features.npy").exists()
            and (cache_dir / "features.shape.json").exists())


def checkpoint_item(ckpt_path: Path) -> StagedItem:
    return StagedItem(name=f"checkpoint_{ckpt_path.name}", path=ckpt_path, min_bytes=1000)


# --------------------------------------------------------------------------- #
# Real AP-10K download (setup-stage only; network required, per
# exea.staging's documented assumption that setup runs while network is up).
# --------------------------------------------------------------------------- #

def _ap10k_source() -> tuple[str, dict]:
    override_url = os.environ.get("EXEA_AP10K_URL", "").strip()
    if override_url:
        return "url", {"url": override_url}
    file_id = os.environ.get("EXEA_AP10K_GDRIVE_ID", "").strip() or AP10K_GDRIVE_FILE_ID_DEFAULT
    return "gdrive", {"file_id": file_id}


def _download_gdrive_file(file_id: str, dest: Path) -> None:
    """Best-effort large-file Google Drive download: the standard
    virus-scan-warning bypass ("confirm" token) that e.g. the `gdown`
    package implements internally. We do not depend on `gdown` (no package
    installs); this uses `requests`, which is already a transitive
    dependency of huggingface_hub/transformers in this environment. Tries
    the modern `drive.usercontent.google.com` endpoint first (works with a
    literal confirm=t for most large files), then falls back to the classic
    uc?export=download + scraped-confirm-token dance.
    """
    try:
        import requests
    except ImportError as e:
        raise DataUnavailable(
            f"'requests' not available to download AP-10K from Google Drive: {e}") from e

    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    resp = session.get(
        "https://drive.usercontent.google.com/download",
        params={"id": file_id, "export": "download", "confirm": "t"},
        stream=True, timeout=120,
    )
    if resp.status_code != 200 or "text/html" in resp.headers.get("content-type", ""):
        resp = session.get(
            "https://drive.google.com/uc",
            params={"id": file_id, "export": "download"},
            stream=True, timeout=120,
        )
        token = next((v for k, v in resp.cookies.items() if k.startswith("download_warning")), None)
        if token is None and resp.headers.get("content-type", "").startswith("text/html"):
            import re
            m = re.search(r"confirm=([0-9A-Za-z_-]+)", resp.text or "")
            token = m.group(1) if m else None
        if token:
            resp = session.get(
                "https://drive.google.com/uc",
                params={"id": file_id, "export": "download", "confirm": token},
                stream=True, timeout=120,
            )

    if resp.status_code != 200:
        raise DataUnavailable(
            f"Google Drive download of AP-10K (file_id={file_id}) returned HTTP {resp.status_code}. "
            f"If Drive is blocking scripted downloads on this host, set EXEA_AP10K_URL to a direct "
            f"mirror zip or EXEA_AP10K_GDRIVE_ID to an alternate file id.")

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
    os.replace(tmp, dest)


def _extract_and_relocate_ap10k(zip_path: Path, dest_root: Path) -> None:
    """Extract the AP-10K zip and relocate its data/+annotations/ subtree to
    dest_root, regardless of the exact top-level folder name inside the zip
    (the README's own extraction instructions rename it, so we search rather
    than assume one fixed name)."""
    import shutil
    import tempfile
    import zipfile

    dest_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dest_root.parent)) as td:
        tdp = Path(td)
        logger.info("[stage] extracting AP-10K zip %s -> %s", zip_path, tdp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tdp)
        candidates = [tdp]
        for p in tdp.iterdir():
            if p.is_dir():
                candidates.append(p)
                for c in p.iterdir():
                    if c.is_dir():
                        candidates.append(c)
        found = next(
            (c for c in candidates if (c / "data").is_dir() and (c / "annotations").is_dir()), None)
        if found is None:
            raise RuntimeError(
                f"could not locate data/+annotations/ inside the extracted AP-10K zip at {zip_path}; "
                f"top-level entries: {[p.name for p in tdp.iterdir()]}")
        dest_root.mkdir(parents=True, exist_ok=True)
        for sub in ("data", "annotations"):
            src, dst = found / sub, dest_root / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        logger.info("[stage] AP-10K relocated to %s", dest_root)


def _ap10k_fully_staged() -> bool:
    images_dir = AP10K_ROOT / "data"
    return (images_dir.is_dir() and any(images_dir.iterdir())
            and ap10k_annotations_item("train").present()
            and ap10k_annotations_item("val").present())


def _ensure_ap10k_bundle(dest: Path) -> None:  # noqa: ARG001 - StagedItem.fetch(dest) contract
    """Shared `fetch` for ALL THREE AP-10K StagedItems below: they all come
    from the SAME zip, so this checks the FULL bundle's presence (not just
    the one `dest` it happened to be called for) and is a no-op the 2nd/3rd
    time ensure_all() calls it. `dest` is accepted (StagedItem.fetch's
    contract) but intentionally unused beyond that idempotency check."""
    if _ap10k_fully_staged():
        logger.info("[stage] AP-10K already fully staged at %s", AP10K_ROOT)
        return
    mode, cfg = _ap10k_source()
    zip_path = DATA_ROOT / "_downloads" / "ap-10k.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "url":
        import urllib.request
        logger.info("[stage] downloading AP-10K from %s -> %s", cfg["url"], zip_path)
        tmp = zip_path.with_suffix(zip_path.suffix + ".tmp")
        urllib.request.urlretrieve(cfg["url"], tmp)
        os.replace(tmp, zip_path)
    else:
        logger.info("[stage] downloading AP-10K from Google Drive (file_id=%s) -> %s",
                    cfg["file_id"], zip_path)
        _download_gdrive_file(cfg["file_id"], zip_path)

    if not zip_path.exists() or zip_path.stat().st_size < 10_000_000:
        raise DataUnavailable(
            f"AP-10K download did not produce a plausible zip at {zip_path} "
            f"(got {zip_path.stat().st_size if zip_path.exists() else 0} bytes)")

    _extract_and_relocate_ap10k(zip_path, AP10K_ROOT)
    if not _ap10k_fully_staged():
        raise DataUnavailable(f"AP-10K extraction did not produce a valid layout at {AP10K_ROOT}")


def get_staged_items() -> list[StagedItem]:
    """Setup-stage entrypoint (see exea/stage_setup.py's `_stage_project`,
    which looks for `get_staged_items()`/`ensure_all_data()` on
    `<project>.staging`). Pre-downloads AP-10K once, while network is
    available, so the timed run window never needs network for raw data.

    All three items share the same `_ensure_ap10k_bundle` fetch (one zip
    contains images + every split's annotations); ensure_all() calling
    .ensure() on each independently is safe because that fetch checks the
    FULL bundle up front and no-ops for the 2nd/3rd item once the 1st has
    already populated everything.
    """
    return [
        StagedItem(name="ap10k_images", path=AP10K_ROOT / "data",
                   fetch=_ensure_ap10k_bundle, is_dir=True),
        StagedItem(name="ap10k_annotations_train1",
                   path=AP10K_ROOT / "annotations" / "ap10k-train-split1.json",
                   fetch=_ensure_ap10k_bundle, min_bytes=1000),
        StagedItem(name="ap10k_annotations_val1",
                   path=AP10K_ROOT / "annotations" / "ap10k-val-split1.json",
                   fetch=_ensure_ap10k_bundle, min_bytes=1000),
    ]


def prefetch() -> None:
    """Setup-stage entrypoint (see exea/stage_setup.py's `_stage_project`,
    which calls `<project>.staging.prefetch()` if present, while network is
    guaranteed available -- same contract as `get_staged_items()` above).

    Warms the local HF/timm weight cache for all 5 ViT backbones P05/P40/P41
    can load (facebook/dinov2-base, facebook/dino-vitb16, facebook/vit-mae-base,
    openai/clip-vit-base-patch16, timm eva02_base_patch14_224.mim_in22k), so
    the timed run window -- which the harness's own contract says assumes NO
    network -- never needs to hit the network for model weights, exactly like
    AP-10K's data is pre-downloaded above rather than fetched lazily during a
    run. Without this, `backbones.load_backbone(..., offline=False)` (called
    from P05/P40/P41's run_unit, i.e. during the timed window) would attempt
    a live download on first use of each backbone.

    Best-effort per backbone: importable via `behaviorfm.backbones` lazily
    (avoids importing torch/transformers/timm at staging-module import time);
    one backbone missing a dependency (e.g. timm not installed -> EVA-02) or
    failing to download must not stop the others, matching `ensure_all()`'s
    per-item fail-soft contract.
    """
    from behaviorfm import backbones as _backbones

    for name, spec in _backbones.BACKBONES.items():
        try:
            _backbones.load_backbone(name, "cpu", offline=False)
            logger.info("[prefetch] warmed backbone cache: %s (%s)", name, spec.hf_id)
        except Exception as e:
            logger.warning(
                "[prefetch] could not warm backbone %s (%s) -- non-fatal, "
                "jobs needing it will soft-skip at run time: %s", name, spec.hf_id, e)


# --------------------------------------------------------------------------- #
# SMOKE-mode synthetic data (right-shaped random tensors, no I/O)
# --------------------------------------------------------------------------- #

def synth_cached_batch(
    n: int,
    gid: int = 0,
    n_patches: int = 256,
    hidden: int = 768,
    n_keypoints: int = 17,
    species_name: str = "synth_species",
    seed: int = 0,
) -> dict:
    """A synthetic batch matching iti.data.feature_cache.collate_cached's
    output schema exactly, so it drops straight into the same adapt/eval
    code paths real cached-feature batches use."""
    g = torch.Generator().manual_seed(seed)
    feature = torch.randn(n, n_patches, hidden, generator=g)
    keypoints_xy = torch.rand(n, n_keypoints, 2, generator=g) * 224.0
    vis = (torch.rand(n, n_keypoints, generator=g) > 0.15).float()
    # guarantee at least one visible keypoint per example (avoid div-by-0 in
    # the vis-masked coord loss / PCK denom for degenerate random draws)
    vis[:, 0] = 1.0
    crop_side = torch.full((n,), 224.0)
    bbox_diag_orig = 80.0 + torch.rand(n, generator=g) * 220.0
    ids = torch.full((n,), gid, dtype=torch.long)
    return {
        "feature": feature,
        "identity_id": ids,
        "global_identity_id": ids,
        "keypoints_xy": keypoints_xy,
        "vis": vis,
        "bbox_diag_orig": bbox_diag_orig,
        "crop_side": crop_side,
        "species_name": [species_name] * n,
        "annot_id": list(range(n)),
        "image_id": list(range(n)),
    }


class SyntheticCachedDataset(torch.utils.data.Dataset):
    """Drop-in stand-in for iti.data.feature_cache.CachedFeatureDataset used
    in SMOKE mode: implements the exact same per-item schema, so it works
    unmodified with `DataLoader(..., collate_fn=collate_cached)` and with any
    vendored code written against a CachedFeatureDataset (e.g.
    iti.train.trainer.evaluate / coord_loss consumers), for jobs (P20) that
    need a real multi-epoch, shuffleable Dataset rather than a single batch.
    """

    def __init__(
        self, n: int, gid: int = 0, n_patches: int = 256, hidden: int = 768,
        n_keypoints: int = 17, species_name: str = "synth_species", seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.feature = torch.randn(n, n_patches, hidden, generator=g)
        self.keypoints_xy = torch.rand(n, n_keypoints, 2, generator=g) * 224.0
        self.vis = (torch.rand(n, n_keypoints, generator=g) > 0.15).float()
        self.vis[:, 0] = 1.0
        self.bbox_diag_orig = 80.0 + torch.rand(n, generator=g) * 220.0
        self.crop_side = torch.full((n,), 224.0)
        self.gid = int(gid)
        self.species_name = species_name
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {
            "feature": self.feature[i],
            "identity_id": self.gid,
            "global_identity_id": self.gid,
            "keypoints_xy": self.keypoints_xy[i],
            "vis": self.vis[i],
            "bbox_diag_orig": float(self.bbox_diag_orig[i]),
            "annot_id": i,
            "image_id": i,
            "species_name": self.species_name,
            "crop_side": float(self.crop_side[i]),
        }


def synth_pixel_batch(
    n: int,
    out_size: int = 224,
    n_keypoints: int = 17,
    seed: int = 0,
) -> dict:
    """Synthetic RAW-PIXEL batch for Tier-B (backbone-forward) smoke units:
    random images of the right shape, ImageNet-normalized scale (~N(0,1) is
    close enough for a frozen backbone's forward pass to run without NaNs)."""
    g = torch.Generator().manual_seed(seed)
    pixel_values = torch.randn(n, 3, out_size, out_size, generator=g) * 0.5
    keypoints_xy = torch.rand(n, n_keypoints, 2, generator=g) * out_size
    vis = (torch.rand(n, n_keypoints, generator=g) > 0.15).float()
    vis[:, 0] = 1.0
    crop_side = torch.full((n,), float(out_size))
    bbox_diag_orig = 80.0 + torch.rand(n, generator=g) * 220.0
    return {
        "pixel_values": pixel_values,
        "keypoints_xy": keypoints_xy,
        "vis": vis,
        "bbox_diag_orig": bbox_diag_orig,
        "crop_side": crop_side,
    }
