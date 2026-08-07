"""Multi-backbone loading for the Tier-B (backbone-forward) PEFT panel + P40
cross-backbone feature precompute.

This is NEW code (no vendored equivalent for anything but DINOv2). For
DINOv2 we deliberately reuse the validated ``iti.model.frozen_backbone
.FrozenDINOv2Base`` for the no-grad / cached-feature path, and only reach
into its already-loaded ``.model`` submodule for the grad-enabled Tier-B
path (never reimplementing its loading logic).

Backbone loading gotchas (see top-level task spec — get these EXACTLY right):
  * facebook/dinov2-base   : patch 14, 256 tokens @224, hidden 768, drop CLS.
  * facebook/dino-vitb16   : patch 16, 196 tokens, drop CLS.
  * facebook/vit-mae-base  : patch 16, 196 tokens, MUST set mask_ratio=0.0
    (default 0.75 silently drops 75% of patches).
  * openai/clip-vit-base-patch16: patch 16, 196 tokens, CLIPVisionModel,
    drop CLS, CLIP image mean/std (NOT ImageNet).
  * EVA-02 (timm eva02_base_patch14_224.mim_in22k): patch 14, 256 tokens,
    drop CLS. timm may be absent locally -> soft-skip (BackboneUnavailable).

HARD ROCm RULES observed here: no flash_attn/xformers/apex/bitsandbytes;
HF models are loaded with attn_implementation="sdpa" where supported;
no hardcoded .cuda() (caller passes device); optional-heavy imports (timm)
are wrapped in try/except at point of use.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Make the vendored `iti` package importable regardless of how this module
# itself was imported (as `behaviorfm.backbones` or as a bare top-level
# `backbones` module, per the exact convention every vendored script under
# behaviorfm/scripts/ already uses).
_BFM_ROOT = Path(__file__).resolve().parent
_SRC = str(_BFM_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

logger = logging.getLogger("behaviorfm.backbones")


class BackboneUnavailable(Exception):
    """Backbone weights/deps not available locally (missing dep, not cached,
    network error while offline). Callers should convert this to
    exea.staging.DataUnavailable so the orchestrator soft-skips the unit."""


# ImageNet normalization (DINOv2 / DINOv1 / MAE) vs CLIP normalization.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    patch: int
    grid: int          # sqrt(n_patches) at 224x224 input
    hidden: int
    kind: str           # "dinov2" | "hf_vit" | "hf_mae" | "hf_clip" | "timm"
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD


BACKBONES: dict[str, BackboneSpec] = {
    "dinov2": BackboneSpec("dinov2", "facebook/dinov2-base", 14, 16, 768, "dinov2"),
    "dino_v1": BackboneSpec("dino_v1", "facebook/dino-vitb16", 16, 14, 768, "hf_vit"),
    "mae": BackboneSpec("mae", "facebook/vit-mae-base", 16, 14, 768, "hf_mae"),
    "clip": BackboneSpec("clip", "openai/clip-vit-base-patch16", 16, 14, 768, "hf_clip",
                          mean=CLIP_MEAN, std=CLIP_STD),
    "eva02": BackboneSpec("eva02", "eva02_base_patch14_224.mim_in22k", 14, 16, 768, "timm"),
}

# Per-backbone-kind module-name targets for peft LoraConfig / IA3Config
# (Tier-B: LoRA-on-backbone(Q,V), IA3). Best-effort; timm's fused qkv linear
# means "Q,V-only" LoRA is not separable there, so we target the whole
# fused qkv projection for eva02 (labeled clearly in results as a caveat).
#
# CORRECTNESS GOTCHA (dinov2 / hf_vit / hf_mae only): HF's ViT-family layer
# naming has TWO distinct nn.Linear modules whose qualified name ends with
# the literal suffix "output.dense" -- the self-attention output projection
# (".../attention.output.dense") AND the MLP's second/down-projection
# (".../output.dense", the intended IA3 "feedforward" target). peft's
# list-based target matching is `key.endswith(f".{target}")`
# (peft.tuners.tuners_utils.check_target_module_exists), so a plain
# ["key", "value", "output.dense"] list matches BOTH linears and would
# incorrectly also treat the attention-output projection as a feedforward
# module. peft's `target_modules`/`feedforward_modules` accept a single
# regex string instead of a list (matched via `re.fullmatch`, see
# peft.utils.other.match_target_against_key) precisely for cases like this.
# The negative lookbehind below matches "...output.dense" only when it is
# NOT immediately preceded by "attention", i.e. only the MLP's output
# projection -- never the attention block's own output.dense.
_HF_VIT_IA3_MLP_OUTPUT_RE = r".*(?<!attention)\.output\.dense$"
_HF_VIT_IA3_TARGETS_RE = r".*\.(?:key|value)$|" + _HF_VIT_IA3_MLP_OUTPUT_RE

PEFT_TARGETS: dict[str, dict] = {
    "dinov2": {"lora": ["query", "value"], "ia3": _HF_VIT_IA3_TARGETS_RE,
               "ia3_ff": _HF_VIT_IA3_MLP_OUTPUT_RE},
    "hf_vit": {"lora": ["query", "value"], "ia3": _HF_VIT_IA3_TARGETS_RE,
               "ia3_ff": _HF_VIT_IA3_MLP_OUTPUT_RE},
    "hf_mae": {"lora": ["query", "value"], "ia3": _HF_VIT_IA3_TARGETS_RE,
               "ia3_ff": _HF_VIT_IA3_MLP_OUTPUT_RE},
    "hf_clip": {"lora": ["q_proj", "v_proj"], "ia3": ["k_proj", "v_proj", "mlp.fc2"],
                "ia3_ff": ["mlp.fc2"]},
    "timm": {"lora": ["qkv"], "ia3": ["qkv", "mlp.fc2"], "ia3_ff": ["mlp.fc2"]},
}


@contextlib.contextmanager
def _offline(enabled: bool):
    """Force HF hub calls to be local-cache-only (no network) without
    touching any vendored loading code. Used during EXEA_SMOKE=1 so the
    smoke test never triggers a background download."""
    if not enabled:
        yield
        return
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class BackboneHandle:
    """Common interface over the five backbones.

    ``patch_features(pixel_values, grad=False)`` always returns
    ``(B, N_patches, hidden)`` with the CLS token already dropped.

    grad=False -> the whole forward runs under torch.no_grad() (the correct
        mode for Tier-A cached-feature precompute).
    grad=True  -> forward runs WITHOUT no_grad (backbone params still have
        requires_grad=False, but gradients can flow through the frozen
        weight matmuls into any inserted PEFT parameters). This is the mode
        Tier-B PEFT (LoRA-on-backbone, AdaptFormer, IA3, VPT) requires,
        because those methods insert trainable modules INSIDE the ViT.
    """

    def __init__(self, spec: BackboneSpec, model: nn.Module, raw_model: nn.Module):
        self.spec = spec
        self.model = model          # the object we call .train()/.eval() on
        self.raw_model = raw_model  # the underlying HF/timm module we forward through
        self.hidden = spec.hidden
        self.grid = spec.grid
        self.patch = spec.patch
        self.n_patches = spec.grid * spec.grid
        # VPT (behaviorfm.peft_panel) injects `extra_prefix_tokens` learnable
        # prompt tokens right after CLS; when active, patch_features() must
        # drop CLS + prompts (not just CLS) to keep the patch count == grid^2.
        self.extra_prefix_tokens = 0

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        # Backbone stays frozen regardless; .train() only matters for any
        # inserted PEFT submodules (dropout etc. in adapters).
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def _raw_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        kind = self.spec.kind
        prefix = 1 + self.extra_prefix_tokens  # CLS (+ VPT prompts, if attached)
        if kind in ("dinov2", "hf_vit", "hf_mae", "hf_clip"):
            out = self.raw_model(pixel_values=pixel_values).last_hidden_state
            return out[:, prefix:]
        if kind == "timm":
            feats = self.raw_model.forward_features(pixel_values)
            # timm ViT-style forward_features returns (B, 1+N, hidden) with
            # the CLS token prepended (eva02_base has a class token).
            if feats.dim() == 3 and feats.shape[1] == self.n_patches + prefix:
                return feats[:, prefix:]
            if feats.dim() == 3 and feats.shape[1] == self.n_patches:
                return feats
            raise BackboneUnavailable(
                f"unexpected timm forward_features shape {tuple(feats.shape)} "
                f"for {self.spec.hf_id} (expected N={self.n_patches} or N+{prefix})"
            )
        raise ValueError(kind)

    def patch_features(self, pixel_values: torch.Tensor, grad: bool = False) -> torch.Tensor:
        if grad:
            return self._raw_forward(pixel_values)
        with torch.no_grad():
            return self._raw_forward(pixel_values)


def _load_dinov2(spec: BackboneSpec, device, offline: bool) -> BackboneHandle:
    # Reuse the validated wrapper exactly as-is for loading + freezing.
    from iti.model.frozen_backbone import FrozenDINOv2Base
    with _offline(offline):
        fb = FrozenDINOv2Base(hf_id=spec.hf_id)
    fb.to(device)
    fb.eval()
    return BackboneHandle(spec, model=fb, raw_model=fb.model)


def _load_hf_vit(spec: BackboneSpec, device, offline: bool) -> BackboneHandle:
    from transformers import AutoModel
    with _offline(offline):
        try:
            model = AutoModel.from_pretrained(spec.hf_id, attn_implementation="sdpa")
        except TypeError:
            # older/newer transformers may not accept attn_implementation here
            model = AutoModel.from_pretrained(spec.hf_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return BackboneHandle(spec, model=model, raw_model=model)


def _load_hf_mae(spec: BackboneSpec, device, offline: bool) -> BackboneHandle:
    from transformers import AutoConfig, ViTMAEModel
    with _offline(offline):
        cfg = AutoConfig.from_pretrained(spec.hf_id)
        # CRITICAL correctness gotcha: default mask_ratio=0.75 drops 75% of
        # patches. We need ALL patches for a fair cached/uncached comparison.
        cfg.mask_ratio = 0.0
        model = ViTMAEModel.from_pretrained(spec.hf_id, config=cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    assert float(model.config.mask_ratio) == 0.0, "MAE mask_ratio did not propagate"
    return BackboneHandle(spec, model=model, raw_model=model)


def _load_hf_clip(spec: BackboneSpec, device, offline: bool) -> BackboneHandle:
    from transformers import CLIPVisionModel
    with _offline(offline):
        try:
            model = CLIPVisionModel.from_pretrained(spec.hf_id, attn_implementation="sdpa")
        except TypeError:
            model = CLIPVisionModel.from_pretrained(spec.hf_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return BackboneHandle(spec, model=model, raw_model=model)


def _load_timm(spec: BackboneSpec, device, offline: bool) -> BackboneHandle:
    try:
        import timm
    except ImportError as e:
        raise BackboneUnavailable(f"timm not installed; EVA-02 unavailable: {e}") from e
    with _offline(offline):
        try:
            model = timm.create_model(spec.hf_id, pretrained=True, num_classes=0)
        except Exception as e:  # offline + not cached, or any timm hub error
            raise BackboneUnavailable(f"timm could not load {spec.hf_id}: {e}") from e
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(device)
    return BackboneHandle(spec, model=model, raw_model=model)


_LOADERS = {
    "dinov2": _load_dinov2,
    "hf_vit": _load_hf_vit,
    "hf_mae": _load_hf_mae,
    "hf_clip": _load_hf_clip,
    "timm": _load_timm,
}


def load_backbone(name: str, device, offline: bool) -> BackboneHandle:
    """Load a backbone by short name. Raises BackboneUnavailable (never a
    bare exception from HF/timm internals) on any missing dep / missing
    local cache / network failure, so callers can soft-skip cleanly."""
    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choices: {sorted(BACKBONES)}")
    spec = BACKBONES[name]
    loader = _LOADERS[spec.kind]
    try:
        handle = loader(spec, device, offline)
    except BackboneUnavailable:
        raise
    except Exception as e:
        raise BackboneUnavailable(f"failed to load backbone {name!r} ({spec.hf_id}): {e}") from e
    logger.info("[backbone] loaded %s (%s) on %s", name, spec.hf_id, device)
    return handle


def renormalize(pixel_values: torch.Tensor, from_mean, from_std, to_mean, to_std) -> torch.Tensor:
    """Convert a batch normalized with (from_mean, from_std) to (to_mean, to_std).

    AP10KPoseDataset bakes in ImageNet normalization; CLIP needs its own
    mean/std (explicit gotcha in the task spec), so cross-backbone precompute
    re-normalizes rather than re-implementing the crop/resize pipeline.
    """
    device = pixel_values.device
    dtype = pixel_values.dtype
    fm = torch.tensor(from_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    fs = torch.tensor(from_std, device=device, dtype=dtype).view(1, 3, 1, 1)
    tm = torch.tensor(to_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    ts = torch.tensor(to_std, device=device, dtype=dtype).view(1, 3, 1, 1)
    raw = pixel_values * fs + fm
    return (raw - tm) / ts
