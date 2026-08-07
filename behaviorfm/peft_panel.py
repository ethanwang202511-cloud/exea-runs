"""Tier-A (cached-feature) and Tier-B (backbone-forward) PEFT method builders
and the shared adapt/eval loop.

THE CRITICAL CORRECTNESS POINT (from the task's literature audit): ITA caches
FROZEN backbone patch features and trains only a token + small cross-attention
decoder over the cache. PEFT baselines differ in whether cached features are
valid for them:

  Tier A (cached-feature, ITA-equal compute) — operate ONLY on the decoder
  over cached patch features:
      ita             : train only the novel id-token (~768 params)
      decoder_ft       : train the whole PoseDecoder (~8.5M params)
      lora_decoder     : LoRA (r=4) on every nn.Linear in the decoder
      bitfit_decoder   : train only decoder bias tensors

  Tier B (backbone-forward, MORE compute) — insert trainable modules INSIDE
  the frozen ViT, so cached features are INVALID; every training step re-runs
  the full backbone forward (backbone weights frozen, grad flows through the
  inserted PEFT params):
      lora_backbone    : LoRA on backbone attention Q,V projections (peft)
      ia3              : IA3 rescaling of backbone K,V + MLP-output (peft)
      adaptformer      : parallel bottleneck adapter across each ViT block
      vpt              : p learnable tokens prepended to the ViT input (VPT-shallow)

Tier-A functions below NEVER touch a backbone. Tier-B functions below NEVER
accept a CachedFeatureDataset batch — they take raw pixel batches and a
`BackboneHandle` (behaviorfm.backbones) and always forward through it with
grad=True. This split is enforced structurally (different function
signatures / module boundaries), not just by convention, precisely so Tier-B
can never be silently fed cached features.

Heavily reuses vendored, already-validated code:
  iti.model.iti.ITIModel / ITIConfig
  iti.baselines.peft.LoRALinear / wrap_decoder_with_lora   (Tier-A LoRA)
  iti.eval.pck.per_keypoint_distance_normalized / pck_metrics / bootstrap_ci
  iti.train.trainer.TrainConfig / evaluate / coord_loss     (P20 full-FT sweep)
The adapt-loop numerics (5-shot support, SGD momentum=0.9, matched budget)
mirror scripts/run_peft_baselines.py and scripts/run_h1_pareto_adapt.py
exactly, generalized to accept batches directly (real cache OR synthetic)
instead of loading via CachedFeatureDataset + DataLoader.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_BFM_ROOT = Path(__file__).resolve().parent
_SRC = str(_BFM_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from iti.model.iti import ITIModel, ITIConfig                       # noqa: E402
from iti.baselines.peft import LoRALinear, wrap_decoder_with_lora   # noqa: E402
from iti.eval.pck import (                                          # noqa: E402
    per_keypoint_distance_normalized, pck_metrics, bootstrap_ci,
)

from behaviorfm.backbones import BackboneHandle, PEFT_TARGETS, BackboneUnavailable  # noqa: E402

logger = logging.getLogger("behaviorfm.peft_panel")


class DependencyUnavailable(Exception):
    """An optional heavy dependency (peft, timm) needed by a Tier-B method
    is not importable. Callers should convert this to DataUnavailable."""


# --------------------------------------------------------------------------- #
# Base model / checkpoint loading (shared by Tier A and the Pareto grid)
# --------------------------------------------------------------------------- #

def smoke_iti_config() -> ITIConfig:
    """Small-but-architecturally-faithful config for smoke mode: same hidden
    dim / grid (256 patches) as real DINOv2 so cached-feature shapes match,
    fewer training identities and a smaller decoder for CPU speed."""
    return ITIConfig(
        n_training_identities=5, hidden=768, grid=16, n_keypoints=17,
        decoder_layers=1, decoder_dim=96, heatmap_size=64,
        use_interpolation_head=True, interp_head_version="v1",
    )


def get_base_state_and_cfg(
    ckpt_path: Path, device, smoke: bool,
) -> tuple[dict, ITIConfig]:
    """Return (state_dict_bundle, cfg) for a fresh ITIModel to load from.

    Real path: load the Stage-1/Q2 checkpoint exactly as every vendored
    script does (state["model_state_dict"], state["cfg"]).
    Smoke / missing-checkpoint path: build ONE randomly-initialized model,
    snapshot its own state_dict, and hand that back — every caller then
    "loads a fresh model from a checkpoint" via the identical code path
    real runs use, just seeded from random weights instead of a trained ckpt.
    """
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                            if k in ITIConfig.__dataclass_fields__})
        return state, cfg
    if not smoke:
        from exea.staging import DataUnavailable
        raise DataUnavailable(f"checkpoint not staged at {ckpt_path}")
    cfg = smoke_iti_config()
    # backbone=nn.Identity(): every code path below is cached-feature-only
    # (forward_from_features / forward_with_token), which never touches
    # model.backbone. Skipping the real FrozenDINOv2Base construction here
    # avoids an unnecessary (and, off the dev machine, possibly network-
    # dependent) backbone load on every single adaptation unit.
    model = ITIModel(cfg, backbone=nn.Identity()).to(device)
    state = {"model_state_dict": model.state_dict(), "cfg": asdict(cfg)}
    logger.info("[smoke] no checkpoint at %s; using a fresh random-init ITIModel", ckpt_path)
    return state, cfg


# --------------------------------------------------------------------------- #
# Tier A: cached-feature adaptation methods
# --------------------------------------------------------------------------- #

TIER_A_METHODS = ["ita", "decoder_ft", "lora_decoder", "bitfit_decoder"]


def build_adapted_model_tier_a(
    state: dict, cfg: ITIConfig, device, method: str, r: int = 4, alpha: float = 8.0,
) -> tuple[ITIModel, nn.Parameter, list[nn.Parameter], int]:
    """Load a fresh ITIModel and configure trainable params for a Tier-A
    method. Mirrors scripts/run_peft_baselines.py::build_adapted_model,
    extended with a `decoder_ft` (full-decoder fine-tune) method — the
    4th Tier-A anchor the P10 panel needs (ITA / decoder_ft / lora_decoder /
    bitfit_decoder), all at the SAME matched adaptation budget so the
    comparison is apples-to-apples.
    """
    # backbone=nn.Identity(): every code path below is cached-feature-only
    # (forward_from_features / forward_with_token), which never touches
    # model.backbone. Skipping the real FrozenDINOv2Base construction here
    # avoids an unnecessary (and, off the dev machine, possibly network-
    # dependent) backbone load on every single adaptation unit.
    model = ITIModel(cfg, backbone=nn.Identity()).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        mean_tok = model.tokens.emb.weight.detach().mean(dim=0)

    if method == "ita":
        # Paper anchor: only the novel id-token is trained. Caller overrides
        # this placeholder with the interpolation-head init (see
        # init_token_via_interp_head) once support features are available.
        tok = nn.Parameter(mean_tok.clone())
        extra_params: list[nn.Parameter] = []
    elif method == "decoder_ft":
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)
        extra_params = list(model.decoder.parameters())
        for p in extra_params:
            p.requires_grad_(True)
    elif method == "lora_decoder":
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)
        wrap_decoder_with_lora(model.decoder, r=r, alpha=alpha)
        model.to(device)
        extra_params = [p for p in model.decoder.parameters() if p.requires_grad]
    elif method == "bitfit_decoder":
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)
        extra_params = [p for name, p in model.decoder.named_parameters()
                         if name.endswith(".bias")]
        for p in extra_params:
            p.requires_grad_(True)
    else:
        raise ValueError(f"unknown Tier-A method: {method!r}")

    trainable = ([tok] + extra_params) if tok.requires_grad else extra_params
    n_trainable = sum(p.numel() for p in trainable)
    return model, tok, extra_params, n_trainable


def init_token_via_interp_head(model: ITIModel, sup_f: torch.Tensor, device) -> torch.Tensor:
    """Initialise a novel id-token via the interpolation head (paper's
    strategy) — identical to run_peft_baselines.py / run_h1_pareto_adapt.py."""
    id_bank = model.tokens.emb.weight.detach()
    with torch.no_grad():
        ex_feat = sup_f.mean(dim=0).unsqueeze(0)
        init_tok, _ = model.interpolation_head(ex_feat, id_bank)
        init_tok = init_tok.squeeze(0)
    return init_tok


def sample_support(train_batch: dict, k_shot: int, seed: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k-shot support sample from a collate_cached-shaped train batch."""
    rng = np.random.default_rng(seed)
    n = train_batch["feature"].shape[0]
    k = min(k_shot, n)
    idx = rng.choice(n, size=k, replace=False)
    sup_f = train_batch["feature"][idx].to(device)
    sup_kp = train_batch["keypoints_xy"][idx].to(device)
    sup_vis = train_batch["vis"][idx].to(device)
    return sup_f, sup_kp, sup_vis


def adapt_loop(
    model: ITIModel,
    tok: nn.Parameter,
    extra_params: list[nn.Parameter],
    sup_f: torch.Tensor, sup_kp: torch.Tensor, sup_vis: torch.Tensor,
    n_adapt_steps: int = 50, adapt_lr: float = 1.0, out_size: int = 224,
) -> None:
    """SGD(momentum=0.9) adaptation loop over the k-shot support set.
    Identical numerics to run_peft_baselines.py / run_h1_pareto_adapt.py."""
    params_to_opt = ([tok] + extra_params) if tok.requires_grad else extra_params
    if not params_to_opt:
        return
    opt = torch.optim.SGD(params_to_opt, lr=adapt_lr, momentum=0.9)
    model.train()
    B = sup_f.shape[0]
    for _ in range(n_adapt_steps):
        pred = model.forward_with_token(sup_f, tok.unsqueeze(0).expand(B, -1))
        target = sup_kp / out_size
        per_kp = ((pred - target) ** 2).sum(dim=-1)
        loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


def eval_on_batches(
    model: ITIModel, tok: torch.Tensor, val_batches: list[dict], device, out_size: int = 224,
) -> dict:
    """Evaluate (model, tok) on a list of collate_cached-shaped batches.
    Returns pck_metrics() plus a bootstrap CI on rmse_norm (both reused
    verbatim from iti.eval.pck)."""
    dists, diags = [], []
    with torch.no_grad():
        for batch in val_batches:
            fv = batch["feature"].to(device)
            Bv = fv.shape[0]
            pred_unit = model.forward_with_token(
                fv, tok.detach().unsqueeze(0).expand(Bv, -1)
            ).cpu()
            pred_xy = pred_unit * out_size
            d = per_keypoint_distance_normalized(
                pred_xy.numpy(), batch["keypoints_xy"].numpy(),
                batch["vis"].numpy(), batch["crop_side"].numpy(), out_size=out_size,
            )
            dists.append(d)
            diags.append(np.asarray(batch["bbox_diag_orig"]))
    d = np.concatenate(dists)
    diag = np.concatenate(diags)
    m = pck_metrics(d, diag)
    flat = (d / diag[:, None]).flatten()
    flat = flat[~np.isnan(flat)]
    if flat.size:
        sq = flat ** 2
        mn, lo, hi = bootstrap_ci(sq)
        m["rmse_norm_ci95_lo"] = float(np.sqrt(lo))
        m["rmse_norm_ci95_hi"] = float(np.sqrt(hi))
    return m


# --------------------------------------------------------------------------- #
# P15: per-budget Pareto grid — extends run_h1_pareto_adapt.py's
# `select_modules` with the missing ~1.2M-param point (both cross-attn
# blocks combined) and the ~8.5M full-decoder point, so all FOUR budgets
# {768, 296K, 1.2M, 8.5M} are covered.
# --------------------------------------------------------------------------- #

PARETO_VARIANTS = ["token_only", "token_plus_id_proj", "token_plus_xattn_all", "full_decoder"]
PARETO_NOMINAL_BUDGET = {
    "token_only": 768,
    "token_plus_id_proj": 296_000,
    "token_plus_xattn_all": 1_200_000,
    "full_decoder": 8_500_000,
}


def select_pareto_modules(model: ITIModel, variant: str) -> list[nn.Parameter]:
    """The id_token is ALWAYS trained (matches run_h1_pareto_adapt.py); this
    returns the additional params for each budget point."""
    if variant == "token_only":
        return []
    if variant == "token_plus_id_proj":
        return list(model.decoder.id_proj.parameters())
    if variant == "token_plus_xattn_all":
        # NEW: both cross_attn_blocks (vs. run_h1_pareto_adapt.py's
        # token_plus_xattn0, which only used block[0], ~590K). Both blocks
        # combined land at ~1.2M trainable params — the budget point the
        # reviewers flagged as unreported.
        return list(model.decoder.cross_attn_blocks.parameters())
    if variant == "full_decoder":
        return list(model.decoder.parameters())
    raise ValueError(f"unknown Pareto variant: {variant!r}")


def build_pareto_model(state: dict, cfg: ITIConfig, device, variant: str) -> tuple[ITIModel, nn.Parameter, list[nn.Parameter], int]:
    # backbone=nn.Identity(): every code path below is cached-feature-only
    # (forward_from_features / forward_with_token), which never touches
    # model.backbone. Skipping the real FrozenDINOv2Base construction here
    # avoids an unnecessary (and, off the dev machine, possibly network-
    # dependent) backbone load on every single adaptation unit.
    model = ITIModel(cfg, backbone=nn.Identity()).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        id_bank = model.tokens.emb.weight.detach()
    extra_params = select_pareto_modules(model, variant)
    for p in extra_params:
        p.requires_grad_(True)
    # token placeholder (mean init); caller overwrites via interp head.
    mean_tok = id_bank.mean(dim=0) if id_bank.shape[0] > 0 else torch.zeros(cfg.hidden, device=device)
    tok = nn.Parameter(mean_tok.clone())
    n_trainable = tok.numel() + sum(p.numel() for p in extra_params)
    return model, tok, extra_params, n_trainable


# --------------------------------------------------------------------------- #
# Statistics helpers (new, small, self-contained — modelled on the style of
# iti.eval.pck.bootstrap_ci, which they call out to where possible).
# --------------------------------------------------------------------------- #

def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on mean(a - b), a and b PAIRED element-wise (e.g.
    the same seeds for two methods, so per-seed noise cancels). Returns
    (mean_diff, lo95, hi95)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, (a.shape, b.shape)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = a - b
    if n == 1:
        return float(diffs[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return float(diffs.mean()), lo, hi


def _r2_aic(y: np.ndarray, yhat: np.ndarray, n_params: int) -> tuple[float, float]:
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = len(y)
    if n <= n_params or ss_res <= 0:
        aic = float("nan")
    else:
        aic = n * np.log(ss_res / n) + 2 * n_params
    return r2, float(aic)


def quadratic_vs_linear_fit(
    x: np.ndarray, y: np.ndarray, n_resamples: int = 1000, seed: int = 0,
) -> dict:
    """Fit y ~ a*x + b (linear, 2 params) and y ~ a*x^2 + b*x + c (quadratic,
    3 params) via least squares; report R2/AIC for both plus a bootstrap CI
    (resampling SPECIES with replacement, refitting each draw) on the
    quadratic coefficient `a`. Used by P30 to test the inverted-U hypothesis
    (adaptation gain peaks at intermediate distance-from-training-pool)
    against a plain linear trend, on a proper multi-point grid (8+ species)
    instead of the reviewer-flagged 3-point fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    out = {"n_points": n}
    if n < 4:
        out["note"] = "insufficient points for a stable quadratic fit (need >= 4)"
        return out

    lin_coef = np.polyfit(x, y, 1)
    lin_yhat = np.polyval(lin_coef, x)
    r2_lin, aic_lin = _r2_aic(y, lin_yhat, n_params=2)

    quad_coef = np.polyfit(x, y, 2)
    quad_yhat = np.polyval(quad_coef, x)
    r2_quad, aic_quad = _r2_aic(y, quad_yhat, n_params=3)

    rng = np.random.default_rng(seed)
    a_draws = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        try:
            cb = np.polyfit(xb, yb, 2)
            a_draws[i] = cb[0]
        except np.linalg.LinAlgError:
            a_draws[i] = np.nan
    a_draws = a_draws[~np.isnan(a_draws)]
    if a_draws.size:
        lo = float(np.quantile(a_draws, 0.025))
        hi = float(np.quantile(a_draws, 0.975))
    else:
        lo = hi = float("nan")

    out.update({
        "linear_coef": lin_coef.tolist(), "r2_linear": r2_lin, "aic_linear": aic_lin,
        "quadratic_coef": quad_coef.tolist(), "r2_quadratic": r2_quad, "aic_quadratic": aic_quad,
        "quadratic_a_coef": float(quad_coef[0]),
        "quadratic_a_ci95_lo": lo, "quadratic_a_ci95_hi": hi,
        "prefers_quadratic": bool(aic_quad < aic_lin) if np.isfinite(aic_quad) and np.isfinite(aic_lin) else None,
        "inverted_u": bool(quad_coef[0] < 0),  # negative leading coef -> concave (inverted U)
    })
    return out


# =============================================================================
# Tier B: backbone-forward PEFT (LoRA-on-backbone, AdaptFormer, IA3, VPT)
#
# STRUCTURAL guarantee against the "silently wrong" failure mode called out
# in the task: every function below takes a `BackboneHandle` and a raw PIXEL
# batch (never a cached-feature batch) and always forwards with grad=True.
# There is no code path here that accepts `iti.data.feature_cache`-shaped
# batches, so Tier-B methods structurally cannot be fed cached features.
# =============================================================================

TIER_B_METHODS = ["lora_backbone", "ia3", "adaptformer", "vpt"]


def _get_embeddings_module(handle: BackboneHandle) -> nn.Module:
    raw = handle.raw_model
    if hasattr(raw, "embeddings"):
        return raw.embeddings
    if hasattr(raw, "vision_model") and hasattr(raw.vision_model, "embeddings"):
        return raw.vision_model.embeddings
    raise BackboneUnavailable(
        f"cannot locate an embeddings module for VPT on backbone kind {handle.spec.kind!r}"
    )


def _get_transformer_layers(handle: BackboneHandle) -> nn.ModuleList:
    raw = handle.raw_model
    if hasattr(raw, "encoder") and hasattr(raw.encoder, "layer"):
        return raw.encoder.layer            # ViT / DINOv2 / ViTMAE style
    if hasattr(raw, "encoder") and hasattr(raw.encoder, "layers"):
        return raw.encoder.layers           # some HF encoders name it "layers"
    if hasattr(raw, "vision_model") and hasattr(raw.vision_model.encoder, "layers"):
        return raw.vision_model.encoder.layers   # CLIPVisionModel
    if hasattr(raw, "blocks"):
        return raw.blocks                    # timm ViT-style
    raise BackboneUnavailable(
        f"cannot locate transformer layers for AdaptFormer on backbone kind {handle.spec.kind!r}"
    )


class VPTPromptInjector(nn.Module):
    """VPT-shallow: p learnable prompt tokens prepended to the ViT input,
    right after CLS and before the patch tokens. Attaches via a forward hook
    on the embeddings module's OUTPUT so it works across HF model families
    (ViT/DINOv2/CLIP/MAE all call `embeddings(pixel_values) -> sequence`
    before the encoder) without touching the encoder's forward at all.

    ViTMAEModel's embeddings return a (embeddings, mask, ids_restore) tuple
    (masking machinery); handled generically below.
    """

    def __init__(self, n_prompts: int, hidden: int) -> None:
        super().__init__()
        self.n_prompts = n_prompts
        self.prompts = nn.Parameter(torch.zeros(1, n_prompts, hidden))
        nn.init.normal_(self.prompts, std=0.02)
        self._handle = None

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple):
            emb, rest = output[0], output[1:]
        else:
            emb, rest = output, None
        B = emb.shape[0]
        prompts = self.prompts.expand(B, -1, -1).to(dtype=emb.dtype, device=emb.device)
        emb2 = torch.cat([emb[:, :1], prompts, emb[:, 1:]], dim=1)  # after CLS, before patches
        return (emb2,) + rest if rest is not None else emb2

    def attach(self, embeddings_module: nn.Module) -> None:
        self._handle = embeddings_module.register_forward_hook(self._hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class AdaptFormerBlockWrapper(nn.Module):
    """Parallel bottleneck adapter attached across a transformer block:
    ``final = block(x) + scale * up(act(down(x)))``.

    DOCUMENTED SIMPLIFICATION: canonical AdaptFormer (Chen et al. 2022)
    inserts the parallel branch specifically alongside the block's MLP
    sub-layer (parallel to `output.dense`, not the whole attention+MLP
    block). Because block internals differ across model families (ViT vs
    DINOv2 vs CLIP vs timm), per-family MLP surgery would multiply the
    implementation surface for the lowest-priority job in this suite. We
    instead attach the parallel branch across the WHOLE block's
    input->output residual (a block-level parallel adapter), which is
    architecture-agnostic via a single wrapper class. This is CLEARLY
    LABELED as `adaptformer` (block-level parallel bottleneck) in every
    result row / metadata field — never conflated with the strict
    per-MLP variant.
    """

    def __init__(self, block: nn.Module, hidden: int, bottleneck: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.block = block
        self.down = nn.Linear(hidden, bottleneck)
        self.up = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)  # init to a no-op so the wrapped block starts identical to unwrapped
        self.act = nn.GELU()
        self.scale = scale

    def forward(self, hidden_states, *args, **kwargs):
        block_out = self.block(hidden_states, *args, **kwargs)
        main, rest = (block_out[0], block_out[1:]) if isinstance(block_out, tuple) else (block_out, None)
        adapter_out = self.up(self.act(self.down(hidden_states))) * self.scale
        combined = main + adapter_out
        return (combined,) + rest if rest is not None else combined


class TierBHandle:
    """Bundles a configured Tier-B method's trainable params + a
    patch_features() closure + a teardown() for symmetry/clarity. Every
    Tier-B unit builds ONE of these per (backbone, species, method, seed) —
    backbones are reloaded fresh each unit (self-contained-unit contract),
    so teardown is mostly a no-op but is provided for explicitness."""

    def __init__(self, handle: BackboneHandle, trainable_params: list[nn.Parameter],
                 label: str, teardown=None) -> None:
        self.handle = handle
        self.trainable_params = trainable_params
        self.label = label
        self.n_trainable = sum(p.numel() for p in trainable_params)
        self._teardown = teardown

    def patch_features(self, pixel_values: torch.Tensor, grad: bool = True) -> torch.Tensor:
        # grad=True is the correct default for Tier-B training steps; eval
        # callers should still pass grad=False explicitly for clarity.
        return self.handle.patch_features(pixel_values, grad=grad)

    def teardown(self) -> None:
        if self._teardown is not None:
            self._teardown()


def build_tier_b_lora(handle: BackboneHandle, r: int = 8, alpha: int = 16) -> TierBHandle:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise DependencyUnavailable(f"peft not installed; lora_backbone unavailable: {e}") from e
    targets = PEFT_TARGETS[handle.spec.kind]["lora"]
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets, bias="none")
    try:
        peft_model = get_peft_model(handle.raw_model, cfg)
    except Exception as e:
        raise DependencyUnavailable(f"peft LoRA wrapping failed for {handle.spec.name}: {e}") from e
    handle.raw_model = peft_model  # route future forwards through the wrapped model
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    return TierBHandle(handle, trainable, label="lora_backbone")


def build_tier_b_ia3(handle: BackboneHandle) -> TierBHandle:
    try:
        from peft import IA3Config, get_peft_model
    except ImportError as e:
        raise DependencyUnavailable(f"peft not installed; ia3 unavailable: {e}") from e
    targets = PEFT_TARGETS[handle.spec.kind]["ia3"]
    ff = PEFT_TARGETS[handle.spec.kind]["ia3_ff"]
    try:
        cfg = IA3Config(target_modules=targets, feedforward_modules=ff)
        peft_model = get_peft_model(handle.raw_model, cfg)
    except Exception as e:
        raise DependencyUnavailable(f"peft IA3 wrapping failed for {handle.spec.name}: {e}") from e
    handle.raw_model = peft_model
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    return TierBHandle(handle, trainable, label="ia3")


def build_tier_b_adaptformer(handle: BackboneHandle, bottleneck: int = 64, scale: float = 1.0) -> TierBHandle:
    layers = _get_transformer_layers(handle)
    trainable: list[nn.Parameter] = []
    for i in range(len(layers)):
        wrapped = AdaptFormerBlockWrapper(layers[i], hidden=handle.hidden, bottleneck=bottleneck, scale=scale)
        layers[i] = wrapped
        trainable += list(wrapped.down.parameters()) + list(wrapped.up.parameters())
    return TierBHandle(handle, trainable, label="adaptformer")


def build_tier_b_vpt(handle: BackboneHandle, n_prompts: int = 10) -> TierBHandle:
    if handle.spec.kind == "timm":
        raise BackboneUnavailable("VPT hook-based injection is not implemented for timm backbones")
    emb_module = _get_embeddings_module(handle)
    injector = VPTPromptInjector(n_prompts=n_prompts, hidden=handle.hidden)
    injector.to(next(handle.raw_model.parameters()).device)
    injector.attach(emb_module)
    handle.extra_prefix_tokens = n_prompts   # patch_features() must now also drop the prompts

    def _teardown():
        injector.detach()
        handle.extra_prefix_tokens = 0

    return TierBHandle(handle, list(injector.parameters()), label="vpt", teardown=_teardown)


TIER_B_BUILDERS = {
    "lora_backbone": build_tier_b_lora,
    "ia3": build_tier_b_ia3,
    "adaptformer": build_tier_b_adaptformer,
    "vpt": build_tier_b_vpt,
}


def build_tier_b(handle: BackboneHandle, method: str) -> TierBHandle:
    if method not in TIER_B_BUILDERS:
        raise ValueError(f"unknown Tier-B method: {method!r}")
    return TIER_B_BUILDERS[method](handle)


def tier_b_adapt_and_eval(
    tb: TierBHandle,
    decoder_model: ITIModel,
    tok: torch.Tensor,
    sup_pixels: torch.Tensor, sup_kp: torch.Tensor, sup_vis: torch.Tensor,
    val_pixel_batches: list[dict],
    n_adapt_steps: int = 20, adapt_lr: float = 1e-3, out_size: int = 224,
) -> dict:
    """Tier-B adapt+eval loop: every step (train AND eval) re-runs the full
    backbone forward on raw pixels (grad=True while training, grad=False
    while evaluating) — the defining property that distinguishes Tier B from
    Tier A. The decoder + id-token come from an ITIModel built for the
    backbone's (hidden, grid); only `tb.trainable_params` (the inserted PEFT
    params) receive gradient updates here — the decoder/token stay fixed at
    whatever state `decoder_model`/`tok` were passed in (matching the task
    spec: Tier-B methods insert trainable modules INSIDE the ViT; the
    backbone weights themselves stay frozen).
    """
    if not tb.trainable_params:
        raise ValueError(f"Tier-B method {tb.label!r} produced zero trainable params")
    opt = torch.optim.AdamW(tb.trainable_params, lr=adapt_lr)
    tb.handle.train()
    for _ in range(n_adapt_steps):
        feats = tb.patch_features(sup_pixels, grad=True)
        B = feats.shape[0]
        pred = decoder_model.forward_with_token(feats, tok.unsqueeze(0).expand(B, -1).detach())
        target = sup_kp / out_size
        per_kp = ((pred - target) ** 2).sum(dim=-1)
        loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    tb.handle.eval()

    dists, diags = [], []
    with torch.no_grad():
        for batch in val_pixel_batches:
            fv = tb.patch_features(batch["pixel_values"], grad=False)
            Bv = fv.shape[0]
            pred_unit = decoder_model.forward_with_token(fv, tok.unsqueeze(0).expand(Bv, -1)).cpu()
            pred_xy = pred_unit * out_size
            d = per_keypoint_distance_normalized(
                pred_xy.numpy(), batch["keypoints_xy"].numpy(),
                batch["vis"].numpy(), np.asarray(batch["crop_side"]), out_size=out_size,
            )
            dists.append(d)
            diags.append(np.asarray(batch["bbox_diag_orig"]))
    d = np.concatenate(dists)
    diag = np.concatenate(diags)
    m = pck_metrics(d, diag)
    tb.teardown()
    return m
