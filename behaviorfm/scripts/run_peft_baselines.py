"""Head-to-head PEFT baseline panel for H1 8-species held-out animal-pose grid.

Compare Identity-Token Adaptation
(ITA, the paper's method) against standard PEFT baselines (LoRA/BitFit/VPT)
at matched adaptation budget (5-shot, 50 SGD steps, same harness as
run_h1_pareto_adapt.py so the comparison is apples-to-apples).

Methods
-------
ita_token_only  : trainable = novel id-token only (768 params)      [paper anchor]
bitfit          : trainable = all bias tensors in model.decoder      [~14k params]
lora_r4         : trainable = LoRA A/B on every Linear in decoder   [~75k params]
vpt_deep        : SKIPPED — see note below.

VPT note: prepending learnable prompt tokens to the cross-attention query
sequence in PoseDecoder.forward would require in-place surgery on a
forward method that is the canonical eval path for the whole paper.  Any
change there risks drifting from the paper's forward pass and breaking the
apples-to-apples comparison.  VPT is therefore omitted; the script reports
this explicitly in the summary.

Outputs
-------
results/peft_baselines.csv          (one row per species × method × seed)
results/peft_baselines_summary.md   (mean±sem per method across 8 species)

Usage
-----
# Smoke test (fast):
python scripts/run_peft_baselines.py --species fox --n-seeds 1

# Full 8-species grid (leave to main thread):
python scripts/run_peft_baselines.py --n-seeds 10
"""

from __future__ import annotations
import argparse
import logging
import sys
import warnings
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset, collate_cached
from iti.eval.pck import per_keypoint_distance_normalized, pck_metrics
from iti.model.iti import ITIModel, ITIConfig
from iti.baselines.peft import LoRALinear, wrap_decoder_with_lora

warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("peft_baselines")

# ---------------------------------------------------------------------------
# Species → train-cache mapping
# ---------------------------------------------------------------------------

SPECIES_8 = [
    "alouatta",
    "chimpanzee",
    "elephant",
    "monkey",
    "gorilla",
    "rabbit",
    "fox",
    "panther",
]

SPECIES_TO_TRAIN_CACHE = {
    "fox":        "dinov2_base_ap10k_train_split1_distance_sweep",
    "monkey":     "dinov2_base_ap10k_train_split1_distance_sweep",
    "rabbit":     "dinov2_base_ap10k_train_split1_distance_sweep",
    "panther":    "dinov2_base_ap10k_train_split1_distance_sweep",
    "alouatta":   "dinov2_base_ap10k_train_split1_alouatta",
    "elephant":   "dinov2_base_ap10k_train_split1_elephant",
    "gorilla":    "dinov2_base_ap10k_train_split1_gorilla",
    "chimpanzee": "dinov2_base_ap10k_train_split1_chimpanzee",
}

# ---------------------------------------------------------------------------
# Adaptation-method setup
# ---------------------------------------------------------------------------

METHODS = ["ita_token_only", "bitfit", "lora_r4"]


def build_adapted_model(
    state: dict,
    cfg: ITIConfig,
    device: str,
    method: str,
) -> tuple[ITIModel, nn.Parameter, list[nn.Parameter], int]:
    """Load a fresh model and configure trainable parameters for *method*.

    Returns
    -------
    model        : freshly loaded ITIModel (backbone + decoder)
    tok          : the novel id-token Parameter (always returned for use in
                   forward_with_token; whether it is trainable depends on method)
    extra_params : additional trainable parameter list (empty for ita_token_only)
    n_trainable  : total trainable parameter count
    """
    model = ITIModel(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # Freeze everything.
    for p in model.parameters():
        p.requires_grad_(False)

    # Compute mean-of-trained-tokens initialisation (used by bitfit + lora_r4).
    with torch.no_grad():
        mean_tok = model.tokens.emb.weight.detach().mean(dim=0)

    if method == "ita_token_only":
        # Paper anchor: only the novel id-token is trained.
        # Token is initialised via the interpolation head (same as run_h1_pareto_adapt.py).
        # NOTE: we defer interp-head init to the caller because it needs the
        # support features; here we just return mean_tok as a placeholder that
        # the caller will overwrite.
        tok = nn.Parameter(mean_tok.clone())
        extra_params: list[nn.Parameter] = []

    elif method == "bitfit":
        # Train all bias tensors in the decoder; id-token frozen at mean init.
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)   # id-token frozen for BitFit
        extra_params = [
            p for name, p in model.decoder.named_parameters()
            if name.endswith(".bias")
        ]
        for p in extra_params:
            p.requires_grad_(True)

    elif method == "lora_r4":
        # Wrap every Linear in decoder with LoRA; id-token frozen at mean init.
        tok = nn.Parameter(mean_tok.clone())
        tok.requires_grad_(False)   # id-token frozen for LoRA
        n_added = wrap_decoder_with_lora(model.decoder, r=4, alpha=8.0)
        model.to(device)  # move newly-created LoRA params/buffers onto the device
        logger.debug("lora_r4: wrapped decoder, added %d LoRA params", n_added)
        extra_params = [
            p for p in model.decoder.parameters()
            if p.requires_grad
        ]
        # LoRA buffers (frozen base weights) should NOT have requires_grad.
        # The wrap function only makes lora_A and lora_B trainable.

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # Count total trainable params (tok may or may not be trainable).
    trainable = [tok] + extra_params if tok.requires_grad else extra_params
    n_trainable = sum(p.numel() for p in trainable)

    return model, tok, extra_params, n_trainable


def init_token_via_interp_head(
    model: ITIModel,
    sup_f: torch.Tensor,   # (k, N, hidden)
    device: str,
) -> torch.Tensor:
    """Initialise a novel id-token via the interpolation head (paper's strategy)."""
    id_bank = model.tokens.emb.weight.detach()
    with torch.no_grad():
        ex_feat = sup_f.mean(dim=0).unsqueeze(0)   # (1, N, hidden) mean over k
        init_tok, _ = model.interpolation_head(ex_feat, id_bank)
        init_tok = init_tok.squeeze(0)              # (hidden,)
    return init_tok


# ---------------------------------------------------------------------------
# Per-species adaptation + evaluation
# ---------------------------------------------------------------------------

def run_species(
    species: str,
    args: argparse.Namespace,
    state: dict,
    cfg: ITIConfig,
    val_full: CachedFeatureDataset,
    name_to_gid_by_cache: dict[str, dict[str, int]],
) -> list[dict]:
    """Run all methods × seeds for one species. Returns list of result rows."""
    train_cache_name = SPECIES_TO_TRAIN_CACHE[species]
    train_cache_path = ROOT / "data/cache" / train_cache_name

    # Look up this species' gid in its train cache.
    cache_records = CachedFeatureDataset(train_cache_path).records
    name_to_gid = {r["species_name"]: r["identity_id"] for r in cache_records}
    gid = name_to_gid[species]

    train_held = CachedFeatureDataset(train_cache_path, identity_subset={gid})
    val_held = CachedFeatureDataset(args.val_cache, identity_subset={gid})

    if len(val_held) == 0:
        logger.warning("No val examples found for %s (gid=%d) — skipping", species, gid)
        return []

    val_batches = list(DataLoader(
        val_held, batch_size=32, shuffle=False, collate_fn=collate_cached
    ))
    train_all = list(DataLoader(
        train_held, batch_size=len(train_held), shuffle=False, collate_fn=collate_cached
    ))
    train_batch = train_all[0]
    train_features = train_batch["feature"].to(args.device)

    rows = []

    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        # Sample k-shot support indices (same logic as run_h1_pareto_adapt.py).
        idx = rng.choice(len(train_held), size=args.k_shot, replace=False)
        sup_f   = train_features[idx]                                # (k, N, hidden)
        sup_kp  = train_batch["keypoints_xy"][idx].to(args.device)  # (k, K, 2)
        sup_vis = train_batch["vis"][idx].to(args.device)            # (k, K)

        for method in args.methods:
            model, tok, extra_params, n_trainable = build_adapted_model(
                state, cfg, args.device, method
            )

            # For ita_token_only, override tok with interp-head init (paper's strategy).
            if method == "ita_token_only":
                init_val = init_token_via_interp_head(model, sup_f, args.device)
                with torch.no_grad():
                    tok.copy_(init_val)
                tok.requires_grad_(True)
                n_trainable = tok.numel()

            # Build optimizer over the union of trainable parameters.
            params_to_opt = (
                ([tok] + extra_params) if tok.requires_grad else extra_params
            )
            opt = torch.optim.SGD(params_to_opt, lr=args.adapt_lr, momentum=0.9)

            # ----------------------------------------------------------------
            # Adapt loop (identical to run_h1_pareto_adapt.py).
            # ----------------------------------------------------------------
            model.train()  # enable potential dropout (none in this decoder, but correct)
            for _step in range(args.n_adapt_steps):
                B = sup_f.shape[0]
                pred = model.forward_with_token(
                    sup_f, tok.unsqueeze(0).expand(B, -1)
                )
                target = sup_kp / 224.0
                per_kp = ((pred - target) ** 2).sum(dim=-1)
                loss = (per_kp * sup_vis).sum() / sup_vis.sum().clamp_min(1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()

            # ----------------------------------------------------------------
            # Evaluate on full val split for this species.
            # ----------------------------------------------------------------
            model.eval()
            dists, diags = [], []
            with torch.no_grad():
                for batch in val_batches:
                    fv = batch["feature"].to(args.device)
                    Bv = fv.shape[0]
                    pred_unit = model.forward_with_token(
                        fv, tok.detach().unsqueeze(0).expand(Bv, -1)
                    ).cpu()
                    pred_xy = pred_unit * 224.0
                    d = per_keypoint_distance_normalized(
                        pred_xy.numpy(),
                        batch["keypoints_xy"].numpy(),
                        batch["vis"].numpy(),
                        batch["crop_side"].numpy(),
                        out_size=224,
                    )
                    dists.append(d)
                    diags.append(batch["bbox_diag_orig"].numpy())

            d = np.concatenate(dists)
            diag = np.concatenate(diags)
            m = pck_metrics(d, diag)

            row = {
                "species":          species,
                "method":           method,
                "n_trainable_params": n_trainable,
                "seed":             seed,
                "rmse_norm":        m["rmse_norm"],
                "pck@0.05":         m["pck@0.05"],
                "pck@0.10":         m["pck@0.10"],
                "pck@0.20":         m["pck@0.20"],
                "n_kp_eval":        m["n_keypoints_evaluated"],
            }
            rows.append(row)
            logger.info(
                "[%s  seed=%d] %-16s  n_params=%-7d  rmse=%.4f  pck@0.05=%.3f",
                species, seed, method, n_trainable,
                m["rmse_norm"], m["pck@0.05"],
            )

    return rows


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------

def write_summary(rows: list[dict], out_path: Path) -> None:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    lines = [
        "# PEFT Baselines Summary — H1 8-species held-out grid",
        "",
        "Columns: mean ± SEM across seeds and species.  "
        "Adaptation: 5-shot, 50 SGD steps, lr=1.0, momentum=0.9.",
        "",
        "| method | n_trainable | RMSE_norm (↓) | PCK@0.05 (↑) | PCK@0.10 (↑) | PCK@0.20 (↑) |",
        "|--------|------------|---------------|--------------|--------------|--------------|",
    ]
    for method in methods:
        sub = [r for r in rows if r["method"] == method]
        if not sub:
            continue
        n_p = sub[0]["n_trainable_params"]
        rmses  = np.asarray([r["rmse_norm"] for r in sub])
        p05    = np.asarray([r["pck@0.05"]  for r in sub])
        p10    = np.asarray([r["pck@0.10"]  for r in sub])
        p20    = np.asarray([r["pck@0.20"]  for r in sub])
        n = len(sub)
        sem = lambda a: a.std() / np.sqrt(n) if n > 1 else 0.0
        lines.append(
            f"| {method} | {n_p:,} "
            f"| {rmses.mean():.4f} ± {sem(rmses):.4f} "
            f"| {p05.mean():.3f} ± {sem(p05):.3f} "
            f"| {p10.mean():.3f} ± {sem(p10):.3f} "
            f"| {p20.mean():.3f} ± {sem(p20):.3f} |"
        )
    lines += [
        "",
        "**VPT (vpt_deep): SKIPPED.**  "
        "Prepending learnable prompt tokens to the PoseDecoder cross-attention "
        "query sequence would require modifying `PoseDecoder.forward`, which is "
        "the canonical eval path shared with the paper.  To preserve the "
        "apples-to-apples comparison, VPT is omitted from this panel.",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Summary written to %s", out_path)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_COLS = [
    "species", "method", "n_trainable_params", "seed",
    "rmse_norm", "pck@0.05", "pck@0.10", "pck@0.20",
]


def append_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a") as f:
        if write_header:
            f.write(",".join(CSV_COLS) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in CSV_COLS) + "\n")
    logger.info("Appended %d rows to %s", len(rows), csv_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="PEFT baseline panel for H1 8-species held-out grid."
    )
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "results/q2_within_species/iti_xattn_aux01_seed0/model.pt")
    p.add_argument("--val-cache", type=Path,
                   default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all")
    p.add_argument("--out-csv", type=Path,
                   default=ROOT / "results/peft_baselines.csv")
    p.add_argument("--out-summary", type=Path,
                   default=ROOT / "results/peft_baselines_summary.md")
    p.add_argument("--device", type=str, default="mps",
                   help="Compute device: 'mps' (default) or 'cpu'.")
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-adapt-steps", type=int, default=50)
    p.add_argument("--adapt-lr", type=float, default=1.0)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--species", nargs="+", default=SPECIES_8,
                   choices=SPECIES_8,
                   help="Subset of species to run (default: all 8).")
    p.add_argument("--methods", nargs="+", default=METHODS,
                   choices=METHODS,
                   help="Subset of methods to run (default: all).")
    args = p.parse_args()

    logger.info("Device: %s | species: %s | methods: %s | n_seeds: %d",
                args.device, args.species, args.methods, args.n_seeds)

    # Load checkpoint once; share across species.
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ITIConfig(**{k: v for k, v in state["cfg"].items()
                       if k in ITIConfig.__dataclass_fields__})
    logger.info("Loaded checkpoint from %s  (cfg: %s)", args.ckpt, cfg)

    val_full = CachedFeatureDataset(args.val_cache)

    all_rows: list[dict] = []

    for species in args.species:
        logger.info("=== Species: %s ===", species)
        rows = run_species(
            species=species,
            args=args,
            state=state,
            cfg=cfg,
            val_full=val_full,
            name_to_gid_by_cache={},
        )
        all_rows.extend(rows)
        append_csv(rows, args.out_csv)

    write_summary(all_rows, args.out_summary)
    logger.info("Done.  Total rows: %d", len(all_rows))


if __name__ == "__main__":
    main()
