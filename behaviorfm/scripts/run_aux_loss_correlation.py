"""Empirical aux-loss -> downstream-accuracy correlation analysis.

For each of the two checkpoints (with-aux, without-aux), and each held-out
species, we:
  1. Draw K random identity tokens ~ N(0, 0.02^2) of dim 768 (seeded).
  2. Run model.forward_with_token for each random token over the species' val
     images -> predicted keypoint coords (B, 17, 2).
  3. Compute the per-coordinate prediction std ACROSS the K random tokens, then
     average over images/keypoints/coords -> a scalar "token-utility std" per
     species.

We then correlate the with-aux per-species token-utility std against the
per-species RMSE-reduction from the existing h1_n10_<species>.csv results.

Outputs
-------
results/aux_loss_correlation.json
results/aux_loss_correlation.md

Usage
-----
# Smoke test (fast):
python scripts/run_aux_loss_correlation.py --species fox --n-tokens 5

# Full 8-species run:
python scripts/run_aux_loss_correlation.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from iti.data.feature_cache import CachedFeatureDataset, collate_cached
from iti.model.iti import ITIModel, ITIConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("aux_loss_correlation")

# ---------------------------------------------------------------------------
# Constants
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

CKPT_WITH_AUX = ROOT / "results/q2_within_species/iti_xattn_aux01_seed0/model.pt"
CKPT_WITHOUT_AUX = ROOT / "results/q2_within_species/iti_xattn_seed0/model.pt"

TOKEN_NOISE_STD = 0.02
TOKEN_DIM = 768
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: str) -> tuple[ITIModel, dict]:
    """Load ITIModel from checkpoint. Returns (model, state_dict)."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ITIConfig(
        **{k: v for k, v in state["cfg"].items()
           if k in ITIConfig.__dataclass_fields__}
    )
    model = ITIModel(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("Loaded %s", ckpt_path)
    return model, state


# ---------------------------------------------------------------------------
# Token-utility std computation
# ---------------------------------------------------------------------------

def compute_token_utility_std(
    model: ITIModel,
    val_batches: list[dict],
    n_tokens: int,
    device: str,
    rng: np.random.Generator,
) -> float:
    """Compute the mean token-utility std for one species.

    For each of n_tokens random identity tokens, run model.forward_with_token
    over all val images -> predicted coords (B, 17, 2).

    Collect all predictions: shape (n_tokens, N_images_total, 17, 2).
    Compute std over axis-0 (the token dimension) -> (N_images_total, 17, 2).
    Return the scalar mean across images/keypoints/coords.
    """
    # Draw K random tokens ~ N(0, sigma^2)
    raw_tokens = rng.standard_normal((n_tokens, TOKEN_DIM)).astype(np.float32)
    raw_tokens *= TOKEN_NOISE_STD
    tokens_t = torch.from_numpy(raw_tokens).to(device)  # (K, 768)

    all_preds: list[np.ndarray] = []  # per-token: (N_val, 17, 2)

    with torch.no_grad():
        for k_idx in range(n_tokens):
            tok = tokens_t[k_idx]  # (768,)
            preds_for_token: list[np.ndarray] = []

            for batch in val_batches:
                features = batch["feature"].to(device)  # (B, grid^2, hidden)
                B = features.shape[0]
                tok_expanded = tok.unsqueeze(0).expand(B, -1)  # (B, 768)
                pred = model.forward_with_token(features, tok_expanded)  # (B, 17, 2)
                preds_for_token.append(pred.cpu().numpy())

            all_preds.append(np.concatenate(preds_for_token, axis=0))  # (N_val, 17, 2)

    # Stack: (K, N_val, 17, 2)
    stacked = np.stack(all_preds, axis=0)

    # Std over K tokens -> (N_val, 17, 2)
    std_per_img_kp_coord = stacked.std(axis=0)

    # Scalar mean over images, keypoints, coords
    token_utility_std = float(std_per_img_kp_coord.mean())
    return token_utility_std


# ---------------------------------------------------------------------------
# RMSE-reduction from h1_n10 CSV
# ---------------------------------------------------------------------------

def load_rmse_reduction(species: str) -> float | None:
    """Load per-species RMSE reduction from h1_n10_<species>.csv.

    Reduction = (base_rmse - adapt_rmse) / base_rmse
    where base_rmse  = mean(no_adapt_mean_token rmse_norm)
          adapt_rmse = mean(adapt_iti_full rmse_norm)

    Returns None if the CSV does not exist or the strategies are missing.
    """
    csv_path = ROOT / f"results/h1_n10_{species}.csv"
    if not csv_path.exists():
        logger.warning("Missing CSV: %s", csv_path)
        return None

    base_vals: list[float] = []
    adapt_vals: list[float] = []

    with csv_path.open() as f:
        header_line = f.readline().strip()
        cols = header_line.split(",")
        strategy_idx = cols.index("strategy")
        rmse_idx = cols.index("rmse_norm")

        for line in f:
            parts = line.strip().split(",")
            if len(parts) <= max(strategy_idx, rmse_idx):
                continue
            strategy = parts[strategy_idx]
            try:
                rmse = float(parts[rmse_idx])
            except ValueError:
                continue

            if strategy == "no_adapt_mean_token":
                base_vals.append(rmse)
            elif strategy == "adapt_iti_full":
                adapt_vals.append(rmse)

    if not base_vals or not adapt_vals:
        logger.warning("Missing strategies in %s", csv_path)
        return None

    base_rmse = float(np.mean(base_vals))
    adapt_rmse = float(np.mean(adapt_vals))
    reduction = (base_rmse - adapt_rmse) / base_rmse
    logger.info(
        "  %s: base_rmse=%.5f  adapt_rmse=%.5f  rmse_reduction=%.4f",
        species, base_rmse, adapt_rmse, reduction,
    )
    return reduction


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Correlate per-species token-utility std with RMSE reduction."
    )
    p.add_argument(
        "--species", nargs="+", default=SPECIES_8, choices=SPECIES_8,
        help="Species to evaluate (default: all 8).",
    )
    p.add_argument(
        "--ckpt-with-aux", type=Path, default=CKPT_WITH_AUX,
        help="Checkpoint trained WITH aux loss.",
    )
    p.add_argument(
        "--ckpt-without-aux", type=Path, default=CKPT_WITHOUT_AUX,
        help="Checkpoint trained WITHOUT aux loss.",
    )
    p.add_argument(
        "--val-cache", type=Path,
        default=ROOT / "data/cache/dinov2_base_ap10k_val_split1_all",
        help="Path to val feature cache.",
    )
    p.add_argument(
        "--n-tokens", type=int, default=50,
        help="Number of random identity tokens (default 50).",
    )
    p.add_argument(
        "--device", type=str, default="mps",
        help="Compute device: 'mps' (default) or 'cpu'.",
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="RNG seed for random token draws (default 42).",
    )
    p.add_argument(
        "--batch-size", type=int, default=32,
        help="Val DataLoader batch size (default 32).",
    )
    args = p.parse_args()

    logger.info(
        "Device=%s | species=%s | n_tokens=%d | seed=%d",
        args.device, args.species, args.n_tokens, args.seed,
    )

    # -----------------------------------------------------------------------
    # Load both checkpoints
    # -----------------------------------------------------------------------
    model_with_aux, _ = load_model(args.ckpt_with_aux, args.device)
    model_without_aux, _ = load_model(args.ckpt_without_aux, args.device)

    # -----------------------------------------------------------------------
    # Load val cache once; filter per species below
    # -----------------------------------------------------------------------
    val_full = CachedFeatureDataset(args.val_cache)
    name_to_gid = {r["species_name"]: r["identity_id"] for r in val_full.records}

    # -----------------------------------------------------------------------
    # Per-species loop
    # -----------------------------------------------------------------------
    results: list[dict] = []

    for species in args.species:
        logger.info("=== Species: %s ===", species)

        if species not in name_to_gid:
            logger.warning("Species %s not found in val cache — skipping", species)
            continue

        gid = name_to_gid[species]
        val_held = CachedFeatureDataset(args.val_cache, identity_subset={gid})

        if len(val_held) == 0:
            logger.warning("No val examples for %s — skipping", species)
            continue

        val_batches = list(DataLoader(
            val_held, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_cached,
        ))
        logger.info("  val images: %d", len(val_held))

        # Seeded RNG — same token draws for both checkpoints (fair comparison).
        rng = np.random.default_rng(args.seed)

        std_with_aux = compute_token_utility_std(
            model_with_aux, val_batches, args.n_tokens, args.device, rng,
        )
        logger.info("  token-utility std  WITH aux = %.6e", std_with_aux)

        # Reset rng to same seed so both checkpoints see identical tokens.
        rng = np.random.default_rng(args.seed)
        std_without_aux = compute_token_utility_std(
            model_without_aux, val_batches, args.n_tokens, args.device, rng,
        )
        logger.info("  token-utility std  WITHOUT aux = %.6e", std_without_aux)
        logger.info(
            "  ratio (with/without) = %.2fx",
            std_with_aux / max(std_without_aux, 1e-12),
        )

        rmse_reduction = load_rmse_reduction(species)

        results.append({
            "species":          species,
            "std_with_aux":     std_with_aux,
            "std_without_aux":  std_without_aux,
            "rmse_reduction":   rmse_reduction,
        })

    if not results:
        logger.error("No results computed — exiting.")
        return

    # -----------------------------------------------------------------------
    # Correlation: with-aux std vs RMSE reduction
    # -----------------------------------------------------------------------
    valid = [r for r in results if r["rmse_reduction"] is not None]
    corr_results: dict = {}

    if len(valid) >= 2:
        x = np.asarray([r["std_with_aux"] for r in valid])
        y = np.asarray([r["rmse_reduction"] for r in valid])

        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        corr_results = {
            "n_species":   len(valid),
            "pearson_r":   float(pearson_r),
            "pearson_p":   float(pearson_p),
            "spearman_r":  float(spearman_r),
            "spearman_p":  float(spearman_p),
        }
        logger.info(
            "Correlation (with-aux std vs RMSE reduction):  "
            "Pearson r=%.3f (p=%.4f)  Spearman r=%.3f (p=%.4f)",
            pearson_r, pearson_p, spearman_r, spearman_p,
        )
    else:
        logger.warning("Not enough valid species for correlation (%d < 2)", len(valid))
        corr_results = {"n_species": len(valid), "note": "insufficient data"}

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "aux_loss_correlation.json"
    payload = {
        "per_species": results,
        "correlation": corr_results,
        "config": {
            "n_tokens":    args.n_tokens,
            "seed":        args.seed,
            "token_noise_std": TOKEN_NOISE_STD,
            "ckpt_with_aux":    str(args.ckpt_with_aux),
            "ckpt_without_aux": str(args.ckpt_without_aux),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved JSON: %s", json_path)

    # -----------------------------------------------------------------------
    # Save Markdown table
    # -----------------------------------------------------------------------
    md_path = out_dir / "aux_loss_correlation.md"
    lines = [
        "# Aux-Loss -> Downstream Accuracy Correlation",
        "",
        "**Token-utility std**: mean std across K random identity tokens, averaged "
        "over images × keypoints × coords.  "
        "Measures how much predictions *respond* to the identity token.",
        "",
        "**RMSE reduction**: `(base_rmse - adapt_rmse) / base_rmse` from "
        "`h1_n10_<species>.csv`, where `base = no_adapt_mean_token` and "
        "`adapt = adapt_iti_full`.",
        "",
        f"Settings: K={args.n_tokens} random tokens, noise σ={TOKEN_NOISE_STD}, "
        f"seed={args.seed}.",
        "",
        "| species | std_with_aux | std_without_aux | ratio (w/wo) | rmse_reduction |",
        "|---------|-------------|----------------|-------------|----------------|",
    ]

    for r in results:
        ratio = r["std_with_aux"] / max(r["std_without_aux"], 1e-12)
        rmse_str = f"{r['rmse_reduction']:.4f}" if r["rmse_reduction"] is not None else "N/A"
        lines.append(
            f"| {r['species']} "
            f"| {r['std_with_aux']:.4e} "
            f"| {r['std_without_aux']:.4e} "
            f"| {ratio:.2f}x "
            f"| {rmse_str} |"
        )

    if corr_results.get("n_species", 0) >= 2:
        lines += [
            "",
            "## Correlation: with-aux token-utility std vs RMSE reduction",
            "",
            f"N = {corr_results['n_species']} species",
            "",
            f"- **Pearson r** = {corr_results['pearson_r']:.3f}  "
            f"(p = {corr_results['pearson_p']:.4f})",
            f"- **Spearman r** = {corr_results['spearman_r']:.3f}  "
            f"(p = {corr_results['spearman_p']:.4f})",
            "",
            "A positive r confirms the empirical aux-loss -> accuracy chain: "
            "species whose predictions respond more to the identity token "
            "(higher token-utility std) also benefit more from token adaptation.",
        ]

    lines.append("")
    md_path.write_text("\n".join(lines))
    logger.info("Saved Markdown: %s", md_path)

    # -----------------------------------------------------------------------
    # Terminal summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AUX-LOSS CORRELATION SUMMARY")
    print("=" * 70)
    print(f"{'species':<14} {'std_with_aux':>14} {'std_without_aux':>16} {'ratio':>8} {'rmse_red':>10}")
    print("-" * 70)
    for r in results:
        ratio = r["std_with_aux"] / max(r["std_without_aux"], 1e-12)
        rmse_str = f"{r['rmse_reduction']:.4f}" if r["rmse_reduction"] is not None else "N/A"
        print(
            f"{r['species']:<14} "
            f"{r['std_with_aux']:>14.4e} "
            f"{r['std_without_aux']:>16.4e} "
            f"{ratio:>7.2f}x "
            f"{rmse_str:>10}"
        )
    if corr_results.get("n_species", 0) >= 2:
        print()
        print(f"Pearson  r = {corr_results['pearson_r']:.3f}  (p = {corr_results['pearson_p']:.4f})")
        print(f"Spearman r = {corr_results['spearman_r']:.3f}  (p = {corr_results['spearman_p']:.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
