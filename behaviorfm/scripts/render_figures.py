"""Render all experiment figures from results/*.csv to figures/*.{pdf,png}."""

from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "figure.figsize": (6.5, 4.0),
})


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# -----------------------------------------------------------------------
# Figure 1: Pareto frontier (params vs PCK@0.05)
# -----------------------------------------------------------------------
def fig1_pareto():
    pareto = load_csv(RES_DIR / "h1_pareto.csv")
    decoder_ft = load_csv(RES_DIR / "decoder_ft_comparator.csv")
    species_data = defaultdict(lambda: defaultdict(list))
    for r in pareto:
        sp = r["species"]; var = r["variant"]
        n_p = int(r["n_adapt_params"])
        species_data[sp][n_p].append(float(r["pck@0.05"]))
    for r in decoder_ft:
        sp = r["species"]
        n_p = int(r["n_trainable"])
        species_data[sp][n_p].append(float(r["post_pck_at_005"]))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"elephant": "#2E86AB", "chimpanzee": "#A23B72", "gorilla": "#F18F01",
              "rabbit": "#3CA040", "panther": "#7E57C2"}
    markers = {"elephant": "o", "chimpanzee": "s", "gorilla": "D",
               "rabbit": "^", "panther": "v"}
    for sp in ("elephant", "chimpanzee", "gorilla", "rabbit", "panther"):
        if sp not in species_data:
            continue
        xs = sorted(species_data[sp].keys())
        ys = [np.mean(species_data[sp][x]) for x in xs]
        ystd = [np.std(species_data[sp][x]) if len(species_data[sp][x]) > 1 else 0 for x in xs]
        sp_cos = {"elephant": 0.50, "chimpanzee": 0.49, "gorilla": 0.52,
                  "rabbit": 0.55, "panther": 0.76}[sp]
        ax.errorbar(xs, ys, yerr=ystd, marker=markers[sp], color=colors[sp],
                    label=f"{sp} (held-out, cos={sp_cos:.2f})",
                    linestyle="-", linewidth=1.5, markersize=8, capsize=3)
        # Annotate sweet spot
        if sp == "elephant":
            sweet_x = 296064; sweet_y = np.mean(species_data[sp].get(sweet_x, [0]))
            ax.annotate(f"sweet spot\n(296K params)",
                        xy=(sweet_x, sweet_y), xytext=(sweet_x*0.05, sweet_y+0.06),
                        fontsize=9, ha="left",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.axhline(y=0.611, color="gray", linestyle=":", linewidth=1, alpha=0.6,
               label="within-species ceiling (PCK@0.05 = 0.611, 5 source species)")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters per held-out species (log scale)")
    ax.set_ylabel("PCK@0.05 (held-out species val)")
    ax.set_title("Figure 1: Parameter-efficiency vs accuracy Pareto frontier")
    # Move legend outside the data area to avoid overlap with markers/error bars.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(0, 0.65)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure1_pareto.pdf")
    fig.savefig(FIG_DIR / "figure1_pareto.png")
    plt.close(fig)
    print("[ok] figure1_pareto")


# -----------------------------------------------------------------------
# Figure 2: Zero-shot RMSE phase curve across 54 species
# -----------------------------------------------------------------------
def fig2_zero_shot_curve():
    rows = load_csv(RES_DIR / "zero_shot_distance_curve.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    in_pool_x, in_pool_y = [], []
    out_pool_x, out_pool_y, out_pool_names = [], [], []
    for r in rows:
        if not r["cos_max_to_train_pool"]:
            continue
        x = float(r["cos_max_to_train_pool"])
        y = float(r["rmse_norm"])
        if r["in_training_pool"] == "True":
            in_pool_x.append(x); in_pool_y.append(y)
        else:
            out_pool_x.append(x); out_pool_y.append(y); out_pool_names.append(r["species"])

    ax.scatter(out_pool_x, out_pool_y, s=24, color="#A23B72", alpha=0.7,
               label=f"out-of-pool species (n={len(out_pool_x)})")
    ax.scatter(in_pool_x, in_pool_y, s=80, color="#2E86AB", marker="*",
               label=f"in-pool training species (n={len(in_pool_x)})", zorder=5)

    # Annotate the H1 binding species. Stagger label offsets so chimpanzee
    # and gorilla (close in cos_max) don't overprint each other.
    label_offsets = {"elephant": (5, -3), "chimpanzee": (5, 6), "gorilla": (5, -10)}
    for x, y, name in zip(out_pool_x, out_pool_y, out_pool_names):
        if name in label_offsets:
            ax.annotate(name, (x, y), xytext=label_offsets[name],
                        textcoords="offset points",
                        fontsize=8, color="#A23B72", weight="bold")

    # Bin medians overlay
    bins = [(0.0, 0.55, "far"), (0.55, 0.70, "mid"), (0.70, 1.0, "near")]
    for lo, hi, label in bins:
        sub = [(x, y) for x, y in zip(out_pool_x, out_pool_y) if lo <= x < hi]
        if sub:
            xs, ys = zip(*sub)
            ax.scatter([np.mean(xs)], [np.mean(ys)], s=120, marker="D",
                       color="orange", edgecolor="black", zorder=10,
                       label=f"bin mean: {label} = {np.mean(ys):.3f}" if lo == 0.0 else None)

    ax.set_xlabel("cos_max similarity to training pool (DINOv2-base patch features)")
    ax.set_ylabel("Zero-shot RMSE (normalized to bbox diagonal)")
    ax.set_title("Figure 2: Zero-shot RMSE phase curve across 54 AP-10K species")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(FIG_DIR / "figure2_zero_shot_curve.pdf")
    fig.savefig(FIG_DIR / "figure2_zero_shot_curve.png")
    plt.close(fig)
    print("[ok] figure2_zero_shot_curve")


# -----------------------------------------------------------------------
# Figure 3: Token-utility aux loss restores prediction sensitivity
# -----------------------------------------------------------------------
def fig3_token_sensitivity():
    # Hard-coded from inner_loop_diagnostic logs (paper §5.6.1, §5.6.2)
    # Panel A: Stage-2 cross-attn no-aux ckpt
    # Panel B: Stage-1 cross-attn + aux ckpt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    lrs = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    val_no_aux  = [0.1333, 0.1333, 0.1333, 0.1333, 0.1328, 0.1248]
    val_aux     = [0.1410, 0.1406, 0.1356, 0.1334, 0.1244, 0.1350]
    init_no_aux, init_aux = 0.1333, 0.1410
    movement_no_aux = [0.0000, 0.0004, 0.0040, 0.0415, 1.1796, 11.0684]
    movement_aux    = [0.0019, 0.0210, 0.3198, 1.0929, 5.2139, 12.8958]
    std_no_aux, std_aux = 0.00010, 0.00083

    ax = axes[0]
    ax.plot(lrs, val_no_aux, "o-", color="#A23B72", label="no aux loss", lw=1.5)
    ax.plot(lrs, val_aux,    "s-", color="#2E86AB", label="with aux loss", lw=1.5)
    ax.axhline(y=init_no_aux, color="#A23B72", linestyle=":", alpha=0.6,
               label=f"no-aux init = {init_no_aux:.4f}")
    ax.axhline(y=init_aux, color="#2E86AB", linestyle=":", alpha=0.6,
               label=f"aux init = {init_aux:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Inner-loop SGD learning rate")
    ax.set_ylabel("Elephant val RMSE_norm\nafter 50 SGD steps")
    ax.set_title("(a) Inner-loop adaptation needs aux loss", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bars = ax.bar(["no aux loss", "with aux loss"], [std_no_aux, std_aux],
                  color=["#A23B72", "#2E86AB"], edgecolor="black")
    for bar, v in zip(bars, [std_no_aux, std_aux]):
        ax.text(bar.get_x() + bar.get_width()/2, v, f" {v:.5f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Per-coord prediction std\n(random tokens, [0,1] units)")
    ax.set_title("(b) Aux loss restores id_token sensitivity (8.3×)", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 3: The token-utility aux loss is load-bearing", fontsize=12, y=0.99)
    # Reserve top space for suptitle, generous left/right margins for y-axis labels.
    fig.tight_layout(rect=[0.04, 0.02, 0.98, 0.92])
    fig.savefig(FIG_DIR / "figure3_token_sensitivity.pdf")
    fig.savefig(FIG_DIR / "figure3_token_sensitivity.png")
    plt.close(fig)
    print("[ok] figure3_token_sensitivity")


# -----------------------------------------------------------------------
# Figure 4: Per-backbone Wasserstein dynamic-range scout
# -----------------------------------------------------------------------
def fig4_wasserstein():
    rows = load_csv(RES_DIR / "wasserstein_scout.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    metrics = [
        ("cos_sim_max_to_train_species_mean", "Cosine sim (max to training species)", True),
        ("euclid_d_to_training_centroid", "L2 distance to training-pool centroid", False),
        ("sinkhorn_w2_min", "Sinkhorn-W2 (min over training species)", False),
    ]
    for ax, (col, title, inv) in zip(axes, metrics):
        in_vals, out_vals = [], []
        for r in rows:
            v = r.get(col, "")
            if v == "" or v == "nan":
                continue
            v = float(v)
            if r["in_training_pool"] == "True":
                in_vals.append(v)
            else:
                out_vals.append(v)
        bp = ax.boxplot([in_vals, out_vals], labels=["in-pool\n(n=5)", "out-of-pool\n(n=49)"],
                        patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#2E86AB", "#A23B72"]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        # spread annotation
        if in_vals and out_vals:
            spread = (np.mean(out_vals) - np.mean(in_vals)) / max(np.std(in_vals), 1e-6)
            ax.set_title(f"spread = {abs(spread):.2f}σ", fontsize=10)

    fig.suptitle("Figure 4: Per-backbone Wasserstein distance has the dynamic range\nrequired for the Year-2 H0 phase-diagram lock (≥ 2σ on Sinkhorn-W2-min)",
                 fontsize=11)
    # Reserve top space so suptitle (2 lines) doesn't collide with subplot titles.
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(FIG_DIR / "figure4_wasserstein.pdf")
    fig.savefig(FIG_DIR / "figure4_wasserstein.png")
    plt.close(fig)
    print("[ok] figure4_wasserstein")


# -----------------------------------------------------------------------
# Figure 6: H1 across 3 held-out species, n=10 paired CIs
# -----------------------------------------------------------------------
def fig6_h1_three_species():
    species = [
        ("elephant",    0.501, -0.0134, -0.0161, -0.0102, 10),
        ("chimpanzee",  0.493, -0.0098, -0.0145, -0.0052,  9),
        ("gorilla",     0.523, -0.0161, -0.0202, -0.0117, 10),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ys = list(range(len(species)))
    means = [s[2] for s in species]
    los = [s[3] for s in species]
    his = [s[4] for s in species]
    err_lo = [m - lo for m, lo in zip(means, los)]
    err_hi = [hi - m for m, hi in zip(means, his)]
    ax.errorbar(means, ys, xerr=[err_lo, err_hi], fmt="s", color="#2E86AB",
                markersize=10, capsize=5, lw=1.5)
    # Place the per-species labels above each marker (no overlap with bars).
    for i, (name, cos, m, lo, hi, n_pos) in enumerate(species):
        ax.annotate(f"{name} (cos={cos:.2f}); {n_pos}/10 seeds favor",
                    (m, i), xytext=(0, 12), textcoords="offset points",
                    fontsize=9, ha="center", va="bottom")
    ax.axvline(x=0, color="black", linestyle="-", lw=0.8)
    ax.axvline(x=-0.0134 * (1 - 0.15), color="orange", linestyle=":",
               alpha=0.5, label="(−15% threshold for v1.4 H1, illustrative)")
    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in species])
    ax.set_xlabel("Mean delta RMSE_norm (adapt − no-adapt-mean)\nNegative = adaptation improves over baseline")
    ax.set_title("Figure 6: H1-prime adaptation gain across 3 held-out species (n_seeds=10)\n95% paired-bootstrap CIs do NOT cross zero on all 3 species")
    ax.set_xlim(-0.025, 0.005)
    ax.set_ylim(-0.5, len(species) - 0.3)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure6_h1_three_species.pdf")
    fig.savefig(FIG_DIR / "figure6_h1_three_species.png")
    plt.close(fig)
    print("[ok] figure6_h1_three_species")


def fig7_h1_distance_sweep():
    """H1 adaptation gain (RMSE delta and PCK ratio) across 8 held-out species
    at n_seeds=10, spanning cos_max 0.46-0.76."""
    species = [
        # name, cos, adapt RMSE, baseline RMSE, adapt PCK, baseline PCK
        ("alouatta",   0.456, 0.1507, 0.1599, 0.178, 0.155),
        ("chimpanzee", 0.493, 0.1697, 0.1795, 0.135, 0.117),
        ("elephant",   0.501, 0.1223, 0.1357, 0.194, 0.147),
        ("monkey",     0.520, 0.1557, 0.1623, 0.165, 0.140),
        ("gorilla",    0.523, 0.1666, 0.1827, 0.190, 0.142),
        ("rabbit",     0.552, 0.0626, 0.0839, 0.536, 0.456),
        ("fox",        0.746, 0.0686, 0.0811, 0.479, 0.422),
        ("panther",    0.760, 0.0653, 0.0780, 0.483, 0.427),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: relative RMSE reduction vs cos_max
    ax = axes[0]
    cos = [s[1] for s in species]
    rel_red = [(s[2] - s[3]) / s[3] * 100 for s in species]   # negative = improvement
    names = [s[0] for s in species]
    sc = ax.scatter(cos, rel_red, s=120, c=rel_red, cmap="coolwarm_r",
                    edgecolor="black", zorder=5, vmin=-30, vmax=0)
    # Stagger labels: alternate above/below; offset close-cos pairs further.
    label_offsets = {
        "alouatta":   (-30, 8),
        "chimpanzee": (-50, 8),
        "elephant":   (8, 4),
        "monkey":     (8, -10),
        "gorilla":    (8, 8),
        "rabbit":     (-8, -14),
        "fox":        (-30, -14),
        "panther":    (8, 6),
    }
    for c, r, n in zip(cos, rel_red, names):
        dx, dy = label_offsets.get(n, (4, -3))
        ax.annotate(n, (c, r), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8)
    ax.axhline(y=-15, color="orange", linestyle="--", alpha=0.7,
               label="v1.4 H1 within-15% threshold")
    ax.axhline(y=0, color="black", linestyle="-", lw=0.8)
    ax.set_xlabel("cos_max similarity to training pool (DINOv2-base patch features)")
    ax.set_ylabel("Adaptation RMSE relative change (%)\n(negative = adaptation improves)")
    ax.set_title("(a) H1 adaptation gain across 7 held-out species\n(rabbit and panther PASS the v1.4 within-15% threshold)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-30, 5)

    # Right: PCK absolute (adapt vs baseline) bar
    ax = axes[1]
    x = np.arange(len(species))
    w = 0.4
    base_pck = [s[5] for s in species]
    adapt_pck = [s[4] for s in species]
    ax.bar(x - w/2, base_pck, w, label="no-adapt mean token", color="#A23B72", alpha=0.8)
    ax.bar(x + w/2, adapt_pck, w, label="adapt-ITI (50 SGD)", color="#2E86AB", alpha=0.8)
    for i, (b, a) in enumerate(zip(base_pck, adapt_pck)):
        rel = (a - b) / b * 100
        ax.annotate(f"+{rel:.0f}%" if rel >= 0 else f"{rel:.0f}%",
                    (i, max(a, b)), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8, weight="bold",
                    color="#2E86AB" if rel > 0 else "#A23B72")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(cos={c:.2f})" for n, c in zip(names, cos)],
                       fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("PCK@0.05 (held-out species val)")
    ax.set_title("(b) Absolute PCK@0.05 lift")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 7: H1-prime adaptation gain across 7 held-out species (cos 0.49 to 0.76)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "figure7_h1_distance_sweep.pdf")
    fig.savefig(FIG_DIR / "figure7_h1_distance_sweep.png")
    plt.close(fig)
    print("[ok] figure7_h1_distance_sweep")


if __name__ == "__main__":
    fig1_pareto()
    fig2_zero_shot_curve()
    fig3_token_sensitivity()
    fig4_wasserstein()
    fig6_h1_three_species()
    fig7_h1_distance_sweep()
    print(f"\nAll figures rendered to {FIG_DIR}")
