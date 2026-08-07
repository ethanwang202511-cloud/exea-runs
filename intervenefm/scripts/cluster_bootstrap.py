"""Cluster-bootstrap CIs (resample by PAIR, not by pair×seed).

Reviewer-flagged: 39 pairs × 3 seeds is not 117 independent observations
because the 3 seeds for the same pair are correlated. Cluster-bootstrap
treats the pair as the resampling unit (with all 3 seeds carried), giving
honest CIs.

Also computes the paired Wilcoxon signed-rank for learned vs pop_mean
across pairs (per-pair averaged over seeds).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def cluster_bootstrap(tag: str = "multiseed", split_short: str = "02", n_boot: int = 2000):
    """Compute cluster-bootstrap CIs by resampling pairs (with all seed observations carried).

    Reads audit_{tag}_norman{split}_all_seeds.csv (long-form: pair, mode, seed, DE_Spearman)
    """
    in_file = RES / f"audit_{tag}_norman{split_short}_all_seeds.csv"
    if not in_file.exists():
        # Try aggregated short-form
        files = sorted(RES.glob(f"audit_{tag}_norman{split_short}_seed*.csv"))
        if not files:
            print(f"[cluster_bs] no input"); return
        dfs = []
        for f in files:
            seed = int(f.stem.split("seed")[-1])
            d = pd.read_csv(f); d['seed'] = seed; dfs.append(d)
        big = pd.concat(dfs, ignore_index=True)
    else:
        big = pd.read_csv(in_file)

    rng = np.random.default_rng(0)

    rows = []
    pairs_list = sorted(big['pair'].unique())
    for mode in big['mode'].unique():
        sub = big[big['mode'] == mode].copy()
        # per-pair mean across seeds
        per_pair = sub.groupby('pair')['DE_Spearman'].mean()
        per_pair_vals = per_pair.values
        all_vals = sub['DE_Spearman'].dropna().values

        # Cluster bootstrap by pair
        boots = np.zeros(n_boot)
        for b in range(n_boot):
            chosen = rng.choice(pairs_list, size=len(pairs_list), replace=True)
            boots[b] = per_pair.loc[chosen].mean()

        # Naive bootstrap over (pair x seed) for comparison
        boots_naive = np.zeros(n_boot)
        for b in range(n_boot):
            boots_naive[b] = rng.choice(all_vals, size=len(all_vals), replace=True).mean()

        rows.append({
            "mode": mode,
            "n_pairs": int(len(per_pair_vals)),
            "n_obs_pair_seed": int(len(all_vals)),
            "mean": float(np.mean(per_pair_vals)),
            "cluster_ci_lower": float(np.percentile(boots, 2.5)),
            "cluster_ci_upper": float(np.percentile(boots, 97.5)),
            "naive_ci_lower": float(np.percentile(boots_naive, 2.5)),
            "naive_ci_upper": float(np.percentile(boots_naive, 97.5)),
        })

    out = pd.DataFrame(rows).sort_values("mean", ascending=False).round(4)
    out.to_csv(RES / f"cluster_bootstrap_{tag}_norman{split_short}.csv", index=False)
    print(out.to_string(index=False))

    # Paired Wilcoxon for learned vs each ablation, per-pair
    print("\n=== Paired Wilcoxon (per-pair averaged over seeds): learned vs ablation ===")
    learned_per_pair = big[big['mode'] == 'learned'].groupby('pair')['DE_Spearman'].mean()
    rows2 = []
    for mode in big['mode'].unique():
        if mode == 'learned': continue
        ab = big[big['mode'] == mode].groupby('pair')['DE_Spearman'].mean()
        common = learned_per_pair.index.intersection(ab.index)
        diff = (learned_per_pair.loc[common] - ab.loc[common]).values
        if len(diff) < 5: continue
        try:
            stat, pval = wilcoxon(diff)
        except ValueError as e:
            stat, pval = float('nan'), float('nan')
        rows2.append({
            "mode": mode,
            "median_pair_diff": float(np.median(diff)),
            "mean_pair_diff": float(np.mean(diff)),
            "n_pairs": int(len(diff)),
            "wilcoxon_W": float(stat),
            "wilcoxon_p": float(pval),
        })
    wdf = pd.DataFrame(rows2).round(6)
    wdf.to_csv(RES / f"cluster_bootstrap_wilcoxon_{tag}_norman{split_short}.csv", index=False)
    print(wdf.to_string(index=False))

    # Gap (learned - mode), cluster-bootstrap CI on the gap directly
    print("\n=== Cluster-bootstrap CIs ON THE GAP (learned − mode), by pair ===")
    rows3 = []
    for mode in big['mode'].unique():
        if mode == 'learned': continue
        ab = big[big['mode'] == mode].groupby('pair')['DE_Spearman'].mean()
        common = learned_per_pair.index.intersection(ab.index)
        gap_per_pair = (learned_per_pair.loc[common] - ab.loc[common])
        boots = np.zeros(n_boot)
        for b in range(n_boot):
            chosen = rng.choice(common, size=len(common), replace=True)
            boots[b] = gap_per_pair.loc[chosen].mean()
        rows3.append({
            "mode": mode,
            "gap_mean": float(gap_per_pair.mean()),
            "gap_cluster_ci_lower": float(np.percentile(boots, 2.5)),
            "gap_cluster_ci_upper": float(np.percentile(boots, 97.5)),
            "n_pairs": int(len(common)),
        })
    gdf = pd.DataFrame(rows3).round(4)
    gdf.to_csv(RES / f"cluster_bootstrap_gaps_{tag}_norman{split_short}.csv", index=False)
    print(gdf.to_string(index=False))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "multiseed"
    cluster_bootstrap(tag)
