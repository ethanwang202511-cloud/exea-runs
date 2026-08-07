"""Aggregate multi-seed audit results into a single summary table with bootstrap CIs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

def aggregate(tag: str = "multiseed", split_short: str = "02"):
    files = sorted(RES.glob(f"audit_{tag}_norman{split_short}_seed*.csv"))
    if not files:
        print(f"[aggregate] no files matching audit_{tag}_norman{split_short}_seed*.csv")
        return None
    dfs = []
    for f in files:
        seed = int(f.stem.split("seed")[-1])
        d = pd.read_csv(f)
        d['seed'] = seed
        dfs.append(d)
    big = pd.concat(dfs, ignore_index=True)
    big.to_csv(RES / f"audit_{tag}_norman{split_short}_all_seeds.csv", index=False)
    print(f"[aggregate] {len(files)} seed files → {RES / f'audit_{tag}_norman{split_short}_all_seeds.csv'}")

    # Per-mode summary across seeds × pairs
    rows = []
    for mode in big['mode'].unique():
        sub = big[big['mode'] == mode]
        vals = sub['DE_Spearman'].dropna().values
        if len(vals) == 0:
            continue
        # Bootstrap 1000 over (seed × pair) joint
        rng = np.random.default_rng(0)
        boots = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(1000)
        ])
        rows.append({
            "mode": mode,
            "n_obs": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)),
            "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
            "ci95_lower": float(np.percentile(boots, 2.5)),
            "ci95_upper": float(np.percentile(boots, 97.5)),
        })
    summary = pd.DataFrame(rows).sort_values("mean", ascending=False).round(4)
    summary.to_csv(RES / f"audit_{tag}_norman{split_short}_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Pairwise gaps vs learned
    if 'learned' in summary['mode'].values:
        learned_mean = float(summary.loc[summary['mode'] == 'learned', 'mean'].values[0])
        gaps = []
        for _, r in summary.iterrows():
            if r['mode'] == 'learned':
                continue
            gaps.append({
                "mode": r['mode'],
                "gap_vs_learned": learned_mean - r['mean'],
                "fraction_retained": r['mean'] / learned_mean if learned_mean > 0 else float('nan'),
            })
        gaps_df = pd.DataFrame(gaps).round(4)
        gaps_df.to_csv(RES / f"audit_{tag}_norman{split_short}_gaps.csv", index=False)
        print("\nGaps vs learned:")
        print(gaps_df.to_string(index=False))

    return summary

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "multiseed"
    aggregate(tag)
