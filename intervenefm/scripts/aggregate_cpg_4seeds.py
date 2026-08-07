"""Aggregate 4-seed CPG audit on Replogle K562: gap = learned − pop_mean per
test gene, then compute multi-seed paired CI and Cohen's d.

Run after all 4 audit_cpg_replogle_k562_seed{0,1,2,3}.csv exist.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_seed(seed: int) -> pd.DataFrame:
    p = RESULTS / f"audit_cpg_replogle_k562_seed{seed}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["seed"] = seed
    return df


def main() -> None:
    seeds = [0, 1, 2, 3]
    all_dfs = [load_seed(s) for s in seeds]
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"[load] {len(seeds)} seeds, {df['test_gene'].nunique()} unique test genes per seed")

    # per-(seed, gene) gap = learned - pop_mean
    pivot = df.pivot_table(
        index=("seed", "test_gene"),
        columns="mode",
        values=("DE_Spearman", "Pearson_delta_full"),
    )
    gap_des = pivot[("DE_Spearman", "learned")] - pivot[("DE_Spearman", "pop_mean")]
    gap_pearson = pivot[("Pearson_delta_full", "learned")] - pivot[("Pearson_delta_full", "pop_mean")]

    # Per-seed summary
    print("\n=== Per-seed CPG gap (learned − pop_mean) ===")
    print(f"{'seed':>4} {'n':>4} {'DE_gap_mean':>13} {'Pearson_gap_mean':>17}")
    per_seed_gaps = []
    for s in seeds:
        sub = gap_des.xs(s, level="seed").dropna()
        sub_p = gap_pearson.xs(s, level="seed").dropna()
        per_seed_gaps.append({"seed": s, "n": len(sub),
                              "DE_gap": float(sub.mean()),
                              "Pearson_gap": float(sub_p.mean())})
        print(f"{s:>4} {len(sub):>4} {sub.mean():>+13.4f} {sub_p.mean():>+17.4f}")

    # Pooled (gene × seed)
    pooled_de = gap_des.dropna()
    pooled_p = gap_pearson.dropna()
    print(f"\n=== Pooled (gene × seed) DE-Spearman gap, n={len(pooled_de)} ===")
    print(f"  mean = {pooled_de.mean():+.4f}, std = {pooled_de.std(ddof=1):.4f}, "
          f"d = {pooled_de.mean()/pooled_de.std(ddof=1):+.3f}")

    # Cluster bootstrap by test_gene (resample genes, take all 4 seeds per gene)
    rng = np.random.default_rng(0)
    by_gene_de = pooled_de.groupby(level="test_gene")
    by_gene_p = pooled_p.groupby(level="test_gene")
    genes = sorted(set(idx for idx, _ in pooled_de.index))
    boot_de = []
    boot_p = []
    for _ in range(2000):
        sample = rng.choice(genes, size=len(genes), replace=True)
        vals_de = []
        vals_p = []
        for g in sample:
            try:
                vals_de.extend(by_gene_de.get_group(g).values)
                vals_p.extend(by_gene_p.get_group(g).values)
            except KeyError:
                pass
        if len(vals_de):
            boot_de.append(np.mean(vals_de))
            boot_p.append(np.mean(vals_p))
    boot_de = np.array(boot_de)
    boot_p = np.array(boot_p)
    print(f"  Cluster CI (by gene, n_boot=2000): "
          f"DE [{np.percentile(boot_de,2.5):+.4f}, {np.percentile(boot_de,97.5):+.4f}]; "
          f"Pearson [{np.percentile(boot_p,2.5):+.4f}, {np.percentile(boot_p,97.5):+.4f}]")

    # Gap reduction vs CPA baseline (-0.161 from 3-seed standard CPA)
    cpa_baseline = -0.161
    pct_closure = (1 - abs(pooled_de.mean()) / abs(cpa_baseline)) * 100
    print(f"\n  CPA baseline: {cpa_baseline:+.4f}; CPG (4-seed): {pooled_de.mean():+.4f}")
    print(f"  Gap closure: {pct_closure:.0f}%")

    # Save
    out = RESULTS / "audit_cpg_4seed_summary.csv"
    pd.DataFrame(per_seed_gaps).to_csv(out, index=False)
    print(f"\n[saved] {out}")

    # Save pooled bootstrap for citation
    out2 = RESULTS / "audit_cpg_4seed_bootstrap.csv"
    pd.DataFrame({"DE_gap_pooled_mean": [float(pooled_de.mean())],
                  "DE_gap_pooled_std": [float(pooled_de.std(ddof=1))],
                  "DE_cohens_d": [float(pooled_de.mean()/pooled_de.std(ddof=1))],
                  "DE_cluster_ci_lo": [float(np.percentile(boot_de, 2.5))],
                  "DE_cluster_ci_hi": [float(np.percentile(boot_de, 97.5))],
                  "Pearson_gap_pooled_mean": [float(pooled_p.mean())],
                  "Pearson_cluster_ci_lo": [float(np.percentile(boot_p, 2.5))],
                  "Pearson_cluster_ci_hi": [float(np.percentile(boot_p, 97.5))],
                  "n_pooled": [len(pooled_de)],
                  "n_seeds": [len(seeds)],
                  "n_genes": [len(genes)],
                  "cpa_baseline_de": [cpa_baseline],
                  "pct_closure": [pct_closure]}).to_csv(out2, index=False)
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
