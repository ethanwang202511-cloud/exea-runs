"""Aggregate multi-seed CPG audits with pooled stats and cluster bootstrap by gene.

Replaces aggregate_cpg_4seeds.py (which had a MultiIndex bug in the bootstrap loop).
Produces audit_cpg_k562_multiseed_summary.csv and audit_cpg_rpe1_multiseed_summary.csv.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def aggregate(prefix: str, seeds: list[int], cpa_baseline: float, label: str) -> dict:
    rows = []
    for s in seeds:
        p = RESULTS / f"{prefix}_seed{s}.csv"
        if not p.exists():
            print(f"  [skip seed {s}] {p.name} not found")
            continue
        df = pd.read_csv(p)
        df["seed"] = s
        rows.append(df)
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    pivot = df.pivot_table(
        index=("seed", "test_gene"),
        columns="mode",
        values=("DE_Spearman", "Pearson_delta_full"),
    )
    gap_de = (pivot[("DE_Spearman", "learned")] - pivot[("DE_Spearman", "pop_mean")]).dropna()
    gap_p = (pivot[("Pearson_delta_full", "learned")] - pivot[("Pearson_delta_full", "pop_mean")]).dropna()

    print(f"\n=== {label} ===")
    print(f"  n_seeds={len(rows)}; n_pooled={len(gap_de)}")
    for s in seeds:
        if s in gap_de.index.get_level_values("seed"):
            sub = gap_de.xs(s, level="seed")
            sub_p = gap_p.xs(s, level="seed")
            print(f"  seed {s}: n={len(sub)}  DE_gap={sub.mean():+.4f}  Pearson_gap={sub_p.mean():+.4f}")

    pooled_de_mean = float(gap_de.mean())
    pooled_de_std = float(gap_de.std(ddof=1))
    pooled_de_d = pooled_de_mean / pooled_de_std
    pooled_p_mean = float(gap_p.mean())

    # Cluster bootstrap by test_gene
    rng = np.random.default_rng(0)
    gap_de_by_gene = gap_de.groupby(level="test_gene")
    gap_p_by_gene = gap_p.groupby(level="test_gene")
    gene_groups_de = {g: vals.values for g, vals in gap_de_by_gene}
    gene_groups_p = {g: vals.values for g, vals in gap_p_by_gene}
    genes = list(gene_groups_de.keys())
    boot_de, boot_p = [], []
    for _ in range(2000):
        sample = rng.choice(genes, size=len(genes), replace=True)
        vals_de = np.concatenate([gene_groups_de[g] for g in sample])
        vals_p = np.concatenate([gene_groups_p[g] for g in sample])
        boot_de.append(vals_de.mean())
        boot_p.append(vals_p.mean())
    boot_de = np.array(boot_de)
    boot_p = np.array(boot_p)
    de_lo, de_hi = float(np.percentile(boot_de, 2.5)), float(np.percentile(boot_de, 97.5))
    p_lo, p_hi = float(np.percentile(boot_p, 2.5)), float(np.percentile(boot_p, 97.5))

    pct_closure = (1 - abs(pooled_de_mean) / abs(cpa_baseline)) * 100

    print(f"  pooled DE_gap     = {pooled_de_mean:+.4f}  std={pooled_de_std:.4f}  d={pooled_de_d:+.3f}")
    print(f"  cluster CI by gene (DE)      = [{de_lo:+.4f}, {de_hi:+.4f}]")
    print(f"  cluster CI by gene (Pearson) = [{p_lo:+.4f}, {p_hi:+.4f}]")
    print(f"  CPA baseline = {cpa_baseline:+.4f}; closure = {pct_closure:.1f}%")

    return {
        "label": label,
        "n_seeds": len(rows),
        "n_pooled": len(gap_de),
        "n_unique_genes": len(genes),
        "DE_gap_mean": pooled_de_mean,
        "DE_gap_std": pooled_de_std,
        "DE_cohens_d": pooled_de_d,
        "DE_cluster_ci_lo": de_lo,
        "DE_cluster_ci_hi": de_hi,
        "Pearson_gap_mean": pooled_p_mean,
        "Pearson_cluster_ci_lo": p_lo,
        "Pearson_cluster_ci_hi": p_hi,
        "cpa_baseline_de": cpa_baseline,
        "pct_closure": pct_closure,
    }


def main() -> None:
    rows = []
    rows.append(aggregate("audit_cpg_replogle_k562", [0, 1, 2, 3],
                          cpa_baseline=-0.161, label="CPG K562 (4-seed)"))
    rows.append(aggregate("audit_cpg_replogle_rpe1", [0, 1, 2, 3],
                          cpa_baseline=-0.071, label="CPG RPE1 (4-seed)"))
    rows.append(aggregate("audit_cpg_replogle_k562_long", [0, 1, 2],
                          cpa_baseline=-0.161, label="CPG K562 long-train (3-seed)"))
    rows = [r for r in rows if r]
    out = pd.DataFrame(rows)
    p = RESULTS / "audit_cpg_multiseed_summary.csv"
    out.to_csv(p, index=False)
    print(f"\n[saved] {p}")


if __name__ == "__main__":
    main()
