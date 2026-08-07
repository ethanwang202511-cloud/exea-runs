"""Per-effect-size split analysis (reviewer Issue 7).

Question: why does GEARS on Norman show negligible gap on full-transcriptome
Pearson δ (+0.017, CI crosses zero) but small-to-medium gap on top-200-DEG
Spearman ρ (+0.066, CI strictly above zero)? Hypothesis from the reviewer:
"Cellular FMs add value only on the tails of the response distribution."

Method (no new training, uses existing audit CSVs):
1. Load all 4 GEARS Norman seeds (audit_gears_norman_e5_seed{1,2,3,4}.csv).
2. Pivot to per-(condition × seed × mode) records of DE-Spearman ρ AND
   Pearson δ on full transcriptome.
3. Per (condition × seed), compute gap_DE = learned − pop_mean (DE-Spearman)
   and gap_P = learned − pop_mean (Pearson δ).
4. Use pop_mean DE-Spearman ρ as a "mean-axis-alignment" proxy for the
   perturbation's effect type:
   - HIGH pop_mean ρ (≥ 0.65 = top tertile): the perturbation aligns with
     the global cell-cycle / stress axis; pop_mean already captures most
     of the response.
   - MED pop_mean ρ (0.45–0.65): mixed alignment.
   - LOW pop_mean ρ (< 0.45 = bottom tertile): the perturbation is
     gene-specific / orthogonal to the global axis.
5. Compare gap_DE and gap_P within each stratum.
6. Also stratify by |obs_delta| proxy: use 1 − pop_mean DE-Spearman ρ as
   "how much of the response is gene-specific."

Outputs:
  - results/effect_size_split_gears_norman.csv (per-stratum gap means + CIs)
  - prints summary
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_seeds(prefix: str, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for s in seeds:
        p = RESULTS / f"{prefix}_seed{s}.csv"
        if not p.exists():
            print(f"  [skip seed {s}] {p.name} not found")
            continue
        df = pd.read_csv(p)
        df["seed"] = s
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def per_condition_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide-format per (condition × seed) with cols
    DE_learned, DE_pop_mean, P_learned, P_pop_mean."""
    pivot = df.pivot_table(
        index=("seed", "condition"),
        columns="mode",
        values=("DE_Spearman", "Pearson_delta_full"),
    )
    out = pd.DataFrame(index=pivot.index)
    out["DE_learned"] = pivot[("DE_Spearman", "learned")]
    out["DE_pop"] = pivot[("DE_Spearman", "pop_mean")]
    out["P_learned"] = pivot[("Pearson_delta_full", "learned")]
    out["P_pop"] = pivot[("Pearson_delta_full", "pop_mean")]
    out["gap_DE"] = out["DE_learned"] - out["DE_pop"]
    out["gap_P"] = out["P_learned"] - out["P_pop"]
    return out.dropna()


def cluster_bootstrap_by_condition(values: pd.Series, n_boot: int = 2000, seed: int = 0):
    """Cluster bootstrap by condition; values is indexed by (seed, condition)."""
    rng = np.random.default_rng(seed)
    by_cond = values.groupby(level="condition")
    cond_groups = {c: v.values for c, v in by_cond}
    conds = list(cond_groups.keys())
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(conds, size=len(conds), replace=True)
        vals = np.concatenate([cond_groups[c] for c in sample])
        boot_means.append(vals.mean())
    boot = np.array(boot_means)
    return float(boot.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def stratify_and_summarize(out: pd.DataFrame, label: str) -> list[dict]:
    """Per-stratum summary using `pop_axis_alignment` = DE_pop tertiles."""
    # Compute stratum on per-condition mean DE_pop (averaged across seeds for stability)
    cond_pop_mean = out.groupby(level="condition")["DE_pop"].mean()
    # Tertiles
    q33 = cond_pop_mean.quantile(1/3)
    q66 = cond_pop_mean.quantile(2/3)

    strata = {
        "low (gene-specific perts; pop_mean DE-ρ < q33)": cond_pop_mean[cond_pop_mean < q33].index,
        "mid (q33 ≤ pop_mean DE-ρ < q66)": cond_pop_mean[(cond_pop_mean >= q33) & (cond_pop_mean < q66)].index,
        "high (mean-axis perts; pop_mean DE-ρ ≥ q66)": cond_pop_mean[cond_pop_mean >= q66].index,
    }

    rows = []
    print(f"\n=== {label} stratified by pop_mean DE-Spearman tertile ===")
    print(f"  Tertile thresholds: q33 = {q33:.3f}; q66 = {q66:.3f}")
    print(f"  {'stratum':<55} {'n_cond':>6} {'gap_DE_mean':>12} {'CI_DE':>22} {'gap_P_mean':>12} {'CI_P':>22}")
    for stratum_name, cond_idx in strata.items():
        sub = out[out.index.get_level_values("condition").isin(cond_idx)]
        gap_de_m, gap_de_lo, gap_de_hi = cluster_bootstrap_by_condition(sub["gap_DE"])
        gap_p_m, gap_p_lo, gap_p_hi = cluster_bootstrap_by_condition(sub["gap_P"])
        n = len(cond_idx)
        print(f"  {stratum_name:<55} {n:>6} {gap_de_m:>+12.4f} [{gap_de_lo:>+7.4f},{gap_de_hi:>+7.4f}] {gap_p_m:>+12.4f} [{gap_p_lo:>+7.4f},{gap_p_hi:>+7.4f}]")
        rows.append({
            "label": label,
            "stratum": stratum_name,
            "n_unique_conditions": n,
            "gap_DE_mean": gap_de_m,
            "gap_DE_ci_lo": gap_de_lo,
            "gap_DE_ci_hi": gap_de_hi,
            "gap_P_mean": gap_p_m,
            "gap_P_ci_lo": gap_p_lo,
            "gap_P_ci_hi": gap_p_hi,
        })
    return rows


def stratify_by_effect_strength(out: pd.DataFrame, label: str) -> list[dict]:
    """An alternative split: strongest-effect perturbations (high |learned − ctrl|)
    vs weakest. Use mae_delta_top200 NOT available for pop_mean directly,
    so we proxy "effect strength" by the absolute learned DE-Spearman ρ.

    A strongly-affected perturbation has large |response|; pop_mean and
    learned both respond. A weakly-affected perturbation produces noise
    around basal; even small per-gene differences can dominate metrics."""
    cond_de_learn = out.groupby(level="condition")["DE_learned"].mean()
    q33 = cond_de_learn.quantile(1/3)
    q66 = cond_de_learn.quantile(2/3)
    strata = {
        "weak-effect perts (learned DE-ρ < q33)": cond_de_learn[cond_de_learn < q33].index,
        "mid-effect (q33 ≤ DE-ρ < q66)": cond_de_learn[(cond_de_learn >= q33) & (cond_de_learn < q66)].index,
        "strong-effect (DE-ρ ≥ q66)": cond_de_learn[cond_de_learn >= q66].index,
    }
    rows = []
    print(f"\n=== {label} stratified by learned DE-Spearman tertile (effect strength proxy) ===")
    print(f"  Tertile thresholds: q33 = {q33:.3f}; q66 = {q66:.3f}")
    print(f"  {'stratum':<55} {'n_cond':>6} {'gap_DE_mean':>12} {'CI_DE':>22} {'gap_P_mean':>12} {'CI_P':>22}")
    for stratum_name, cond_idx in strata.items():
        sub = out[out.index.get_level_values("condition").isin(cond_idx)]
        gap_de_m, gap_de_lo, gap_de_hi = cluster_bootstrap_by_condition(sub["gap_DE"])
        gap_p_m, gap_p_lo, gap_p_hi = cluster_bootstrap_by_condition(sub["gap_P"])
        n = len(cond_idx)
        print(f"  {stratum_name:<55} {n:>6} {gap_de_m:>+12.4f} [{gap_de_lo:>+7.4f},{gap_de_hi:>+7.4f}] {gap_p_m:>+12.4f} [{gap_p_lo:>+7.4f},{gap_p_hi:>+7.4f}]")
        rows.append({
            "label": label + " | by effect strength",
            "stratum": stratum_name,
            "n_unique_conditions": n,
            "gap_DE_mean": gap_de_m,
            "gap_DE_ci_lo": gap_de_lo,
            "gap_DE_ci_hi": gap_de_hi,
            "gap_P_mean": gap_p_m,
            "gap_P_ci_lo": gap_p_lo,
            "gap_P_ci_hi": gap_p_hi,
        })
    return rows


def main():
    print("# Effect-size split analysis on GEARS / Norman 2019")
    df = load_seeds("audit_gears_norman_e5", [1, 2, 3, 4])
    print(f"[load] {len(df)} rows; {df['condition'].nunique()} unique conditions; {df['seed'].nunique()} seeds")
    out = per_condition_pivot(df)
    print(f"[pivot] {len(out)} (seed × condition) records")

    # Overall
    overall_de = cluster_bootstrap_by_condition(out["gap_DE"])
    overall_p = cluster_bootstrap_by_condition(out["gap_P"])
    print(f"\n=== Overall ===")
    print(f"  gap_DE = {overall_de[0]:+.4f} [{overall_de[1]:+.4f}, {overall_de[2]:+.4f}]")
    print(f"  gap_P  = {overall_p[0]:+.4f} [{overall_p[1]:+.4f}, {overall_p[2]:+.4f}]")

    rows = []
    rows += stratify_and_summarize(out, "GEARS Norman 2019")
    rows += stratify_by_effect_strength(out, "GEARS Norman 2019")

    pd.DataFrame(rows).to_csv(RESULTS / "effect_size_split_gears_norman.csv", index=False)
    print(f"\n[saved] {RESULTS / 'effect_size_split_gears_norman.csv'}")


if __name__ == "__main__":
    main()
