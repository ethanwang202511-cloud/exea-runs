"""Pull and aggregate GEARS multi-seed results from Modal volume.

Reads results/audit_gears_norman_e5_seed{1,2,3,4}.csv (seed 1 was the original
single-seed run; seeds 2,3,4 are the multi-seed extension). Computes paired
bootstrap CIs across (condition × seed) pairs.
"""
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def fetch_modal():
    """Pull seed-2/3/4 audit CSVs from Modal volume."""
    for seed in [2, 3, 4]:
        for prefix in ["audit_gears_norman_e5", "audit_summary_gears_norman_e5"]:
            remote = f"results/{prefix}_seed{seed}.csv"
            local = RES / f"{prefix}_seed{seed}.csv"
            cmd = ["modal", "volume", "get", "intervenefm-data", remote, str(local), "--force"]
            print(f"[fetch] {remote}")
            rc = subprocess.run(cmd, capture_output=True).returncode
            if rc != 0:
                print(f"  FAILED rc={rc}; file may not yet exist")


def aggregate():
    """Aggregate seeds 1-4 into a single 4-seed result."""
    dfs = []
    for seed in [1, 2, 3, 4]:
        f = RES / f"audit_gears_norman_e5_seed{seed}.csv"
        if not f.exists():
            print(f"[skip] {f}"); continue
        d = pd.read_csv(f)
        d['seed'] = seed
        dfs.append(d)
    if not dfs:
        print("[aggregate] no files")
        return
    big = pd.concat(dfs, ignore_index=True)
    big.to_csv(RES / "audit_gears_norman_e5_allseeds.csv", index=False)
    print(f"[aggregate] {len(dfs)} seeds, {len(big)} (condition × mode × seed) rows")

    print("\n=== GEARS multi-seed audit summary (n={} obs per mode = 107 conds × {} seeds) ===".format(
        len(big[big['mode'] == 'learned']), big['seed'].nunique()))

    rng = np.random.default_rng(0)
    rows = []
    for mode in big['mode'].unique():
        sub = big[big['mode'] == mode]
        de_vals = sub['DE_Spearman'].dropna().values
        pe_vals = sub['Pearson_delta_full'].dropna().values
        if len(de_vals) > 0:
            de_boot = np.array([rng.choice(de_vals, size=len(de_vals), replace=True).mean() for _ in range(2000)])
            de_lo, de_hi = np.percentile(de_boot, 2.5), np.percentile(de_boot, 97.5)
        else:
            de_lo = de_hi = float('nan')
        pe_boot = np.array([rng.choice(pe_vals, size=len(pe_vals), replace=True).mean() for _ in range(2000)])
        rows.append({
            "mode": mode,
            "n_obs": len(de_vals),
            "DE_Spearman_mean": float(de_vals.mean()) if len(de_vals) else float('nan'),
            "DE_Spearman_ci_lo": de_lo, "DE_Spearman_ci_hi": de_hi,
            "Pearson_full_mean": float(pe_vals.mean()),
            "Pearson_full_ci_lo": float(np.percentile(pe_boot, 2.5)),
            "Pearson_full_ci_hi": float(np.percentile(pe_boot, 97.5)),
        })
    sdf = pd.DataFrame(rows).round(4).sort_values("DE_Spearman_mean", ascending=False)
    sdf.to_csv(RES / "audit_gears_norman_e5_allseeds_summary.csv", index=False)
    print(sdf.to_string(index=False))

    # Paired gap (learned − pop_mean) per (condition × seed)
    print("\n=== Paired gap (learned − pop_mean) ===")
    piv_de = big.pivot_table(index=['condition', 'seed'], columns='mode', values='DE_Spearman').dropna(subset=['learned', 'pop_mean'])
    gap_de = piv_de['learned'] - piv_de['pop_mean']
    boot_de = np.array([rng.choice(gap_de.values, size=len(gap_de), replace=True).mean() for _ in range(2000)])
    print(f"  DE-Spearman: n_paired_obs={len(gap_de)}, mean={gap_de.mean():+.4f}, 95% CI [{np.percentile(boot_de, 2.5):+.4f}, {np.percentile(boot_de, 97.5):+.4f}]")
    print(f"    Cohen's d (paired) = {gap_de.mean() / gap_de.std(ddof=1):+.3f}")

    piv_pe = big.pivot_table(index=['condition', 'seed'], columns='mode', values='Pearson_delta_full').dropna(subset=['learned', 'pop_mean'])
    gap_pe = piv_pe['learned'] - piv_pe['pop_mean']
    boot_pe = np.array([rng.choice(gap_pe.values, size=len(gap_pe), replace=True).mean() for _ in range(2000)])
    print(f"  Pearson δ (full): n_paired_obs={len(gap_pe)}, mean={gap_pe.mean():+.4f}, 95% CI [{np.percentile(boot_pe, 2.5):+.4f}, {np.percentile(boot_pe, 97.5):+.4f}]")
    print(f"    Cohen's d (paired) = {gap_pe.mean() / gap_pe.std(ddof=1):+.3f}")

    # Cluster bootstrap by condition (paired across seeds within condition)
    print("\n  --- Cluster bootstrap by condition ---")
    conds = sorted(piv_de.index.get_level_values('condition').unique())
    gap_per_cond_de = piv_de.groupby(level='condition').apply(lambda g: (g['learned'] - g['pop_mean']).mean())
    boot_c_de = np.array([rng.choice(gap_per_cond_de.values, size=len(gap_per_cond_de), replace=True).mean() for _ in range(2000)])
    print(f"  DE-Spearman cluster (n={len(gap_per_cond_de)} unique conditions): mean={gap_per_cond_de.mean():+.4f}, 95% CI [{np.percentile(boot_c_de, 2.5):+.4f}, {np.percentile(boot_c_de, 97.5):+.4f}]")
    gap_per_cond_pe = piv_pe.groupby(level='condition').apply(lambda g: (g['learned'] - g['pop_mean']).mean())
    boot_c_pe = np.array([rng.choice(gap_per_cond_pe.values, size=len(gap_per_cond_pe), replace=True).mean() for _ in range(2000)])
    print(f"  Pearson δ cluster (n={len(gap_per_cond_pe)} unique conditions): mean={gap_per_cond_pe.mean():+.4f}, 95% CI [{np.percentile(boot_c_pe, 2.5):+.4f}, {np.percentile(boot_c_pe, 97.5):+.4f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fetch":
        fetch_modal()
    aggregate()
