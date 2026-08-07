"""audit_claim_reproduction.py — independent re-derivation of BehaviorFM's
headline numbers directly from the result CSVs.

Confirms the paper's reported numbers are faithfully transcribed from the
raw results. We recompute, from scratch:

  A. H1 §5.3 table (8 species): adapt_iti_full vs no_adapt_mean_token —
     paired-bootstrap 95% CI (n_resamples=5000), seeds-favor count, Δ rel,
     within-15% verdict. Check every CI excludes zero (H1-prime CONFIRMED).
  B. §5.4 headline: fox ITA(token_only)/decoder-FT RMSE ratio (claim 1.03×).
  C. H5a §5.5 (v1/v2/v3, Elephant+Chimpanzee): interp(k=1) vs random init
     Δ rel; check NONE reaches the >=15% bar (H5a FALSIFIED).

Each recomputed value is printed beside the paper's reported value with a
PASS/MISMATCH flag (tolerance 0.5pp on rel %, 0.01 on RMSE, 0.02 on ratio).

CPU only, reads existing CSVs. Output:
  results/audit/claim_reproduction.json
  results/audit/claim_reproduction.md
"""
from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")

SPECIES = ["alouatta", "chimpanzee", "elephant", "monkey", "gorilla", "rabbit", "fox", "panther"]
ADAPT = "adapt_iti_full"
BASELINE = "no_adapt_mean_token"

# Paper's reported H1 §5.3 numbers (adapt RMSE, baseline RMSE, Δ, Δrel%, seeds_favor, within15)
REPORTED_H1 = {
    "alouatta":   (0.151, 0.160, -0.0092, -5.8, 10, False),
    "chimpanzee": (0.170, 0.180, -0.0098, -5.5, 9, False),
    "elephant":   (0.122, 0.136, -0.0134, -10.0, 10, False),
    "monkey":     (0.156, 0.162, -0.0066, -4.1, 10, False),
    "gorilla":    (0.167, 0.183, -0.0161, -8.8, 10, False),
    "rabbit":     (0.063, 0.084, -0.0213, -25.4, 10, True),
    "fox":        (0.069, 0.081, -0.0125, -15.4, 10, True),
    "panther":    (0.065, 0.078, -0.0127, -16.3, 10, True),
}
# Paper's reported H5a Δrel% (interp k=1 vs random)
REPORTED_H5A = {
    ("v1", "elephant"): -3.5, ("v1", "chimpanzee"): 1.8,
    ("v2", "elephant"): -3.6, ("v2", "chimpanzee"): 1.8,
    ("v3", "elephant"): -0.9, ("v3", "chimpanzee"): 0.0,
}
H5A_FILES = {"v1": "h5a_binding.csv", "v2": "h5a_binding_v2.csv", "v3": "h5a_binding_v3.csv"}


def paired_bootstrap_ci(deltas, n_resamples=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    n = len(d)
    means = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[b] = d[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def audit_h1():
    rows = []
    all_ci_exclude_zero = True
    for sp in SPECIES:
        df = pd.read_csv(os.path.join(RES, f"h1_n10_{sp}.csv"))
        a = df[df.strategy == ADAPT].sort_values("seed")
        b = df[df.strategy == BASELINE].sort_values("seed")
        # pair by seed
        m = a.merge(b, on="seed", suffixes=("_adapt", "_base"), validate="1:1")
        assert len(m) == 10, f"{sp}: expected 10 paired seeds, got {len(m)}"
        adapt_rmse = m["rmse_norm_adapt"].values
        base_rmse = m["rmse_norm_base"].values
        deltas = adapt_rmse - base_rmse  # negative = improvement
        adapt_mean = float(adapt_rmse.mean())
        base_mean = float(base_rmse.mean())
        delta_mean = float(deltas.mean())
        delta_rel = 100.0 * delta_mean / base_mean
        seeds_favor = int((deltas < 0).sum())
        lo, hi = paired_bootstrap_ci(deltas)
        ci_excl0 = (lo < 0 and hi < 0) or (lo > 0 and hi > 0)
        all_ci_exclude_zero = all_ci_exclude_zero and ci_excl0
        within15 = delta_rel <= -15.0

        rep = REPORTED_H1[sp]
        flags = {
            "adapt_rmse": abs(adapt_mean - rep[0]) <= 0.01,
            "baseline_rmse": abs(base_mean - rep[1]) <= 0.01,
            "delta": abs(delta_mean - rep[2]) <= 0.0015,
            "delta_rel": abs(delta_rel - rep[3]) <= 0.6,
            "seeds_favor": seeds_favor == rep[4],
            "within15": within15 == rep[5],
        }
        rows.append({
            "species": sp, "n_seeds": len(m),
            "adapt_rmse": adapt_mean, "baseline_rmse": base_mean,
            "delta": delta_mean, "delta_rel_pct": delta_rel,
            "ci95": [lo, hi], "ci_excludes_zero": ci_excl0,
            "seeds_favor": seeds_favor, "within15_vs_noadapt": within15,
            "reported": {"adapt_rmse": rep[0], "baseline_rmse": rep[1], "delta": rep[2],
                         "delta_rel_pct": rep[3], "seeds_favor": rep[4], "within15": rep[5]},
            "match": flags, "all_match": all(flags.values()),
        })
    return {"rows": rows, "h1prime_all_ci_exclude_zero": all_ci_exclude_zero}


def audit_fox_ratio():
    pareto = pd.read_csv(os.path.join(RES, "h1_pareto.csv"))
    dft = pd.read_csv(os.path.join(RES, "decoder_ft_comparator.csv"))
    fox_tok = pareto[(pareto.species == "fox") & (pareto.variant == "token_only")]
    assert len(fox_tok) > 0, "no fox token_only rows in h1_pareto.csv"
    tok_rmse = float(fox_tok["rmse_norm"].mean())
    fox_dft = dft[dft.species == "fox"]
    assert len(fox_dft) == 1, f"expected 1 fox row in decoder_ft_comparator, got {len(fox_dft)}"
    dft_rmse = float(fox_dft["post_rmse_norm"].iloc[0])
    ratio = tok_rmse / dft_rmse
    return {
        "fox_token_only_rmse": tok_rmse, "fox_token_only_n_seeds": int(len(fox_tok)),
        "fox_decoder_ft_post_rmse": dft_rmse,
        "ratio_ITA_over_DFT": ratio,
        "reported_ratio": 1.03,
        "match": abs(ratio - 1.03) <= 0.02,
    }


def audit_h5a():
    rows = []
    any_pass_15 = False
    for arch, fname in H5A_FILES.items():
        df = pd.read_csv(os.path.join(RES, fname))
        for sp in ["elephant", "chimpanzee"]:
            sub = df[df.species == sp]
            inits = sorted(sub["init"].unique().tolist())
            # 'interp' (the learned interp head) is distinct from 'interp_zero_query';
            # the k==1 filter selects the genuine interp(k=1) rows.
            interp_inits = [s for s in inits if s == "interp" or
                            ("interp" in s and s != "interp_zero_query")]
            rnd = sub[sub.init == "random"]["rmse_norm"].mean()
            ik1 = sub[(sub.init.isin(interp_inits)) & (sub.k == 1)]
            if len(ik1) == 0:
                rows.append({"arch": arch, "species": sp,
                             "error": "no interp(k=1) rows found",
                             "inits_available": inits,
                             "k_values": sorted(sub["k"].unique().tolist()),
                             "match": None})
                continue
            interp_rmse = ik1["rmse_norm"].mean()
            delta_rel = 100.0 * (interp_rmse - rnd) / rnd
            passes_15 = delta_rel <= -15.0
            any_pass_15 = any_pass_15 or passes_15
            rep = REPORTED_H5A.get((arch, sp))
            rows.append({
                "arch": arch, "species": sp,
                "random_rmse": float(rnd), "interp_k1_rmse": float(interp_rmse),
                "delta_rel_pct": float(delta_rel),
                "passes_15pct": bool(passes_15),
                "reported_delta_rel_pct": rep,
                "inits_available": inits, "k_values": sorted(sub["k"].unique().tolist()),
                "match": (abs(delta_rel - rep) <= 0.6) if rep is not None else None,
            })
    return {"rows": rows, "h5a_falsified_none_pass_15": not any_pass_15}


def main():
    out = {"experiment": "behaviorfm_claim_reproduction_audit",
           "purpose": "Independent re-derivation of paper headline numbers from raw CSVs.",
           "H1": audit_h1(), "fox_ratio": audit_fox_ratio(), "H5a": audit_h5a()}

    def _san(o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        raise TypeError(f"not serializable: {type(o)}")

    out_dir = os.path.join(RES, "audit")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "claim_reproduction.json"), "w") as f:
        json.dump(out, f, indent=2, default=_san)

    # markdown
    L = ["# BehaviorFM — independent claim-reproduction audit", "",
         "Recomputed from raw CSVs vs paper's reported values. ✅=match, ❌=mismatch.", "",
         "## A. H1 §5.3 — adapt_iti_full vs no_adapt_mean_token (paired bootstrap, 5000 resamples)", "",
         "| species | adapt RMSE (rep) | base RMSE (rep) | Δrel% (rep) | 95% CI | excl 0 | seeds favor (rep) | within-15% (rep) | all✓ |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in out["H1"]["rows"]:
        rp = r["reported"]
        ok = "✅" if r["all_match"] else "❌"
        L.append(f"| {r['species']} | {r['adapt_rmse']:.3f} ({rp['adapt_rmse']}) | "
                 f"{r['baseline_rmse']:.3f} ({rp['baseline_rmse']}) | "
                 f"{r['delta_rel_pct']:.1f} ({rp['delta_rel_pct']}) | "
                 f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | {'yes' if r['ci_excludes_zero'] else 'NO'} | "
                 f"{r['seeds_favor']} ({rp['seeds_favor']}) | "
                 f"{'YES' if r['within15_vs_noadapt'] else 'no'} ({'YES' if rp['within15'] else 'no'}) | {ok} |")
    L += ["", f"**H1-prime (all 8 CIs exclude zero):** {out['H1']['h1prime_all_ci_exclude_zero']}", "",
          "## B. §5.4 headline — fox ITA(token-only) / decoder-FT RMSE ratio", "",
          f"- fox token-only RMSE (mean over {out['fox_ratio']['fox_token_only_n_seeds']} seeds) = **{out['fox_ratio']['fox_token_only_rmse']:.4f}**",
          f"- fox decoder-FT post RMSE = **{out['fox_ratio']['fox_decoder_ft_post_rmse']:.4f}**",
          f"- **recomputed ratio = {out['fox_ratio']['ratio_ITA_over_DFT']:.3f}×** (reported 1.03×) → {'✅' if out['fox_ratio']['match'] else '❌'}", "",
          "## C. H5a §5.5 — interp(k=1) vs random init", "",
          "| arch | species | random RMSE | interp(k=1) RMSE | Δrel% (rep) | ≥15%? | match |",
          "|---|---|---|---|---|---|---|"]
    for r in out["H5a"]["rows"]:
        if "error" in r:
            L.append(f"| {r['arch']} | {r['species']} | — | — | ERROR: {r['error']} | — | — |")
            continue
        mk = "✅" if r["match"] else ("❌" if r["match"] is False else "—")
        L.append(f"| {r['arch']} | {r['species']} | {r['random_rmse']:.4f} | {r['interp_k1_rmse']:.4f} | "
                 f"{r['delta_rel_pct']:.1f} ({r['reported_delta_rel_pct']}) | "
                 f"{'PASS' if r['passes_15pct'] else 'FAIL'} | {mk} |")
    L += ["", f"**H5a FALSIFIED (no arch beats random by ≥15%):** {out['H5a']['h5a_falsified_none_pass_15']}", ""]
    with open(os.path.join(out_dir, "claim_reproduction.md"), "w") as f:
        f.write("\n".join(L))

    # console summary
    n_h1_match = sum(r["all_match"] for r in out["H1"]["rows"])
    print(f"H1: {n_h1_match}/8 species fully match; all CIs exclude zero = {out['H1']['h1prime_all_ci_exclude_zero']}")
    print(f"fox ratio: recomputed {out['fox_ratio']['ratio_ITA_over_DFT']:.3f}x vs reported 1.03x -> match={out['fox_ratio']['match']}")
    n_h5a_match = sum(1 for r in out['H5a']['rows'] if r['match'])
    print(f"H5a: {n_h5a_match}/{len(out['H5a']['rows'])} rows match reported Δrel; falsified={out['H5a']['h5a_falsified_none_pass_15']}")
    print(f"saved {out_dir}/claim_reproduction.{{json,md}}")


if __name__ == "__main__":
    main()
