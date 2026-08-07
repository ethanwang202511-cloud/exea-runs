"""Replogle vocabulary-size sweep: tests the unified mechanism for Norman/Replogle inversion.

For each N ∈ {100, 500, 2000} train perturbations, train minimal CPA at default
config (1.27 M params), 20 epochs. Run the same 6-mode audit on a fixed-across-N
test set of 50 held-out genes.

Hypothesis: gap (learned − pop_mean) declines with N. At N=100 (≈Norman scale),
gap should be positive or near zero. At N=2000 (Replogle scale), gap should be
strongly negative, matching the 3-seed Replogle result.
"""
import subprocess, sys
N_VALUES = [100, 500, 2000]
SEEDS = [0]  # single seed for the sweep; can extend if results are interesting
EPOCHS = 20
for n_perts in N_VALUES:
    for seed in SEEDS:
        cmd = [
            sys.executable, "scripts/train_audit_replogle_vocab.py",
            "--n_train_perts", str(n_perts),
            "--max_test_genes", "50",
            "--epochs", str(EPOCHS),
            "--max_cells", "80000",
            "--seed", str(seed),
            "--out_tag", "replogle_vocab",
        ]
        print(f"\n=== n_train_perts={n_perts} seed={seed} ===")
        rc = subprocess.run(cmd).returncode
        print(f"  rc={rc}")
