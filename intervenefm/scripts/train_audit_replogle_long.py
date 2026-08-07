"""Run Replogle audit with longer training to address 'undertrained' concern.

40 epochs vs the original 12. Single seed first; if the inversion (learned <
pop_mean) holds, the finding is robust. If it closes, we report both and
discuss training-budget sensitivity.
"""
import subprocess, sys
for seed in [0]:
    cmd = [
        sys.executable, "scripts/train_audit_replogle.py",
        "--seed", str(seed), "--epochs", "40",
        "--max_cells", "80000",
        "--out_tag", f"replogle_long_seed{seed}",
    ]
    print(f"\n=== Replogle long seed={seed} ===")
    rc = subprocess.run(cmd).returncode
    print(f"  rc={rc}")
