"""Sequentially run extras after capacity sweep ends:
  1. Wait for cap sweep to finish (no procs)
  2. Run orthogonal scaling
  3. Run Replogle long-trained
"""
import subprocess, sys, time, os

def wait_no_procs(name_substr, max_wait_s=1500):
    start = time.time()
    while True:
        out = subprocess.run(["pgrep", "-f", name_substr], capture_output=True).stdout.decode()
        if not out.strip():
            return True
        if time.time() - start > max_wait_s:
            print(f"timeout waiting for {name_substr}")
            return False
        time.sleep(30)

print("=== waiting for capacity 3-seed sweep to finish ===")
wait_no_procs("run_capacity_3seed.py")

print("\n=== orthogonal scaling sweep ===")
subprocess.run([sys.executable, "scripts/orthogonal_scaling.py"], check=False)

print("\n=== Replogle long (40 epochs) ===")
subprocess.run([sys.executable, "scripts/train_audit_replogle_long.py"], check=False)

print("\n=== ALL DONE ===")
