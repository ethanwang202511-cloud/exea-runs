"""Pull GEARS audit results from the Modal volume to local results/."""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

# Modal volume CLI: modal volume get <volume> <remote_path> <local_path>
files_to_get = [
    "results/audit_summary_gears_norman_e5_seed1.csv",
    "results/audit_gears_norman_e5_seed1.csv",
]
for remote in files_to_get:
    local = RES / Path(remote).name
    cmd = ["modal", "volume", "get", "intervenefm-data", remote, str(local), "--force"]
    print(f"[fetch] {remote} -> {local}")
    rc = subprocess.run(cmd).returncode
    print(f"  rc={rc}")
