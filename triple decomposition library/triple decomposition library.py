import subprocess
import sys
from pathlib import Path

CWD = Path(__file__).parent

STEPS = [
    {
        "name": "Step 1 — Copy triple-decomp images",
        "cmd": [sys.executable, str(CWD / "copy_triple_decomp_images.py")],
    },
    {
        "name": "Step 2 — Flow fingerprint (skill)",
        "cmd": ["claude", "-p", "/flow-fingerprint", "--dangerously-skip-permissions"],
    },
    {
        "name": "Step 3 — Jaccard matching",
        "cmd": [sys.executable, str(CWD / "jaccard_match.py")],
    },
    {
        "name": "Step 4 — Flow comparison (skill)",
        "cmd": ["claude", "-p", "/flow-comparison", "--dangerously-skip-permissions"],
    },
    {
        "name": "Step 5 — Save results to knowledge & results/",
        "cmd": [sys.executable, str(CWD / "save_results.py")],
    },
]

def run_step(step: dict) -> bool:
    print(f"\n{'='*60}")
    print(f"  {step['name']}")
    print(f"{'='*60}")
    result = subprocess.run(
        step["cmd"],
        cwd=str(CWD),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"\n[FAILED] {step['name']} exited with code {result.returncode}. Pipeline stopped.")
        return False
    print(f"\n[OK] {step['name']}")
    return True

if __name__ == "__main__":
    for step in STEPS:
        if not run_step(step):
            sys.exit(1)
    print(f"\n{'='*60}")
    print("  All steps completed successfully.")
    print(f"{'='*60}")
