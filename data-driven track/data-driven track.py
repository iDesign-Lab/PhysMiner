#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline Scheduler — runs all data-processing scripts in sequence.
"""

import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

# Path to pvpython (ParaView's embedded Python interpreter).
# Adjust this to match your ParaView installation if it differs.
PVPYTHON = Path("D:/software/paraView/bin/pvpython.exe")

# Scripts that import paraview.simple and must run under pvpython.
PVPYTHON_SCRIPTS = {
    "profile_cauchy_stokes_yDirection.py",
    "profile_triple_decomp_yDirection.py",
    "slice_cauchy_stokes_decomposition_zDirection.py",
    "slice_streamwise_velocity_zDirection.py",
    "slice_triple_decomposition_zDirection.py",
    "triple_decomposition_donut_chart.py",
}

SCRIPTS_TO_RUN = [
    "compute_all_derived_fields.py",
    "profile_cauchy_stokes_yDirection.py",
    "profile_triple_decomp_yDirection.py",
    "slice_cauchy_stokes_decomposition_zDirection.py",
    "slice_streamwise_velocity_zDirection.py",
    "slice_triple_decomposition_zDirection.py",
    "triple_decomposition_donut_chart.py",
]


def run_script(script_name):
    """Run a single script with the appropriate interpreter. Returns True on success."""
    script_path = CURRENT_DIR / script_name

    if not script_path.exists():
        print(f"ERROR: script not found — {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}\n")

    if script_name in PVPYTHON_SCRIPTS:
        if not PVPYTHON.exists():
            print(f"ERROR: pvpython not found at {PVPYTHON}")
            print("       Update PVPYTHON at the top of pipeline_scheduler.py.")
            return False
        interpreter = str(PVPYTHON)
    else:
        interpreter = sys.executable

    try:
        subprocess.run(
            [interpreter, str(script_path)],
            cwd=str(CURRENT_DIR),
            check=True,
        )
        print(f"\nOK: {script_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFAILED: {script_name}  (exit code {e.returncode})\n")
        return False
    except Exception as e:
        print(f"\nERROR in {script_name}: {e}\n")
        return False


def main():
    """Run all scripts in order; stop on first failure."""
    print("\n" + "="*60)
    print("Pipeline Scheduler — starting")
    print("="*60)

    total = len(SCRIPTS_TO_RUN)
    passed = 0

    for idx, name in enumerate(SCRIPTS_TO_RUN, 1):
        print(f"\n[{idx}/{total}] {name}")
        if run_script(name):
            passed += 1
        else:
            print(f"\nPipeline stopped at '{name}'")
            break

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total scripts : {total}")
    print(f"Succeeded     : {passed}")
    print(f"Failed/stopped: {total - passed}")

    if passed == total:
        print("\nAll scripts completed successfully.\n")
        return 0
    else:
        print("\nPipeline interrupted — check the error output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
