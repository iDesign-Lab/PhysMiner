"""
Pipeline scheduler: runs wordcloud_papers.py for highlyCited, overall, and recent
in sequence. All outputs go to ../results/.
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    os.path.join(BASE_DIR, "highlyCited", "wordcloud_papers.py"),
    os.path.join(BASE_DIR, "overall",      "wordcloud_papers.py"),
    os.path.join(BASE_DIR, "recent",       "wordcloud_papers.py"),
]


def run_script(script_path: str) -> bool:
    label = os.path.basename(os.path.dirname(script_path))
    print(f"\n{'='*60}")
    print(f"  Running: {label}/wordcloud_papers.py")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"[ERROR] {label} exited with code {result.returncode}")
        return False
    return True


def main():
    print("Pipeline scheduler started.")
    failed = []
    for script in SCRIPTS:
        if not run_script(script):
            failed.append(script)

    print(f"\n{'='*60}")
    if failed:
        print(f"Pipeline finished with {len(failed)} error(s):")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("Pipeline finished successfully. All word clouds saved to ../results/")


if __name__ == "__main__":
    main()
