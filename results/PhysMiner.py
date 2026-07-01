import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
CASE_DIR    = RESULTS_DIR.parent

LIT_TRACK   = CASE_DIR / "literature-driven track"
DATA_TRACK  = CASE_DIR / "data-driven track"
TD_LIB      = CASE_DIR / "triple decomposition library"
COMMENTS_MD = RESULTS_DIR / "comments.md"

MAX_RETRY = 10   # guard against infinite loop

# ── helpers ──────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    line = "=" * 62
    print(f"\n{line}\n  {text}\n{line}")

def run(step_name: str, cmd: list, cwd: Path) -> bool:
    banner(step_name)
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} exited with code {result.returncode}. Pipeline stopped.")
        return False
    print(f"\n[OK] {step_name}")
    return True

def read_verdict() -> str:
    if not COMMENTS_MD.exists():
        return "missing"
    first_line = COMMENTS_MD.read_text(encoding="utf-8").splitlines()[0].strip().lower()
    return first_line

# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Step 1: literature-driven track ──────────────────────────────────────
    ok = run(
        "Step 1 — Literature-driven track",
        [sys.executable, str(LIT_TRACK / "literature-driven track.py")],
        cwd=LIT_TRACK,
    )
    if not ok:
        sys.exit(1)

    # ── Step 2: data-driven track ─────────────────────────────────────────────
    ok = run(
        "Step 2 — Data-driven track",
        [sys.executable, str(DATA_TRACK / "data-driven track.py")],
        cwd=DATA_TRACK,
    )
    if not ok:
        sys.exit(1)

    # ── Step 3: triple decomposition library ─────────────────────────────────
    ok = run(
        "Step 3 — Triple decomposition library",
        [sys.executable, str(TD_LIB / "triple decomposition library.py")],
        cwd=TD_LIB,
    )
    if not ok:
        sys.exit(1)

    # ── Steps 4-5-6 loop ─────────────────────────────────────────────────────
    for attempt in range(1, MAX_RETRY + 1):
        loop_label = f"(attempt {attempt})" if attempt > 1 else ""

        # Step 4: discover-physics skill
        ok = run(
            f"Step 4 — discover-physics agent skill {loop_label}",
            ["claude", "-p", "/discover-physics", "--dangerously-skip-permissions"],
            cwd=RESULTS_DIR,
        )
        if not ok:
            sys.exit(1)

        # Step 5: review-report skill
        ok = run(
            f"Step 5 — review-report agent skill {loop_label}",
            ["claude", "-p", "/review-report", "--dangerously-skip-permissions"],
            cwd=RESULTS_DIR,
        )
        if not ok:
            sys.exit(1)

        # Step 6: read comments.md verdict
        banner(f"Step 6 — Read comments.md verdict {loop_label}")
        verdict = read_verdict()
        print(f"  Verdict: {verdict!r}")

        if verdict == "pass":
            print("\n[OK] All review criteria passed. Pipeline complete.")
            break
        elif verdict == "fail":
            if attempt == MAX_RETRY:
                print(f"\n[STOPPED] Reached maximum retry limit ({MAX_RETRY}). "
                      "Review still failing after repeated attempts.")
                sys.exit(1)
            print(f"  Review failed — re-running Steps 4-5 (attempt {attempt + 1})...")
        else:
            print(f"\n[ERROR] Unexpected verdict {verdict!r} in comments.md. Pipeline stopped.")
            sys.exit(1)

    banner("Pipeline finished")


if __name__ == "__main__":
    main()
