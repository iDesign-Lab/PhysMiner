import re
import shutil
from pathlib import Path

current_dir = Path(__file__).parent
knowledge_dir = current_dir / "knowledge"
temp_dir = current_dir / "temp"

# ── 1. Copy temp/fingerprint/ → knowledge/fingerprint_N/ ──────────────────
src_fp = temp_dir / "fingerprint"
if not src_fp.exists():
    raise FileNotFoundError(f"Source folder not found: {src_fp}")

existing = [
    int(m.group(1))
    for d in knowledge_dir.iterdir()
    if d.is_dir() and (m := re.fullmatch(r"fingerprint_(\d+)", d.name))
]
next_n = max(existing, default=0) + 1
dest_fp = knowledge_dir / f"fingerprint_{next_n}"

shutil.copytree(src_fp, dest_fp)
print(f"Copied  temp/fingerprint  →  knowledge/fingerprint_{next_n}")

# ── 2. Copy temp/flow_comparison_table.md → ../results/ ───────────────────
src_md = temp_dir / "flow_comparison_table.md"
if not src_md.exists():
    raise FileNotFoundError(f"Source file not found: {src_md}")

results_dir = current_dir.parent / "results"
results_dir.mkdir(exist_ok=True)
shutil.copy2(src_md, results_dir / src_md.name)
print(f"Copied  temp/flow_comparison_table.md  →  {results_dir / src_md.name}")
