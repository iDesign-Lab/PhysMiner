import ast
import shutil
from pathlib import Path

current_dir = Path(__file__).parent
temp_dir = current_dir / "temp/"
knowledge_dir = current_dir / "knowledge"

# Read target fingerprint (first line of temp/fingerprint/fingerprint.txt)
target_file = temp_dir / "fingerprint/fingerprint.txt"
target_set = set(ast.literal_eval(target_file.read_text().splitlines()[0]))

# Compute Jaccard tree distance for every knowledge/fingerprint_*.txt
best_dist = float("inf")
best_file = None
results = []

for fp in sorted(knowledge_dir.glob("fingerprint_*/fingerprint_*.txt")):
    hist_set = set(ast.literal_eval(fp.read_text().splitlines()[0]))
    intersection = len(target_set & hist_set)
    union = len(target_set | hist_set)
    dist = 1 - intersection / union if union > 0 else 0.0
    results.append((fp, dist, intersection, union))
    if dist < best_dist:
        best_dist = dist
        best_file = fp

# Print results table
print(f"Target fingerprint: {sorted(target_set)}\n")
print(f"{'File':<30}  {'Intersection':>12}  {'Union':>5}  {'D_tree':>8}")
print("-" * 62)
for fp, dist, inter, union in results:
    marker = " <-- best" if fp == best_file else ""
    print(f"{fp.name:<30}  {inter:>12}  {union:>5}  {dist:>8.4f}{marker}")

# Copy the best-matching folder to temp/
best_folder = best_file.parent          # e.g. knowledge/fingerprint_1
dest = temp_dir / best_folder.name      # e.g. temp/fingerprint_1
if dest.exists():
    shutil.rmtree(dest)
shutil.copytree(best_folder, dest)
print(f"\nCopied folder '{best_folder.name}' to temp/ (D_tree = {best_dist:.4f})")
