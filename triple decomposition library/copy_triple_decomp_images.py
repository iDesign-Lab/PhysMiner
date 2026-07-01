import os
import shutil
from pathlib import Path

current_dir = Path(__file__).parent
results_dir = current_dir.parent / "results"
tem_dir = current_dir / "temp" / "fingerprint"

tem_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".svg"}

copied = []
for f in results_dir.iterdir():
    if f.is_file() and "triple_decomp" in f.name and f.suffix.lower() in image_extensions:
        dest = tem_dir / f.name
        shutil.copy2(f, dest)
        copied.append(f.name)

if copied:
    print(f"Copied {len(copied)} image(s) to {tem_dir}:")
    for name in copied:
        print(f"  {name}")
else:
    print("No image files containing 'triple_decomp' were found.")
