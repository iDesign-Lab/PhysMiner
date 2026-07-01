"""
profile_cauchy_stokes_yDirection.py

For each x in SLICE_X_LIST:
  1. Slice the mesh at that x -> the result IS exactly the fluid cross-section.
  2. Fetch the raw point data from the slice.
  3. Read Cauchy-Stokes fields (WW_over_GG, SS_over_GG) directly from the case.
  4. Group all points by their (rounded) y-coordinate -> each group spans all z.
  5. Average the field values within each y-group -> z-arithmetic average.
  6. Plot the two Cauchy-Stokes profiles (WW_over_GG, SS_over_GG) vs y.

No ResampleToImage, no hill-surface detection, no artefacts.

Run with pvpython:
    "D:/software/paraView/bin/pvpython.exe" profile_cauchy_stokes_yDirection.py
"""

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Use Times New Roman if available, otherwise fall back to Times
_available = {f.name for f in _fm.fontManager.ttflist}
_font_family = "Times New Roman" if "Times New Roman" in _available else "Times"
plt.rcParams.update({
    "font.family":       _font_family,
    "mathtext.fontset":  "stix",
})

# ── paths ──────────────────────────────────────────────────────────────────
script_dir  = Path(__file__).parent.resolve()
case_dir    = (script_dir / "../../").resolve()
foam_file   = case_dir / "para.foam"
results_dir = (script_dir / "../results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)
output_png  = results_dir / "profile_cauchy_stokes_yDirection.png"

print(f"Case   : {case_dir}")
print(f"Output : {output_png}")

# ── user parameters ────────────────────────────────────────────────────────
SLICE_X_LIST = [0.036, 0.071, 0.107, 0.143, 0.179, 0.214]

# x-axis (component value) limits — None = auto
XLIM = (0.0, 1.0)

# Precision (decimal places) used when grouping points into y-levels.
Y_ROUND = 5

# Output image
IMAGE_DPI    = 300
IMAGE_WIDTH  = 9
IMAGE_HEIGHT = 4

# ── field definitions ──────────────────────────────────────────────────────
FIELD_NAMES = ["SS_over_GG", "WW_over_GG"]
FIELD_STYLE = {
    "SS_over_GG": dict(color="#d62728", lw=1.8, ls="--", label=r"$gg_{ss}$  Strain-rate"),
    "WW_over_GG": dict(color="#1f77b4", lw=1.8, ls="-",  label=r"$gg_{ww}$  Vorticity"),
}

# ── ParaView pipeline ──────────────────────────────────────────────────────
from paraview.simple import OpenFOAMReader, CellDatatoPointData, Slice
import paraview.simple as pvs
from paraview.servermanager import Fetch
from vtk.util.numpy_support import vtk_to_numpy

pvs._DisableFirstRenderCameraReset()


def unwrap(raw):
    """Return the first non-empty block from a (possibly multi-block) Fetch."""
    if hasattr(raw, "GetNumberOfBlocks"):
        for bi in range(raw.GetNumberOfBlocks()):
            blk = raw.GetBlock(bi)
            if blk is not None and blk.GetNumberOfPoints() > 0:
                return blk
        raise RuntimeError("Fetch returned an empty MultiBlockDataSet.")
    return raw


print("\nOpening OpenFOAM case ...")
reader = OpenFOAMReader(FileName=str(foam_file))
reader.MeshRegions = ["internalMesh"]
reader.UpdatePipelineInformation()
reader.UpdatePipeline()

times = reader.TimestepValues
if not times:
    raise RuntimeError("No time steps found.")
last_time = times[-1]
print(f"  Last time step: {last_time}")
pvs.GetAnimationScene().AnimationTime = last_time
reader.UpdatePipeline(last_time)

print("CellDatatoPointData ...")
c2p = CellDatatoPointData(Input=reader)
c2p.UpdatePipeline(last_time)

# ── extract profiles ───────────────────────────────────────────────────────
all_profiles: list[dict] = []

for slice_x in SLICE_X_LIST:
    print(f"\nSlice at x = {slice_x:.6f} ...")

    slc = Slice(Input=c2p)
    slc.SliceType        = "Plane"
    slc.SliceType.Origin = [slice_x, 0.0, 0.0]
    slc.SliceType.Normal = [1.0, 0.0, 0.0]
    slc.Triangulatetheslice = 0
    slc.UpdatePipeline(last_time)

    data   = unwrap(Fetch(slc))
    pts    = data.GetPoints()
    n      = data.GetNumberOfPoints()
    pd_slc = data.GetPointData()

    y_pts = np.array([pts.GetPoint(i)[1] for i in range(n)])

    y_r = np.round(y_pts, Y_ROUND)
    y_unique, y_inv = np.unique(y_r, return_inverse=True)
    counts = np.bincount(y_inv).astype(float)

    print(f"  {n} slice points  →  {len(y_unique)} y-levels  "
          f"y=[{y_unique[0]:.4f}, {y_unique[-1]:.4f}]")

    entry: dict = {"x": slice_x, "y": y_unique}
    for name in FIELD_NAMES:
        arr = pd_slc.GetArray(name)
        if arr is None:
            raise KeyError(f"'{name}' not found in slice at x={slice_x:.4f}")
        vals        = vtk_to_numpy(arr).astype(float)
        sums        = np.bincount(y_inv, weights=vals).astype(float)
        entry[name] = sums / counts
        vmin, vmax  = entry[name].min(), entry[name].max()
        print(f"  {name}: z-avg [{vmin:.4f}, {vmax:.4f}]")

    all_profiles.append(entry)

    pvs.Delete(slc)
    del slc

# ── figure ─────────────────────────────────────────────────────────────────
print("\nBuilding figure ...")
n     = len(all_profiles)
ncols = min(n, 3)
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(IMAGE_WIDTH, IMAGE_HEIGHT * nrows),
    squeeze=False,
    sharey=True,
)
fig.patch.set_facecolor("white")
axes_flat = axes.ravel()

for i, entry in enumerate(all_profiles):
    ax = axes_flat[i]
    y  = entry["y"]
    for name in FIELD_NAMES:
        ax.plot(entry[name], y, **FIELD_STYLE[name])
    ax.set_xlabel("Component value", fontsize=20)
    ax.set_ylabel("$y$", fontsize=20)
    ax.set_title(f"$x = {entry['x']:.3f}$", fontsize=20)
    if XLIM is not None:
        ax.set_xlim(XLIM)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=20)

for ax in axes_flat[n:]:
    ax.set_visible(False)

handles = [plt.Line2D([0], [0], **{k: v for k, v in FIELD_STYLE[nm].items()})
           for nm in FIELD_NAMES]
labels  = [FIELD_STYLE[nm]["label"] for nm in FIELD_NAMES]

fig.tight_layout(rect=[0, 0, 1, 0.85])
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97),
           ncol=2, fontsize=20, framealpha=0.85)
fig.savefig(str(output_png), dpi=IMAGE_DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved -> {output_png}")
print("Done.")
