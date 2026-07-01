r"""
step_pv_slice_combined_binary_zDirection.py

Combined z = 0 slice showing both binary-decomposition components
(WW_over_GG and SS_over_GG) overlaid in a single figure.

Each component is rendered only where its value exceeds a per-variable
cut-off threshold; below-threshold (and out-of-domain) regions are
transparent, so the white background or a lower layer shows through.
Where two components are simultaneously above threshold the layer drawn
last (see LAYER_ORDER) appears on top.

Two horizontal colour bars are placed below the main plot.

Run with pvpython:
    "D:\software\paraView\bin\pvpython.exe" slice_cauchy_stokes_decomposition_zDirection.py

Tuning guide
────────────
  THRESH_*   : cut-off value for each component (set to ~10 % of max)
  CLIM_*     : colour-bar limits for each component
  LAYER_ORDER: rendering order — first = bottom, last = top in overlap
  GRID_NX/NY : resampling resolution (higher → sharper but slower)
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — must come before pyplot
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase

# ── paths ──────────────────────────────────────────────────────────────────
script_dir  = Path(__file__).parent.resolve()
case_dir    = (script_dir / "../../").resolve()
foam_file   = case_dir / "para.foam"
results_dir = (script_dir / "../results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)
output_png  = results_dir / "slice_cauchy_stokes_decomposition_zDirection.png"

print(f"Case   : {case_dir}")
print(f"Output : {output_png}")

# ── user parameters ────────────────────────────────────────────────────────
# Cut-off thresholds — pixels below these values are not shown
THRESH_WW = 0.50   # ← WW_over_GG threshold
THRESH_SS = 0.50   # ← SS_over_GG threshold

# Colour-bar limits (vmin, vmax) for each component
CLIM_WW = (0.50, 1.00)
CLIM_SS = (0.50, 1.00)

# Axis display range.  None = use full data extent from the mesh.
# Example: XLIM = (0.0, 20.0)  YLIM = (-1.0, 3.0)
XLIM = None             # (x_min, x_max) or None
# XLIM = (-0.1444, 0.76)             # (x_min, x_max) or None
YLIM = None             # (y_min, y_max) or None

# Resampling grid (number of pixels in x and y directions).
# 5000×5000 requires ~800 MB per RGBA layer (float64); 1500×1500 is
# sufficient for 300 DPI output and keeps peak memory under ~100 MB.
GRID_NX = 1500
GRID_NY = 1500

# Slice z-position: None = auto-detect (uses mesh z-centre)
# To fix the slice at a specific z position, replace None with a numeric value.
SLICE_Z = None
# SLICE_Z = 3.7044

# Output image
IMAGE_DPI    = 300
IMAGE_WIDTH  = 16     # figure width  in inches
IMAGE_HEIGHT = 9      # figure height in inches

# Rendering order: earlier entries are drawn first (background); later entries
# are drawn on top.  Change this if you want a different priority in overlaps.
LAYER_ORDER = ["SS_over_GG", "WW_over_GG"]

# Per-layer visual settings
LAYER_PARAMS = {
    "SS_over_GG": dict(cmap="YlOrRd", clim=CLIM_SS, thresh=THRESH_SS,
                       label=r"$gg_{ss}$  Strain-rate"),
    "WW_over_GG": dict(cmap="Blues",  clim=CLIM_WW, thresh=THRESH_WW,
                       label=r"$gg_{ww}$  Vorticity"),
}

# ── ParaView pipeline ──────────────────────────────────────────────────────
from paraview.simple import (
    OpenFOAMReader, CellDatatoPointData, Slice, ResampleToImage,
)
import paraview.simple as pvs
from paraview.servermanager import Fetch
from vtk.util.numpy_support import vtk_to_numpy

pvs._DisableFirstRenderCameraReset()

print("\nOpening OpenFOAM case …")
reader = OpenFOAMReader(FileName=str(foam_file))
reader.MeshRegions = ["internalMesh"]
reader.UpdatePipelineInformation()               # discover available arrays first
all_cell_arrays = list(reader.CellArrays)
print(f"  Available CellArrays ({len(all_cell_arrays)} total):")
for arr in all_cell_arrays:
    print(f"    '{arr}'")
# Load ALL arrays — do not restrict, to avoid silent name-mismatch drops
reader.UpdatePipeline()

times = reader.TimestepValues
if not times:
    raise RuntimeError("No time steps found.")
last_time = times[-1]
print(f"  Using last time step: {last_time}")
pvs.GetAnimationScene().AnimationTime = last_time
reader.UpdatePipeline(last_time)

print("CellDatatoPointData …")
c2p = CellDatatoPointData(Input=reader)
c2p.UpdatePipeline(last_time)

print("Computing velocity gradient …")
_grad_fn = (pvs.GradientOfUnstructuredDataSet
            if hasattr(pvs, "GradientOfUnstructuredDataSet")
            else pvs.Gradient)
grad_filter = _grad_fn(Input=c2p)
grad_filter.ScalarArray    = ["POINTS", "U"]
grad_filter.ResultArrayName = "gradU"
grad_filter.ComputeGradient  = 1
grad_filter.ComputeVorticity = 0
grad_filter.UpdatePipeline(last_time)

# Determine slice z-position from mesh bounds if not set by user
mesh_bounds = grad_filter.GetDataInformation().GetBounds()
z_lo, z_hi = mesh_bounds[4], mesh_bounds[5]
print(f"  Mesh z-extent: [{z_lo:.6f}, {z_hi:.6f}]")
slice_z = SLICE_Z if SLICE_Z is not None else 0.5 * (z_lo + z_hi)
print(f"Slice at z = {slice_z:.6f} …")
slc = Slice(Input=grad_filter)
slc.SliceType        = "Plane"
slc.SliceType.Origin = [0.0, 0.0, slice_z]
slc.SliceType.Normal = [0.0, 0.0, 1.0]
slc.Triangulatetheslice = 0
slc.UpdatePipeline(last_time)

bounds = slc.GetDataInformation().GetBounds()
x_min, x_max = bounds[0], bounds[1]
y_min, y_max = bounds[2], bounds[3]
print(f"  Bounds: x=[{x_min:.4f}, {x_max:.4f}]  y=[{y_min:.4f}, {y_max:.4f}]")

print(f"ResampleToImage ({GRID_NX} × {GRID_NY}) …")
EPS = 1e-6    # avoids zero-extent z dimension in VTK
rsi = ResampleToImage(Input=slc)
rsi.SamplingDimensions = [GRID_NX, GRID_NY, 1]
rsi.SamplingBounds     = [x_min, x_max, y_min, y_max, slice_z - EPS, slice_z + EPS]
rsi.UpdatePipeline(last_time)

print("Fetching data to client …")
img_data = Fetch(rsi)       # vtkImageData on client
pd = img_data.GetPointData()
print(f"  Arrays in resampled image ({pd.GetNumberOfArrays()} total):")
for _i in range(pd.GetNumberOfArrays()):
    print(f"    '{pd.GetArrayName(_i)}'")

# VTK image layout (x fastest): flat index = ix + Nx*iy + Nx*Ny*iz
# With Nz=1:  flat index = ix + GRID_NX*iy  →  reshape(GRID_NY, GRID_NX)
def fetch_grid(name):
    arr = pd.GetArray(name)
    if arr is None:
        raise KeyError(f"Array '{name}' not found in resampled data.")
    return vtk_to_numpy(arr).reshape(GRID_NY, GRID_NX)

# Compute SS_over_GG and WW_over_GG from the gradU tensor (9-component).
# gradU layout: [dUx/dx, dUx/dy, dUx/dz, dUy/dx, ..., dUz/dz]  (row-major)
_raw = pd.GetArray("gradU")
if _raw is None:
    raise KeyError("'gradU' not found — gradient filter may have failed.")
_g = vtk_to_numpy(_raw).reshape(GRID_NY, GRID_NX, 9)
A = {(i, j): _g[..., i * 3 + j] for i in range(3) for j in range(3)}

S_norm_sq = (A[0,0]**2 + A[1,1]**2 + A[2,2]**2
             + 0.5*(A[0,1]+A[1,0])**2
             + 0.5*(A[0,2]+A[2,0])**2
             + 0.5*(A[1,2]+A[2,1])**2)
W_norm_sq = (0.5*(A[0,1]-A[1,0])**2
             + 0.5*(A[0,2]-A[2,0])**2
             + 0.5*(A[1,2]-A[2,1])**2)
GG = S_norm_sq + W_norm_sq + 1e-20

arrays = {
    "SS_over_GG": S_norm_sq / GG,
    "WW_over_GG": W_norm_sq / GG,
}

# Valid-point mask: 1 inside the mesh, 0 outside (e.g. propeller interior)
vpm = pd.GetArray("vtkValidPointMask")
valid = (vtk_to_numpy(vpm).reshape(GRID_NY, GRID_NX).astype(bool)
         if vpm is not None else np.ones((GRID_NY, GRID_NX), dtype=bool))

# ── Build composite RGBA image ─────────────────────────────────────────────
def make_rgba(data, vmin, vmax, thresh, cmap_name):
    """RGBA image for one component; transparent where below threshold or outside mesh."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    # Cast to float32 immediately — colormaps return float64 by default,
    # which costs ~800 MB at 5000×5000 and ~72 MB at 1500×1500.
    rgba = plt.get_cmap(cmap_name)(norm(data)).astype(np.float32)  # (NY, NX, 4)
    rgba[(data < thresh) | ~valid, 3] = 0.0                        # set alpha = 0
    return rgba

# ── Matplotlib figure ──────────────────────────────────────────────────────
print("Building figure …")
extent = [x_min, x_max, y_min, y_max]

fig, ax = plt.subplots(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=IMAGE_DPI)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for name in LAYER_ORDER:
    p = LAYER_PARAMS[name]
    rgba = make_rgba(arrays[name], *p["clim"], p["thresh"], p["cmap"])
    ax.imshow(rgba, origin="lower", extent=extent, aspect="equal",
              interpolation="bilinear")

if XLIM is not None:
    ax.set_xlim(XLIM)
if YLIM is not None:
    ax.set_ylim(YLIM)

ax.set_xlabel("$x$", fontsize=35, fontfamily="Times New Roman")
ax.set_ylabel("$y$", fontsize=35, fontfamily="Times New Roman")
ax.tick_params(labelsize=35)

# ── Colour bars ────────────────────────────────────────────────────────────
# cb_w: width of each colorbar in figure-fraction (smaller = shorter bar)
cb_w      = 0.35
cb_h      = 0.055
cb_y      = 0.06   # ↑ raise colorbars toward the plot
cb_centers = [0.25, 0.75]
fig.subplots_adjust(bottom=0.15, top=0.97, left=0.07, right=0.97)  # ↓ shrink bottom margin

for i, name in enumerate(LAYER_ORDER):
    p    = LAYER_PARAMS[name]
    x0   = cb_centers[i] - cb_w / 2
    cax  = fig.add_axes([x0, cb_y, cb_w, cb_h])
    norm = mcolors.Normalize(vmin=p["clim"][0], vmax=p["clim"][1])
    tick_vals = np.linspace(p["clim"][0], p["clim"][1], 6)
    cb = ColorbarBase(cax, cmap=plt.get_cmap(p["cmap"]),
                      norm=norm, orientation="horizontal", ticks=tick_vals)
    cb.set_label(p["label"], fontsize=35, fontfamily="Times New Roman")
    cb.ax.tick_params(labelsize=35)
    cb.set_ticklabels([f"{t:.1f}" for t in tick_vals])
    # Mark threshold on the colour bar with a dashed vertical line
    thresh_norm = (p["thresh"] - p["clim"][0]) / (p["clim"][1] - p["clim"][0])
    if 0 < thresh_norm < 1:
        cax.axvline(thresh_norm * (p["clim"][1] - p["clim"][0]) + p["clim"][0],
                    color="k", linewidth=1.0, linestyle="--")

# ── Save ────────────────────────────────────────────────────────────────────
fig.savefig(str(output_png), dpi=IMAGE_DPI, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"\nSaved → {output_png}")
print("Done.")
