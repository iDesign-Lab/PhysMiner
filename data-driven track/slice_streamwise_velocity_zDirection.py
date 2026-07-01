"""
slice_streamwise_velocity_zDirection.py

z = const slice showing the streamwise (x-component) velocity Ux.

The slice plane is x-y; x is the streamwise direction, y is the wall-normal
direction.

Run with pvpython:
    "D:\\software\\paraView\\bin\\pvpython.exe" slice_streamwise_velocity_zDirection.py

Tuning guide
────────────
  CLIM_UX        : colour-bar limits [vmin, vmax]; None = auto from data
  XLIM / YLIM    : axis display range, e.g. (0.0, 20.0); None = full data extent
  SLICE_Z        : z-position of the slice (None = mesh z-centre)
  VELOCITY_FIELD : "U" for instantaneous, "UMean" for time-averaged
  GRID_NX/NY    : resampling resolution (higher → sharper but slower)
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
output_png  = results_dir / "slice_streamwise_velocity_zDirection.png"

print(f"Case   : {case_dir}")
print(f"Output : {output_png}")

# ── user parameters ────────────────────────────────────────────────────────
# Colour-bar limits for Ux.  Set to None to auto-detect from the slice data.
# CLIM_UX = None          # e.g. (0.0, 1.5) to fix range; None = auto
CLIM_UX = (0.0, 1.0)

# Axis display range.  None = use full data extent from the mesh.
# Example: XLIM = (0.0, 20.0)  YLIM = (-1.0, 3.0)
XLIM = None             # (x_min, x_max) or None
# XLIM = (-0.1444, 0.76)             # (x_min, x_max) or None
YLIM = None             # (y_min, y_max) or None
# Resampling grid (number of pixels in x and y directions)
GRID_NX = 1500
GRID_NY = 1500

# Slice z-position: None = auto-detect (uses mesh z-centre)
# To fix the slice at a specific z position, replace None with a numeric value.
# SLICE_Z = None
SLICE_Z = None

# Velocity field name:
#   "U"     → instantaneous velocity (shows turbulent structures — recommended)
#   "UMean" → time-averaged (uniform in x-z for channel flow → boring flat colour)
VELOCITY_FIELD = "U"

# Output image
IMAGE_DPI    = 300
IMAGE_WIDTH  = 16     # figure width  in inches
IMAGE_HEIGHT = 9      # figure height in inches

# Colour-bar position in figure coordinates [0, 1].
# [left, bottom, width, height]  — all four values are fractions of the figure size.
# CB_BOTTOM also controls how much space is reserved below the axes (subplots_adjust bottom).
CB_LEFT   = 0.15    # left edge of the colour bar
CB_BOTTOM = 0.10    # bottom edge of the colour bar
CB_WIDTH  = 0.70    # width  of the colour bar
CB_HEIGHT = 0.055   # height of the colour bar
# Space between the axes bottom and the figure bottom (must be > CB_BOTTOM + CB_HEIGHT)
SUBPLOT_BOTTOM = 0.20

# Colour-bar tick label decimal places (e.g. 1 → "11.3", 2 → "11.30", 0 → "11")
CB_DECIMALS = 1

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

# Determine slice z-position from mesh bounds if not set by user
mesh_bounds = c2p.GetDataInformation().GetBounds()
z_lo, z_hi = mesh_bounds[4], mesh_bounds[5]
print(f"  Mesh z-extent: [{z_lo:.6f}, {z_hi:.6f}]")
slice_z = SLICE_Z if SLICE_Z is not None else 0.5 * (z_lo + z_hi)
print(f"Slice at z = {slice_z:.6f} …")
slc = Slice(Input=c2p)
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

# VTK image layout (x fastest): flat index = ix + Nx*iy + Nx*Ny*iz
# With Nz=1:  flat index = ix + GRID_NX*iy  →  reshape(GRID_NY, GRID_NX)
def fetch_scalar_grid(name):
    """Fetch a scalar field and reshape to (GRID_NY, GRID_NX)."""
    arr = pd.GetArray(name)
    if arr is None:
        raise KeyError(f"Array '{name}' not found in resampled data.")
    return vtk_to_numpy(arr).reshape(GRID_NY, GRID_NX)


def fetch_vector_component(name, component=0):
    """Fetch one component of a vector field and reshape to (GRID_NY, GRID_NX).

    vtk_to_numpy on a 3-component array returns shape (N, 3); this function
    extracts the requested component and reshapes to the image grid.
    """
    arr = pd.GetArray(name)
    if arr is None:
        raise KeyError(f"Array '{name}' not found in resampled data.")
    data = vtk_to_numpy(arr)
    if data.ndim == 2:
        # shape (N, ncomp) — standard for multi-component VTK arrays
        return data[:, component].reshape(GRID_NY, GRID_NX)
    else:
        # shape (N*ncomp,) — interleaved fallback
        ncomp = arr.GetNumberOfComponents()
        return data[component::ncomp].reshape(GRID_NY, GRID_NX)


# Extract streamwise (x) component of the chosen velocity field
print(f"  Reading '{VELOCITY_FIELD}' x-component …")
ux_grid = fetch_vector_component(VELOCITY_FIELD, component=0)
print(f"  Ux range on slice: [{ux_grid.min():.4f}, {ux_grid.max():.4f}]")

# Valid-point mask: 1 inside the mesh, 0 outside
vpm = pd.GetArray("vtkValidPointMask")
valid = (vtk_to_numpy(vpm).reshape(GRID_NY, GRID_NX).astype(bool)
         if vpm is not None else np.ones((GRID_NY, GRID_NX), dtype=bool))

# Mask Ux outside the mesh domain
ux_masked = np.ma.array(ux_grid, mask=~valid)

# ── Matplotlib figure ──────────────────────────────────────────────────────
print("Building figure …")
extent = [x_min, x_max, y_min, y_max]

fig, ax = plt.subplots(figsize=(IMAGE_WIDTH, IMAGE_HEIGHT), dpi=IMAGE_DPI)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Auto colour limits from valid data if not specified by user
if CLIM_UX is None:
    vmin = float(np.nanmin(ux_masked.compressed()))
    vmax = float(np.nanmax(ux_masked.compressed()))
    print(f"  Auto colour limits: [{vmin:.4f}, {vmax:.4f}]")
else:
    vmin, vmax = CLIM_UX

norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
cmap = plt.get_cmap("coolwarm").copy()
cmap.set_bad(color="white")   # masked (out-of-domain) pixels → white

im = ax.imshow(ux_masked, origin="lower", extent=extent, aspect="equal",
               interpolation="bilinear", cmap=cmap, norm=norm)

if XLIM is not None:
    ax.set_xlim(XLIM)
if YLIM is not None:
    ax.set_ylim(YLIM)

ax.set_xlabel("$x$", fontsize=35, fontfamily="Times New Roman")
ax.set_ylabel("$y$", fontsize=35, fontfamily="Times New Roman")
# ax.set_title(f"Streamwise velocity  $U_x$  at  $z = {slice_z:.4f}$",
            #  fontsize=35, fontfamily="Times New Roman")
ax.tick_params(labelsize=35)

# ── Colour bar ─────────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=SUBPLOT_BOTTOM, top=0.97, left=0.07, right=0.97)

cax = fig.add_axes([CB_LEFT, CB_BOTTOM, CB_WIDTH, CB_HEIGHT])
tick_vals = np.linspace(vmin, vmax, 7)
cb = ColorbarBase(cax, cmap=cmap, norm=norm,
                  orientation="horizontal", ticks=tick_vals)
cb.set_label(r"$U_x$", fontsize=35, fontfamily="Times New Roman")
cb.ax.tick_params(labelsize=35)
cb.set_ticklabels([f"{t:.{CB_DECIMALS}f}" for t in tick_vals])

# ── Save ────────────────────────────────────────────────────────────────────
fig.savefig(str(output_png), dpi=IMAGE_DPI, bbox_inches="tight",
            facecolor="white")
plt.close(fig)
print(f"\nSaved → {output_png}")
print("Done.")
