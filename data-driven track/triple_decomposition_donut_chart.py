"""
triple_decomposition_donut_chart.py

Read the four triple-decomposition scalar fields (gg_rr, gg_ps, gg_ns, gg_rs)
over the entire computational domain, compute each component's domain-averaged
mean, normalise to 100 %, and plot a donut (ring) chart.

Run with pvpython:
    "D:\software\paraView\bin\pvpython.exe" triple_decomposition_donut_chart.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# ── paths ──────────────────────────────────────────────────────────────────
script_dir  = Path(__file__).parent.resolve()
case_dir    = (script_dir / "../../").resolve()
foam_file   = case_dir / "para.foam"
results_dir = (script_dir / "../results").resolve()
results_dir.mkdir(parents=True, exist_ok=True)
output_png  = results_dir / "triple_decomposition_donut_chart.png"

print(f"Case   : {case_dir}")
print(f"Output : {output_png}")

# ── field metadata ─────────────────────────────────────────────────────────
# Rendering order in the pie (clockwise from 12 o'clock)
FIELD_ORDER = ["gg_ns", "gg_rs", "gg_rr", "gg_ps"]

FIELD_LABEL = {
    "gg_rr": r"$gg_{rr}$",
    "gg_ps": r"$gg_{ps}$",
    "gg_ns": r"$gg_{ns}$",
    "gg_rs": r"$gg_{rs}$",
}
FIELD_DESC = {
    "gg_rr": "Rigid Rotation",
    "gg_ps": "Pure Shear",
    "gg_ns": "Normal Strain",
    "gg_rs": "Rotation-shear",
}
FIELD_COLOR = {
    "gg_rr": "#D62728",   # red
    "gg_ps": "#2CA02C",   # green
    "gg_ns": "#1F77B4",   # blue
    "gg_rs": "#FF7F0E",   # orange
}

# ── ParaView pipeline ──────────────────────────────────────────────────────
from paraview.simple import OpenFOAMReader, CellDatatoPointData
import paraview.simple as pvs
from paraview.servermanager import Fetch
from vtk.util.numpy_support import vtk_to_numpy

pvs._DisableFirstRenderCameraReset()

print("\nOpening OpenFOAM case …")
reader = OpenFOAMReader(FileName=str(foam_file))
reader.MeshRegions = ["internalMesh"]
reader.UpdatePipelineInformation()
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

print("Fetching full-domain data to client …")
domain_data = Fetch(c2p)   # returns vtkMultiBlockDataSet for OpenFOAM

def collect_point_arrays(multiblock) -> dict:
    """Traverse all blocks and concatenate point-data arrays by name."""
    result = {}
    it = multiblock.NewIterator()
    it.InitTraversal()
    while not it.IsDoneWithTraversal():
        ds = it.GetCurrentDataObject()
        if ds is not None and hasattr(ds, "GetPointData"):
            pd = ds.GetPointData()
            for j in range(pd.GetNumberOfArrays()):
                name = pd.GetArrayName(j)
                vals = vtk_to_numpy(pd.GetArray(j))
                if name in result:
                    result[name] = np.concatenate([result[name], vals])
                else:
                    result[name] = vals
        it.GoToNextItem()
    return result

all_arrays = collect_point_arrays(domain_data)
print(f"  Available arrays ({len(all_arrays)}):")
for name in sorted(all_arrays):
    print(f"    '{name}'  shape={all_arrays[name].shape}")

# ── compute domain-averaged mean for each component ────────────────────────
means = {}
for name in FIELD_ORDER:
    if name not in all_arrays:
        raise KeyError(f"Array '{name}' not found in fetched data.")
    means[name] = float(np.mean(all_arrays[name]))
    print(f"  mean({name}) = {means[name]:.6f}")

total   = sum(means.values())
pct_raw = {k: v / total * 100.0 for k, v in means.items()}

def largest_remainder_round(d: dict, decimals: int = 1) -> dict:
    """Round values so their displayed sum is exactly 100.0 (largest-remainder method)."""
    factor   = 10 ** decimals
    keys     = list(d.keys())
    scaled   = [d[k] * factor for k in keys]
    floored  = [int(v) for v in scaled]
    deficit  = int(round(100 * factor)) - sum(floored)
    fracs    = sorted(range(len(keys)),
                      key=lambda i: scaled[i] - floored[i], reverse=True)
    for i in fracs[:deficit]:
        floored[i] += 1
    return {keys[i]: floored[i] / factor for i in range(len(keys))}

pct = largest_remainder_round(pct_raw, decimals=1)
print("\nComposition (rounded, sum=100%):")
for k, p in pct.items():
    print(f"  {k}: {p:.1f}%")

# ── figure layout parameters ───────────────────────────────────────────────
# Donut size is constrained by min(axes_width, axes_height).
# Since set_aspect="equal", increasing only FIG_HEIGHT past FIG_WIDTH
# gives no bigger donut — increase FIG_WIDTH to grow the chart.
FIG_WIDTH  = 14     # inches  ← primary knob for donut size
FIG_HEIGHT = 30     # inches  ← keep close to FIG_WIDTH for a near-square layout
FIG_DPI    = 300

# Legend vertical position in axes-fraction coordinates (negative = below axes).
LEGEND_Y   = -0.04

# ── matplotlib figure ──────────────────────────────────────────────────────
rcParams["font.family"]       = "serif"
rcParams["font.serif"]        = ["Times New Roman", "Times", "DejaVu Serif"]
rcParams["mathtext.fontset"]  = "stix"   # STIX ≈ Times for math

FONT_LABEL  = {"fontfamily": "serif", "fontsize": 40}
FONT_PCT_IN = {"fontfamily": "serif", "fontsize": 40, "fontweight": "bold", "color": "black"}
FONT_PCT_OUT= {"fontfamily": "serif", "fontsize": 40, "fontweight": "bold", "color": "black"}

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

sizes  = [pct[k]          for k in FIELD_ORDER]
colors = [FIELD_COLOR[k]  for k in FIELD_ORDER]

wedges, _ = ax.pie(
    sizes,
    colors=colors,
    startangle=90,
    counterclock=False,
    wedgeprops=dict(width=0.50, edgecolor="white", linewidth=2.5),
)

# ── annotate wedges ────────────────────────────────────────────────────────
# Threshold: wedges larger than this get their % printed inside
INSIDE_THRESH = 5.0

for wedge, field, size in zip(wedges, FIELD_ORDER, sizes):
    mid_angle = (wedge.theta2 + wedge.theta1) / 2.0
    rad       = np.deg2rad(mid_angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    pct_str = f"{size:.1f}%"

    if size >= INSIDE_THRESH:
        # Percent inside the ring band
        r_in = 0.76
        ax.text(r_in * cos_a, r_in * sin_a, pct_str,
                ha="center", va="center", **FONT_PCT_IN)

        # Component label outside, no leader line needed for large wedges
        r_lbl = 1.18
        ax.text(r_lbl * cos_a, r_lbl * sin_a, FIELD_LABEL[field],
                ha="center", va="center",
                style="italic", **FONT_LABEL)
    else:
        # Small wedge: leader line starts at outer edge of ring (r=1.02),
        # goes to label anchor outside — no line passes through the hole.
        r_arrow_start = 1.02   # just outside the ring outer edge
        r_arrow_end   = 1.18
        r_text        = 1.22

        ax.annotate(
            "",
            xy     =(r_arrow_start * cos_a, r_arrow_start * sin_a),
            xytext =(r_arrow_end   * cos_a, r_arrow_end   * sin_a),
            arrowprops=dict(arrowstyle="-", color="gray", lw=1.0),
            annotation_clip=False,
        )

        # Label (italic) and percent (bold) stacked at tip
        # Nudge slightly so they don't overlap when the wedges are adjacent
        offset_x = 0.0
        offset_y = 0.0
        if cos_a > 0:
            ha = "left"
            offset_x = 0.05
        elif cos_a < 0:
            ha = "right"
            offset_x = -0.05
        else:
            ha = "center"

        ax.text(r_text * cos_a + offset_x,
                r_text * sin_a + offset_y + 0.12,
                pct_str,
                ha=ha, va="bottom", **FONT_PCT_OUT)
        ax.text(r_text * cos_a + offset_x,
                r_text * sin_a + offset_y - 0.02,
                FIELD_LABEL[field],
                ha=ha, va="top", style="italic", **FONT_LABEL)

ax.set_xlim(-1.55, 1.55)
ax.set_ylim(-1.50, 1.20)   # reduced top margin to cut white space
ax.set_aspect("equal")
ax.axis("off")
# Reserve space at bottom for the (now large-font) legend; rest goes to donut.
# bottom=0.14 ≈ 14 % of figure height for the legend area.
fig.subplots_adjust(top=0.97, bottom=0.14, left=0.01, right=0.99)

# ── legend ─────────────────────────────────────────────────────────────────
legend_order = ["gg_ns", "gg_rr", "gg_ps", "gg_rs"]
handles = [
    mpatches.Patch(
        color=FIELD_COLOR[k],
        label=f"{FIELD_LABEL[k]} — {FIELD_DESC[k]}  ({pct[k]:.1f}%)",
    )
    for k in legend_order
]

leg = ax.legend(
    handles=handles,
    loc="lower center",
    bbox_to_anchor=(0.5, LEGEND_Y),
    ncol=2,
    frameon=True,
    framealpha=1.0,
    edgecolor="black",
    prop={"family": "serif", "size": 35},
)

# ── save ───────────────────────────────────────────────────────────────────
fig.savefig(str(output_png), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved → {output_png}")
print("Done.")
