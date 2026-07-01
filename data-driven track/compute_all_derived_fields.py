#!/usr/bin/env python3
"""
compute_all_derived_fields.py
==============================
Offline (no OpenFOAM, no HDF5) computation of ALL derived fields for an
OpenFOAM case.  Reads directly from constant/polyMesh and the last time
directory's U field, then writes the following volFields back into that
same time directory:

  Liutex        volVectorField  [0 0 -1 0 0 0 0]
  LiutexMag     volScalarField  [0 0 -1 0 0 0 0]
  gg_rr         volScalarField  [0 0 0 0 0 0 0]   pure rotation
  gg_ps         volScalarField  [0 0 0 0 0 0 0]   pure shear
  gg_ns         volScalarField  [0 0 0 0 0 0 0]   normal strain
  gg_rs         volScalarField  [0 0 0 0 0 0 0]   residual
  vorticity     volVectorField  [0 0 -1 0 0 0 0]
  vorticityMag  volScalarField  [0 0 -1 0 0 0 0]
  Q             volScalarField  [0 0 -2 0 0 0 0]
  WW_over_GG    volScalarField  [0 0 0 0 0 0 0]   W:W / G:G
  SS_over_GG    volScalarField  [0 0 0 0 0 0 0]   S:S / G:G

Usage
-----
    python compute_all_derived_fields.py [case_dir]

    case_dir  (optional) path to the OpenFOAM case root.
              Default: two directory levels above this script
              (i.e. the layout expected when this file lives in
               <case>/PhysMiner/postprocessing_codes/).

Requirements: Python 3.10+, numpy >= 1.20
"""

from __future__ import annotations

import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
SMALL = 1.0e-15
EPS   = 1.0e-6


# ── timing helper ─────────────────────────────────────────────────────────────

class _T:
    def __init__(self, msg: str):
        self.msg = msg
    def __enter__(self):
        print(f"  {self.msg} ...", end="", flush=True)
        self._t = time.perf_counter()
        return self
    def __exit__(self, *_):
        print(f"  {time.perf_counter()-self._t:.1f}s")


# ── low-level file helpers ────────────────────────────────────────────────────

def _strip_header(text: str) -> str:
    i = text.find("FoamFile")
    if i == -1:
        return text
    depth = 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:]
        i += 1
    return text


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _list_block(text: str) -> str:
    text = _strip_comments(text)
    m = re.search(r"\d+\s*\(", text)
    if not m:
        raise ValueError("No list block (N followed by '(') found")
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        i += 1
    return text[start: i - 1]


def _is_binary(raw: bytes) -> bool:
    m = re.search(rb'format\s+(\w+)', raw[:1200])
    return bool(m and m.group(1) == b'binary')


def _bin_section(raw: bytes, search_from: int = 0) -> tuple[int, int]:
    m = re.search(rb'(\d+)\s*\(', raw[search_from:])
    if not m:
        raise ValueError("Cannot find count+( pattern in binary data")
    return int(m.group(1)), search_from + m.end()


# ── mesh readers ──────────────────────────────────────────────────────────────

def read_points(poly_dir: Path) -> np.ndarray:
    path = poly_dir / "points"
    with _T("read points"):
        raw = path.read_bytes()
        if _is_binary(raw):
            N, pos = _bin_section(raw)
            pts = np.frombuffer(raw, dtype='<f8', count=N * 3, offset=pos).reshape(-1, 3).copy()
        else:
            text = raw.decode("utf-8", errors="replace")
            block = _list_block(_strip_header(text))
            flat = block.replace("(", " ").replace(")", " ")
            pts = np.fromstring(flat, dtype=np.float64, sep=" ").reshape(-1, 3)
    print(f"    {len(pts):,} points")
    return pts


def read_label_list(path: Path, name: str) -> np.ndarray:
    with _T(f"read {name}"):
        raw = path.read_bytes()
        if _is_binary(raw):
            N, pos = _bin_section(raw)
            arr = np.frombuffer(raw, dtype='<i4', count=N, offset=pos).copy()
        else:
            text = raw.decode("utf-8", errors="replace")
            block = _list_block(_strip_header(text))
            arr = np.fromstring(block, dtype=np.int32, sep="\n")
    print(f"    {len(arr):,} labels")
    return arr


def _parse_mixed_faces(block: str, nf: int) -> np.ndarray:
    verts = np.zeros((nf, 4), dtype=np.int32)
    pattern = re.compile(r"(\d+)\(([^)]+)\)")
    for idx, m in enumerate(pattern.finditer(block)):
        if idx >= nf:
            break
        n = int(m.group(1))
        v = np.fromstring(m.group(2), dtype=np.int32, sep=" ")
        verts[idx, :n] = v[:4]
        if n == 3:
            verts[idx, 3] = v[0]
    return verts


def read_faces(poly_dir: Path) -> tuple[np.ndarray, int | None]:
    path = poly_dir / "faces"
    with _T("read faces"):
        raw = path.read_bytes()

    note_m = re.search(rb'nInternalFaces\s*:\s*(\d+)', raw[:800])
    n_int = int(note_m.group(1)) if note_m else None

    if _is_binary(raw):
        N1, pos1 = _bin_section(raw)
        offsets = np.frombuffer(raw, dtype='<i4', count=N1, offset=pos1).copy()
        pos_after = pos1 + N1 * 4
        N2, pos2 = _bin_section(raw, pos_after)
        verts_flat = np.frombuffer(raw, dtype='<i4', count=N2, offset=pos2).copy()
        nFaces = N1 - 1
        face_sizes = np.diff(offsets)
        if len(np.unique(face_sizes)) == 1 and face_sizes[0] == 4:
            verts = verts_flat.reshape(-1, 4)
        else:
            print(f"  Mixed faces binary — padding to 4")
            verts = np.zeros((nFaces, 4), dtype=np.int32)
            for i in range(nFaces):
                s, e = int(offsets[i]), int(offsets[i + 1])
                n = e - s
                verts[i, :min(n, 4)] = verts_flat[s:s + min(n, 4)]
                if n < 4:
                    verts[i, n:] = verts_flat[s]
        print(f"    {nFaces:,} faces  (internal={n_int})")
        return verts, n_int

    text = raw.decode("utf-8", errors="replace")
    block = _list_block(_strip_header(text))
    n4 = block.count("4(")
    n3 = block.count("3(")
    n5 = block.count("5(")
    nf = n4 + n3 + n5
    if n3 == 0 and n5 == 0:
        with _T("parse quad faces"):
            flat = block.replace("4(", " ").replace(")", " ")
            verts = np.fromstring(flat, dtype=np.int32, sep=" ").reshape(-1, 4)
    else:
        print(f"  Mixed faces ({n4} quads, {n3} tris, {n5} penta) — general parser")
        with _T("parse mixed faces"):
            verts = _parse_mixed_faces(block, nf)
    print(f"    {len(verts):,} faces  (internal={n_int})")
    return verts, n_int


def read_boundary(poly_dir: Path) -> dict:
    path = poly_dir / "boundary"
    raw = _strip_comments(_strip_header(
        path.read_text(encoding="utf-8", errors="replace")
    ))
    patches: dict = {}
    pat = re.compile(r"(\w+)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
    for m in pat.finditer(raw):
        name, body = m.group(1), m.group(2)
        info: dict = {}
        for key in ("nFaces", "startFace"):
            km = re.search(rf"{key}\s+(\d+)", body)
            if km:
                info[key] = int(km.group(1))
        tm = re.search(r"type\s+(\w+)", body)
        if tm:
            info["type"] = tm.group(1)
        if "nFaces" in info:
            patches[name] = info
    return patches


# ── field reader ──────────────────────────────────────────────────────────────

def read_vector_internal(path: Path) -> np.ndarray:
    with _T(f"read {path.name}"):
        raw = path.read_bytes()
    if _is_binary(raw):
        m = re.search(
            rb"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", raw
        )
        if not m:
            raise ValueError(f"Cannot find internalField (nonuniform vector) in {path}")
        N = int(m.group(1))
        pos = m.end()
        arr = np.frombuffer(raw, dtype='<f8', count=N * 3, offset=pos).reshape(-1, 3).copy()
    else:
        text = raw.decode("utf-8", errors="replace")
        m = re.search(
            r"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", text
        )
        if not m:
            raise ValueError(f"Cannot find internalField (nonuniform vector) in {path}")
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        block = text[start: i - 1]
        flat = block.replace("(", " ").replace(")", " ")
        arr = np.fromstring(flat, dtype=np.float64, sep=" ").reshape(-1, 3)
    print(f"    {len(arr):,} cell vectors")
    return arr


# ── face / cell geometry ──────────────────────────────────────────────────────

def compute_face_geometry(
    pts: np.ndarray, verts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    with _T("compute face geometry"):
        p0 = pts[verts[:, 0]]
        p1 = pts[verts[:, 1]]
        p2 = pts[verts[:, 2]]
        p3 = pts[verts[:, 3]]
        fc = (p0 + p1 + p2 + p3) * 0.25
        Sf = 0.5 * np.cross(p2 - p0, p3 - p1)
    return fc, Sf


def compute_cell_volumes(
    owner: np.ndarray,
    neighbour: np.ndarray,
    fc: np.ndarray,
    Sf: np.ndarray,
    nCells: int,
) -> np.ndarray:
    with _T("compute cell volumes"):
        nIF = len(neighbour)
        fc_dot_Sf = np.einsum("fi,fi->f", fc, Sf)
        vols = np.zeros(nCells, dtype=np.float64)
        np.add.at(vols, owner,      fc_dot_Sf / 3.0)
        np.add.at(vols, neighbour, -fc_dot_Sf[:nIF] / 3.0)
        vols = np.maximum(np.abs(vols), SMALL)
    print(f"    vol_min={vols.min():.3e}  vol_max={vols.max():.3e}")
    return vols


# ── velocity gradient (Green-Gauss / Gauss-linear) ───────────────────────────

def compute_gradU(
    U: np.ndarray,
    owner: np.ndarray,
    neighbour: np.ndarray,
    fc: np.ndarray,
    Sf: np.ndarray,
    vols: np.ndarray,
) -> np.ndarray:
    nIF    = len(neighbour)
    nCells = len(vols)
    with _T("assemble gradU (internal faces)"):
        Uf_int = 0.5 * (U[owner[:nIF]] + U[neighbour])
        contrib_int = Uf_int[:, :, np.newaxis] * Sf[:nIF, np.newaxis, :]
        gradU = np.zeros((nCells, 3, 3), dtype=np.float64)
        np.add.at(gradU, owner[:nIF],  contrib_int)
        np.add.at(gradU, neighbour,   -contrib_int)
    with _T("assemble gradU (boundary faces)"):
        Uf_bnd = U[owner[nIF:]]
        contrib_bnd = Uf_bnd[:, :, np.newaxis] * Sf[nIF:, np.newaxis, :]
        np.add.at(gradU, owner[nIF:], contrib_bnd)
    with _T("normalise gradU by cell volume"):
        gradU /= vols[:, np.newaxis, np.newaxis]
    return gradU


# ── combined physics: ALL derived fields in one pass ─────────────────────────

def compute_all_fields(gradU: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute every derived field from the velocity gradient tensor.

    gradU : (nCells, 3, 3)  where gradU[c, i, j] = dU_i/dx_j

    Returns dict with keys:
        Liutex, LiutexMag, gg_rr, gg_ps, gg_ns, gg_rs,
        vorticity, vorticityMag, Q, WW_over_GG, SS_over_GG
    """
    nC = len(gradU)
    g  = gradU.reshape(nC, 9)

    dUdX, dUdY, dUdZ = g[:, 0], g[:, 1], g[:, 2]
    dVdX, dVdY, dVdZ = g[:, 3], g[:, 4], g[:, 5]
    dWdX, dWdY, dWdZ = g[:, 6], g[:, 7], g[:, 8]

    # ── Frobenius norm² ──────────────────────────────────────────────────────
    D = (dUdX**2 + dUdY**2 + dUdZ**2
       + dVdX**2 + dVdY**2 + dVdZ**2
       + dWdX**2 + dWdY**2 + dWdZ**2)

    # ── vorticity ω = curl U ─────────────────────────────────────────────────
    omX = dWdY - dVdZ
    omY = dUdZ - dWdX
    omZ = dVdX - dUdY
    omega = np.stack([omX, omY, omZ], axis=1)        # (nC, 3)
    omega_mag = np.linalg.norm(omega, axis=1)        # (nC,)

    # ── Q-criterion: Q = -½ tr(G²) = -½ Σ_ij G_ij G_ji ─────────────────────
    GG_tr = (dUdX*dUdX + dUdY*dVdX + dUdZ*dWdX
           + dVdX*dUdY + dVdY*dVdY + dVdZ*dWdY
           + dWdX*dUdZ + dWdY*dVdZ + dWdZ*dWdZ)
    Q = -0.5 * GG_tr

    # ── W:W / G:G and S:S / G:G ──────────────────────────────────────────────
    WW = 0.5 * (D - GG_tr)          # W_ij W_ij
    safe = D > 1e-30 * max(float(D.max()), 1.0)
    WW_over_GG = np.where(safe, WW / np.where(safe, D, 1.0), 0.0)
    SS_over_GG = 1.0 - WW_over_GG

    # ── Liutex: characteristic polynomial ────────────────────────────────────
    I1 = dUdX + dVdY + dWdZ
    I2 = ((dUdX*dVdY + dUdX*dWdZ + dVdY*dWdZ)
        - (dUdY*dVdX + dUdZ*dWdX + dVdZ*dWdY))
    I3 = (dUdX*(dVdY*dWdZ - dVdZ*dWdY)
        - dUdY*(dVdX*dWdZ - dVdZ*dWdX)
        + dUdZ*(dVdX*dWdY - dVdY*dWdX))

    pp   = (3.0*I2 - I1*I1) / 3.0
    qq   = (2.0*I1*I1*I1 - 9.0*I1*I2 + 27.0*I3) / 27.0
    disc = (qq / 2.0)**2 + (pp / 3.0)**3

    vortical = disc > 0.0

    R_arr  = np.zeros(nC, dtype=np.float64)
    rx_arr = np.zeros(nC, dtype=np.float64)
    ry_arr = np.zeros(nC, dtype=np.float64)
    rz_arr = np.zeros(nC, dtype=np.float64)

    if np.any(vortical):
        v      = vortical
        sqrtD  = np.sqrt(disc[v])
        S1     = np.cbrt(-qq[v] / 2.0 + sqrtD)
        S2     = np.cbrt(-qq[v] / 2.0 - sqrtD)

        lambdaR  = S1 + S2 + I1[v] / 3.0
        lambdaCi = np.abs(np.sqrt(3.0) / 2.0 * (S1 - S2))

        B11 = dUdX[v] - lambdaR;  B12 = dUdY[v];             B13 = dUdZ[v]
        B21 = dVdX[v];            B22 = dVdY[v] - lambdaR;   B23 = dVdZ[v]
        det2 = B11*B22 - B12*B21

        rx = np.zeros(np.count_nonzero(v))
        ry = np.zeros_like(rx)
        rz = np.ones_like(rx)

        nd = np.abs(det2) > SMALL
        if np.any(nd):
            rx[nd] = (B13[nd]*B22[nd] - B12[nd]*B23[nd]) / det2[nd]
            ry[nd] = (B11[nd]*B23[nd] - B13[nd]*B21[nd]) / det2[nd]

        degen = ~nd
        if np.any(degen):
            B31 = dWdX[v][degen];  B32 = dWdY[v][degen]
            B33 = dWdZ[v][degen] - lambdaR[degen]
            det2b = B11[degen]*B33 - B13[degen]*B31
            good2 = np.abs(det2b) > SMALL
            if np.any(good2):
                idx_d  = np.where(degen)[0]
                idx_g  = idx_d[good2]
                rx[idx_g] = (B12[degen][good2]*B33[good2] - B13[degen][good2]*B32[good2]) / det2b[good2]
                rz[idx_g] = (B11[degen][good2]*B32[good2] - B12[degen][good2]*B31[good2]) / det2b[good2]
                ry[idx_g] = 1.0

        nm = np.maximum(np.sqrt(rx**2 + ry**2 + rz**2), SMALL)
        rx /= nm;  ry /= nm;  rz /= nm

        omDotR = omX[v]*rx + omY[v]*ry + omZ[v]*rz
        flip   = omDotR < 0.0
        rx[flip] = -rx[flip];  ry[flip] = -ry[flip];  rz[flip] = -rz[flip]

        Wr = omX[v]*rx + omY[v]*ry + omZ[v]*rz
        R  = Wr - np.sqrt(np.maximum(Wr**2 - 4.0*lambdaCi**2, 0.0))

        R_arr[v]  = R
        rx_arr[v] = rx;  ry_arr[v] = ry;  rz_arr[v] = rz

    # ── triple decomposition ─────────────────────────────────────────────────
    gg_rr = R_arr**2 / (2.0*D + EPS)

    sx = omX - R_arr*rx_arr
    sy = omY - R_arr*ry_arr
    sz = omZ - R_arr*rz_arr
    gg_ps = (sx**2 + sy**2 + sz**2) / (D + EPS)

    symNorm = (dUdX**2 + dVdY**2 + dWdZ**2
             + 0.5*(dUdY + dVdX)**2
             + 0.5*(dUdZ + dWdX)**2
             + 0.5*(dVdZ + dWdY)**2)
    gg_ns = symNorm / (D + EPS) - 0.5*gg_ps
    gg_rs = 1.0 - gg_rr - gg_ps - gg_ns

    return {
        "Liutex":       np.stack([R_arr*rx_arr, R_arr*ry_arr, R_arr*rz_arr], axis=1),
        "LiutexMag":    R_arr,
        "gg_rr":        gg_rr,
        "gg_ps":        gg_ps,
        "gg_ns":        gg_ns,
        "gg_rs":        gg_rs,
        "vorticity":    omega,
        "vorticityMag": omega_mag,
        "Q":            Q,
        "WW_over_GG":   WW_over_GG,
        "SS_over_GG":   SS_over_GG,
    }


# ── OpenFOAM ASCII field writers ──────────────────────────────────────────────

_FOAM_HEADER = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2312                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {cls};
    location    "{loc}";
    object      {obj};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""

_FOAM_FOOTER = "// ************************************************************************* //\n"


def _boundary_section(
    patches: dict,
    owner: np.ndarray,
    data: np.ndarray,
    field_type: str,
) -> str:
    lines = ["boundaryField\n{\n"]
    for name, info in patches.items():
        nf   = info["nFaces"]
        sf   = info["startFace"]
        vals = data[owner[sf: sf + nf]]
        lines.append(f"    {name}\n    {{\n")
        lines.append("        type            calculated;\n")
        if field_type == "scalar":
            lines.append(f"        value           nonuniform List<scalar>\n{nf}\n(\n")
            lines.append("\n".join(f"{v:.10g}" for v in vals))
            lines.append("\n)\n;\n")
        else:
            lines.append(f"        value           nonuniform List<vector>\n{nf}\n(\n")
            lines.append("\n".join(
                f"({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})" for v in vals
            ))
            lines.append("\n)\n;\n")
        lines.append("    }\n")
    lines.append("}\n\n")
    return "".join(lines)


def write_scalar_field(
    path: Path,
    name: str,
    time_name: str,
    data: np.ndarray,
    patches: dict,
    owner: np.ndarray,
    dims: str = "[0 0 0 0 0 0 0]",
) -> None:
    n = len(data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_FOAM_HEADER.format(cls="volScalarField", loc=time_name, obj=name))
        f.write(f"\ndimensions      {dims};\n\n")
        f.write(f"internalField   nonuniform List<scalar>\n{n}\n(\n")
        for v in data:
            f.write(f"{v:.10g}\n")
        f.write(")\n;\n\n")
        f.write(_boundary_section(patches, owner, data, "scalar"))
        f.write(_FOAM_FOOTER)


def write_vector_field(
    path: Path,
    name: str,
    time_name: str,
    data: np.ndarray,
    patches: dict,
    owner: np.ndarray,
    dims: str = "[0 0 -1 0 0 0 0]",
) -> None:
    n = len(data)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_FOAM_HEADER.format(cls="volVectorField", loc=time_name, obj=name))
        f.write(f"\ndimensions      {dims};\n\n")
        f.write(f"internalField   nonuniform List<vector>\n{n}\n(\n")
        for v in data:
            f.write(f"({v[0]:.10g} {v[1]:.10g} {v[2]:.10g})\n")
        f.write(")\n;\n\n")
        f.write(_boundary_section(patches, owner, data, "vector"))
        f.write(_FOAM_FOOTER)


# ── path helpers ──────────────────────────────────────────────────────────────

def find_last_time_dir(case_dir: Path) -> Path:
    hits: list[tuple[float, Path]] = []
    for d in case_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            hits.append((float(d.name), d))
        except ValueError:
            pass
    if not hits:
        raise FileNotFoundError(f"No numeric time directories in {case_dir}")
    return max(hits, key=lambda x: x[0])[1]


# ── main pipeline ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute all derived fields (Liutex, TD, vorticity, Q, …) offline."
    )
    parser.add_argument(
        "case_dir", nargs="?", default=None,
        help="OpenFOAM case root (default: two levels above this script)",
    )
    args = parser.parse_args()

    if args.case_dir is not None:
        case_dir = Path(args.case_dir).resolve()
    else:
        case_dir = (Path(__file__).resolve().parent.parent.parent)

    if not case_dir.is_dir():
        print(f"ERROR: case directory not found: {case_dir}", file=sys.stderr)
        return 1

    poly_dir  = case_dir / "constant" / "polyMesh"
    last_dir  = find_last_time_dir(case_dir)
    time_name = last_dir.name
    U_path    = last_dir / "U"

    print(f"\n{'='*60}")
    print(f"  compute_all_derived_fields.py")
    print(f"  Case      : {case_dir}")
    print(f"  Last time : {time_name}")
    print(f"{'='*60}\n")

    if not U_path.exists():
        print(f"ERROR: U field missing: {U_path}", file=sys.stderr)
        return 2

    # ── Step 1: read mesh ─────────────────────────────────────────────────────
    print("[1/5] Reading mesh ...")
    pts       = read_points(poly_dir)
    owner     = read_label_list(poly_dir / "owner",     "owner")
    neighbour = read_label_list(poly_dir / "neighbour", "neighbour")
    verts, n_int_note = read_faces(poly_dir)
    patches   = read_boundary(poly_dir)

    nCells = int(max(owner.max(), neighbour.max())) + 1
    nIF    = len(neighbour)
    if n_int_note is not None and n_int_note != nIF:
        print(f"  WARNING: faces note says nInternalFaces={n_int_note}, "
              f"neighbour has {nIF} — using {nIF}")
    print(f"\n  {nCells:,} cells  |  {len(owner):,} faces  |  {nIF:,} internal\n")

    # ── Step 2: face & cell geometry ──────────────────────────────────────────
    print("[2/5] Computing geometry ...")
    fc, Sf = compute_face_geometry(pts, verts)
    del pts, verts
    vols = compute_cell_volumes(owner, neighbour, fc, Sf, nCells)

    # ── Step 3: read U ────────────────────────────────────────────────────────
    print("\n[3/5] Reading U field ...")
    U = read_vector_internal(U_path)
    if len(U) != nCells:
        print(f"ERROR: U has {len(U)} values but mesh has {nCells} cells",
              file=sys.stderr)
        return 2

    # ── Step 4: velocity gradient ─────────────────────────────────────────────
    print("\n[4/5] Computing gradU (Gauss linear) ...")
    gradU = compute_gradU(U, owner, neighbour, fc, Sf, vols)
    del fc, Sf, U, vols

    # ── Step 5: all derived fields ────────────────────────────────────────────
    print("\n[5/5] Computing all derived fields ...")
    with _T("per-cell physics"):
        fields = compute_all_fields(gradU)
    del gradU

    # diagnostics
    print(f"  LiutexMag  max = {fields['LiutexMag'].max():.3e}")
    print(f"  vorticityMag max= {fields['vorticityMag'].max():.3e}")
    print(f"  Q          max = {fields['Q'].max():.3e}  min = {fields['Q'].min():.3e}")
    print(f"  WW/GG  range   = {fields['WW_over_GG'].min():.4f} – {fields['WW_over_GG'].max():.4f}")

    # ── write output ──────────────────────────────────────────────────────────
    print(f"\nWriting fields to {last_dir}/ ...")

    SCALAR_DIMS = {
        "LiutexMag":    "[0 0 -1 0 0 0 0]",
        "gg_rr":        "[0 0 0 0 0 0 0]",
        "gg_ps":        "[0 0 0 0 0 0 0]",
        "gg_ns":        "[0 0 0 0 0 0 0]",
        "gg_rs":        "[0 0 0 0 0 0 0]",
        "vorticityMag": "[0 0 -1 0 0 0 0]",
        "Q":            "[0 0 -2 0 0 0 0]",
        "WW_over_GG":   "[0 0 0 0 0 0 0]",
        "SS_over_GG":   "[0 0 0 0 0 0 0]",
    }
    VECTOR_DIMS = {
        "Liutex":    "[0 0 -1 0 0 0 0]",
        "vorticity": "[0 0 -1 0 0 0 0]",
    }

    write_order = [
        "Liutex", "LiutexMag",
        "gg_rr", "gg_ps", "gg_ns", "gg_rs",
        "vorticity", "vorticityMag",
        "Q", "WW_over_GG", "SS_over_GG",
    ]

    for fname in write_order:
        arr = fields[fname]
        out = last_dir / fname
        with _T(f"write {fname}"):
            if arr.ndim == 2:
                write_vector_field(out, fname, time_name, arr, patches, owner,
                                   dims=VECTOR_DIMS.get(fname, "[0 0 -1 0 0 0 0]"))
            else:
                write_scalar_field(out, fname, time_name, arr, patches, owner,
                                   dims=SCALAR_DIMS.get(fname, "[0 0 0 0 0 0 0]"))
        print(f"    → {out}")

    print(f"\n{'='*60}")
    print(f"SUCCESS — 11 fields written to {last_dir}/")
    print(f"  Liutex  LiutexMag  gg_rr  gg_ps  gg_ns  gg_rs")
    print(f"  vorticity  vorticityMag  Q  WW_over_GG  SS_over_GG")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
