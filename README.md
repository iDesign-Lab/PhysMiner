# PhysMiner

**An LLM-based framework for discovering novel flow physics from CFD results.**

PhysMiner combines classical CFD post-processing with large language models to automatically extract, classify, and interpret physical insights from OpenFOAM simulation data. It runs as a fully automated pipeline — from raw velocity fields to a reviewed PDF report — with no manual intervention.

---

![PhysMiner Demo](video/PhysMiner.mp4)

---

## Overview

PhysMiner consists of four sequential stages:

```
OpenFOAM case
      │
      ▼
┌─────────────────────────────┐
│  Stage 1 · Literature track │  Word clouds from downloaded papers
│  (literature-driven track/) │  → wordcloud_*.png
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 2 · Data track       │  Compute derived fields (offline, pure Python)
│  (data-driven track/)       │  Visualise with ParaView → *.png
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3 · TD Library       │  Flow fingerprinting + Jaccard matching
│  (triple decomposition      │  against historical knowledge base
│   library/)                 │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 4 · AI Discovery     │  /discover-physics  → report.pdf
│  (results/)                 │  /review-report     → comments.md
│                             │  Loops until report passes all criteria
└─────────────────────────────┘
```

---

## Repository Structure

```
PhysMiner/
│
├── literature-driven track/          # Stage 1 — literature analysis
│   ├── highlyCited/                  # Highly-cited papers (pre-2021)
│   │   └── wordcloud_papers.py
│   ├── overall/                      # All papers combined
│   │   └── wordcloud_papers.py
│   ├── recent/                       # Recent publications (2021–present)
│   │   └── wordcloud_papers.py
│   └── literature-driven track.py    # Stage 1 scheduler
│
├── data-driven track/                # Stage 2 — CFD data processing
│   ├── compute_all_derived_fields.py # Offline field computation (pure Python/NumPy)
│   ├── triple_decomposition_donut_chart.py
│   ├── profile_triple_decomp_yDirection.py
│   ├── profile_cauchy_stokes_yDirection.py
│   ├── slice_triple_decomposition_zDirection.py
│   ├── slice_cauchy_stokes_decomposition_zDirection.py
│   ├── slice_streamwise_velocity_zDirection.py
│   └── data-driven track.py          # Stage 2 scheduler
│
├── triple decomposition library/     # Stage 3 — flow knowledge base
│   ├── knowledge/                    # Stored historical fingerprints
│   │   ├── fingerprint_1/ … fingerprint_N/
│   │   └── fingerprint_number.png    # Classification reference table
│   ├── temp/                         # Working area for current case
│   ├── .claude/skills/               # Claude Code agent skills
│   │   ├── flow-fingerprint/         # Classifies flow type from images
│   │   └── flow-comparison/          # Compares current vs. historical case
│   ├── copy_triple_decomp_images.py
│   ├── jaccard_match.py
│   ├── save_results.py
│   └── triple decomposition library.py  # Stage 3 scheduler
│
└── results/                          # Stage 4 — AI physics discovery
    ├── .claude/skills/
    │   ├── discover-physics/         # LLM physics discovery + PDF generation
    │   │   ├── SKILL.md
    │   │   └── pdf_writer.py
    │   └── review-report/            # Automated quality review
    │       └── SKILL.md
    └── PhysMiner.py                  # Main pipeline entry point
```

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| Python 3.10+ | All computation and scripting |
| NumPy ≥ 1.20 | Offline field computation (Stage 2) |
| ParaView (pvpython) | Visualisation scripts in Stage 2 |
| wordcloud, matplotlib | Word cloud generation (Stage 1) |
| reportlab, Pillow | PDF report generation (Stage 4) |
| [Claude Code CLI](https://github.com/anthropics/claude-code) (`claude`) | AI agent skills (Stages 3 & 4) |
| OpenFOAM case data | Source CFD data (velocity field `U` + mesh) |

Install Python dependencies:

```bash
pip install numpy wordcloud matplotlib reportlab pillow
```

Install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
```

---

## Quick Start

### 1. Prepare your case

Place your OpenFOAM case in the directory **two levels above** `data-driven track/`:

```
<case_root>/
├── constant/polyMesh/   ← mesh files
├── <time>/U             ← velocity field
└── PhysMiner/           ← this repository
```

Place downloaded papers (PDF/TXT/MD) into:

```
literature-driven track/highlyCited/
literature-driven track/overall/
literature-driven track/recent/
```

### 2. Configure ParaView path

Edit the `PVPYTHON` path in `data-driven track/data-driven track.py`:

```python
PVPYTHON = Path("D:/software/paraView/bin/pvpython.exe")  # ← adjust to your install
```

### 3. Run the full pipeline

```bash
cd results
python PhysMiner.py
```

The pipeline runs all four stages automatically. Stages 4–5 (AI discovery + review) loop until the generated report passes all quality criteria or the retry limit is reached.

---

## Pipeline Stages

### Stage 1 — Literature-driven track

Reads PDF/TXT papers from three sub-folders and generates frequency word clouds that identify dominant physical keywords. The most prominent word (largest font) drives the AI analysis in Stage 4.

**Output:** `results/wordcloud_highlyCited.png`, `wordcloud_overall.png`, `wordcloud_recent.png`

---

### Stage 2 — Data-driven track

**`compute_all_derived_fields.py`** reads the OpenFOAM mesh and velocity field directly (no OpenFOAM installation required at runtime) and writes 11 derived fields back into the case time directory:

| Field | Type | Description |
|---|---|---|
| `Liutex` | volVectorField | Liutex vector (true rigid-body rotation axis × magnitude) |
| `LiutexMag` | volScalarField | Liutex magnitude \|R\| |
| `gg_rr` | volScalarField | Triple decomp — rigid rotation fraction |
| `gg_ps` | volScalarField | Triple decomp — pure shear fraction |
| `gg_ns` | volScalarField | Triple decomp — normal strain fraction |
| `gg_rs` | volScalarField | Triple decomp — shear–rotation interaction |
| `vorticity` | volVectorField | Vorticity vector ω |
| `vorticityMag` | volScalarField | \|ω\| |
| `Q` | volScalarField | Q-criterion |
| `WW_over_GG` | volScalarField | W:W / G:G (rotation fraction, binary) |
| `SS_over_GG` | volScalarField | S:S / G:G (strain fraction, binary) |

The remaining visualisation scripts (`slice_*.py`, `profile_*.py`, `triple_decomposition_donut_chart.py`) use **pvpython** (ParaView's embedded Python) to render 2D slices, wall-normal profiles, and a pie chart of the triple-decomposition energy budget.

**Output:** PNG images saved to `results/`

---

### Stage 3 — Triple decomposition library

Maintains a growing knowledge base of classified flow cases and uses AI + Jaccard similarity to match the current case against historical ones.

| Step | Script / Skill | Description |
|---|---|---|
| 1 | `copy_triple_decomp_images.py` | Copies current-case images to `temp/fingerprint/` |
| 2 | `/flow-fingerprint` (Claude Code skill) | Classifies flow across 10 dimensions → `fingerprint.txt` |
| 3 | `jaccard_match.py` | Finds the closest historical fingerprint in `knowledge/` |
| 4 | `/flow-comparison` (Claude Code skill) | Generates a structured comparison table |
| 5 | `save_results.py` | Saves fingerprint + images into `knowledge/` |

The 10 classification dimensions (Phase, Configuration, Viscous Effect, Rheology, Temporal, Separation, Thermal, Dimensionality, Boundary Motion, Compressibility) produce a compact integer vector, e.g. `[1, 3, 5, 7, 10, 12, 13, 18, 19, 23]`.

---

### Stage 4 — AI physics discovery (results/)

Two Claude Code agent skills run in a quality-control loop:

**`/discover-physics`**
- Extracts the dominant keyword from `wordcloud_overall.png`
- Reads all available flow field data and statistics
- Surveys downloaded literature (local files, no web search)
- Analyses all result images using the Triple-Decomposition × LLM framework
- Synthesises 3 novel cross-component physical findings
- Writes `discover_report_content.json` and renders `report.pdf`

**`/review-report`**
- Reads `report.pdf` in full
- Evaluates four criteria: internal logical consistency, dimensional consistency, physical consistency (gg values ∈ [0,1]), and literature grounding
- Writes `comments.md` with first line `pass` or `fail`

The main orchestrator (`PhysMiner.py`) re-runs both skills if the verdict is `fail`, up to 10 times.

**Output:** `results/report.pdf`, `results/comments.md`

---

## Output Files

| File | Description |
|---|---|
| `results/wordcloud_*.png` | Keyword frequency maps from three literature subsets |
| `results/slice_*.png` | 2D flow field slices (triple decomp, streamwise velocity, Cauchy-Stokes) |
| `results/profile_*.png` | Wall-normal profiles at multiple streamwise stations |
| `results/triple_decomposition_donut_chart.png` | Energy budget pie chart (gg_rr / gg_ps / gg_ns / gg_rs) |
| `results/discover_report_content.json` | Intermediate structured analysis data |
| `results/report.pdf` | Final A4 PDF physics discovery report |
| `results/comments.md` | Review verdict (`pass` / `fail` + per-criterion failure reasons) |

---

## How the AI Skills Work

The Claude Code skills (`.claude/skills/`) are Markdown instruction files that tell Claude what to do step by step. They are invoked via `claude -p /skill-name` and run autonomously without human interaction (`--dangerously-skip-permissions`).

Skills bundled with this repository:

| Skill | Location | Invoked by |
|---|---|---|
| `flow-fingerprint` | `triple decomposition library/.claude/skills/` | Stage 3 |
| `flow-comparison` | `triple decomposition library/.claude/skills/` | Stage 3 |
| `discover-physics` | `results/.claude/skills/` | Stage 4 |
| `review-report` | `results/.claude/skills/` | Stage 4 |

---

## Citation

If you use PhysMiner in your research, please cite:

```
@software{physminer2025,
  author = {Jiawei Chen},
  title  = {PhysMiner: An LLM-based framework for discovering flow physics},
  year   = {2025},
  url    = {https://github.com/chenjiaweiDr/PhysMiner}
}
```

---

## License

MIT License — see `LICENSE` for details.
