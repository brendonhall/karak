# Reproduction Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A public repository `brendonhall/nwa4587-karak-paper` with a package `nwa4587_paper` and a CLI `paper` that regenerates Tables 1 to 4, Figs 1 to 8 and the colormap round-trip numbers of the NWA 4587 paper from the HDF5 that `karak run --builtin paper` writes, with `--check` against committed expected numbers.

**Architecture:** One reader (`io.Run`) opens the karak HDF5 and resolves phases by name key. One constants module holds every colour, name, mapping, offset, window and threshold. Each result is one stage module with `run(run, out, data) -> dict`; the dict goes to `results/numbers/<stage>.json`, and `numbers.py` merges, renders LaTeX macros and checks against `expected/numbers.json`. A synthetic HDF5 builder (`synth.py`) serves the tests and the CI smoke run. Six PRs, one per part below.

**Tech Stack:** Python 3.12, numpy, scipy, scikit-image, scikit-learn, h5py, pyyaml, matplotlib (Agg), Pillow, imageio, karak (git dependency at commit `7ccd7a7`), pytest, uv, hatchling, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-09-reproduction-repo-design.md` (in the karak repository). The port source is the private paper repository `R = /home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26`, read-only.

## Global Constraints

- The new repository is created at Task 1.1 only after the user has said yes in that session. Until then nothing is pushed anywhere.
- Every change goes through a PR branched from an up-to-date `main` of the new repository; never commit to `main`. Commit messages end with the attribution lines the session reminder gives. The user merges.
- `R` (the paper repository) is read-only: never write a file there, never run `uv` or `python` with `R` as the working directory. The published `R/data/eds_pipeline.h5` is never opened with mode `a` or `w`. The karak reference file `/home/brendon/Projects/science/karak/output/nwa4587/item2/paperflow.h5` is opened read-only.
- karak is pinned in `pyproject.toml` as `karak = { git = "https://github.com/brendonhall/karak.git", rev = "7ccd7a7" }` under `[tool.uv.sources]`. karak needs Python >= 3.12 and its `compute_fingerprints` and `load_valid_mask` take keyword-only `accumulate` and `left_trim_px`.
- Phases are resolved by name key only (the HDF5 name before the first ` (`, see Task 1.4). No module outside `constants.py` and `synth.py` contains a label number, a colour, a phase name, an offset or a threshold.
- Every number a stage reports is a key in `numbers.REGISTRY`; a stage that reports an unlisted key fails its test.
- matplotlib uses the Agg backend (`matplotlib.use("Agg")` in `nwa4587_paper/__init__.py`); every figure is saved as `.pdf` and `.png` at 300 dpi with `bbox_inches="tight"` and a white face unless the task says otherwise.
- `data/` and `results/` are git-ignored. Tests use `tmp_path` and the synthetic builder; no test reads real data. Real-data steps are marked "local" and run only on the user's machine.
- Prose (README, PR bodies, docstrings) follows the user's style rules: short sentences, active voice, no em dashes, concrete numbers.

## Review Focus

1. An HDF5 whose names list lacks a key that a stage needs (for example a run without the `pyx` split, so no `Augite`) must fail with a message that names the missing key and lists the available names, not with a numpy error. Pinned in Task 1.4 (`test_phase_unknown_key`).
2. A TIMA raster whose decoded size does not overlap the karak map at the fixed offset must raise `ValueError("Translation produces no overlap")` before any metric runs. Pinned in Task 3.1 (`test_align_no_overlap`).
3. `labels_1x` when the hires map is one row or column short of `ratio * shape_2x` (odd source dimensions) must clip the repeated 2x map, not raise. Pinned in Task 1.4 (`test_labels_1x_clips`).
4. The lamellae stage when no grain reaches the analysis minimum must write a summary with zero analyzed grains and NaN statistics and exit 0, not divide by zero. Pinned in Task 5.1 (`test_lamellae_no_grains`).
5. `--check` when `expected/numbers.json` lacks a key that the fresh run reports, or reports one that the run lacks, must list the key as a miss rather than pass silently. Pinned in Task 2.1 (`test_check_missing_key`).

## File structure

```
pyproject.toml, uv.lock, README.md, CITATION.cff, LICENSE, LICENSE-FIGURES, .gitignore
.github/workflows/ci.yml
src/nwa4587_paper/
  __init__.py          # Agg backend, __version__
  constants.py         # Task 1.2
  synth.py             # Task 1.3: synthetic HDF5, PNGs and TIMA raster
  io.py                # Task 1.4: Run, Geometry, DataPaths, PNG loaders
  numbers.py           # Task 2.1: REGISTRY, write_stage, merge, render_tex, check
  cli.py               # Task 2.2 onwards: `paper` subcommands
  tables.py            # Task 2.2
  fingerprints.py      # Task 2.3
  tima_core.py         # Task 3.1 and 3.2: decode, align, combine, metrics
  tima.py              # Task 3.3: Figs 3, 4, 5, Tables 3, 4
  pyroxene.py          # Task 4.1
  xenotime.py          # Task 4.2
  workflow.py          # Task 4.3
  lamellae_core.py     # Task 5.1
  lamellae.py          # Task 5.2
  colormap.py          # Task 5.3
  data.py              # Task 6.1: `paper data`, `paper karak`
expected/h5_layout.json, expected/numbers.json
tests/conftest.py, tests/test_constants.py, tests/test_synth.py, tests/test_io.py,
tests/test_numbers.py, tests/test_tables.py, tests/test_fingerprints.py,
tests/test_tima_core.py, tests/test_tima.py, tests/test_pyroxene.py,
tests/test_xenotime.py, tests/test_workflow.py, tests/test_lamellae.py,
tests/test_colormap.py, tests/test_cli.py, tests/test_data.py
```

Name keys used throughout (the HDF5 name before the first ` (`; `Zn-bearing Phase` and `Weathering Assemblage` have no parenthesis and are whole):
`Ilmenite`, `Silica polymorph`, `Pyroxene`, `Plagioclase`, `Epoxy`, `Fe-Ti-Cr Spinel`, `Calcite`, `Apatite`, `Fe Oxyhydroxide`, `Zn-bearing Phase`, `Xenotime`, `Ferroan Olivine`, `Weathering Assemblage`, `Pigeonite`, `Augite`, `Merrillite`, `Chlorapatite`.

---

# Part 1: skeleton, constants, synthetic data, the reader (PR 1, branch `feat/skeleton-constants-io`)

### Task 1.1: repository skeleton, packaging, CI

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/nwa4587_paper/__init__.py`, `.github/workflows/ci.yml`, `README.md` (stub), `LICENSE`, `LICENSE-FIGURES`, `tests/test_package.py`

**Interfaces:**
- Produces: the importable package `nwa4587_paper` with `__version__ = "0.1.0"`; the console script `paper` bound to `nwa4587_paper.cli:main` (the module arrives in Task 2.2; until then the script is declared but not importable, which is fine for `uv sync`).

- [ ] **Step 1: Ask the user for consent, then create the repository**

Say: "Task 1.1 creates the public GitHub repository brendonhall/nwa4587-karak-paper (empty, MIT). May I create it now?" Stop until the answer is yes. Then:

```bash
mkdir -p ~/Projects/science/nwa4587-karak-paper && cd ~/Projects/science/nwa4587-karak-paper
git init -b main
gh repo create brendonhall/nwa4587-karak-paper --public --description "Reproduction of the NWA 4587 automated-mineralogy paper from karak output" --source . --remote origin
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "nwa4587-paper"
version = "0.1.0"
description = "Regenerates the figures, tables and numbers of the NWA 4587 karak paper"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12"
dependencies = [
  "karak",
  "numpy>=2.0",
  "scipy>=1.14",
  "scikit-image>=0.26",
  "scikit-learn>=1.6",
  "h5py>=3.12",
  "pyyaml>=6.0",
  "matplotlib>=3.9",
  "pillow>=10.0",
  "imageio>=2.35",
]

[project.scripts]
paper = "nwa4587_paper.cli:main"

[tool.uv.sources]
karak = { git = "https://github.com/brendonhall/karak.git", rev = "7ccd7a7" }

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/nwa4587_paper"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `.gitignore`, `__init__.py`, the licenses and the README stub**

`.gitignore`:
```
data/
results/
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.cache/
```

`src/nwa4587_paper/__init__.py`:
```python
"""Regenerates the NWA 4587 paper's main results from karak output."""
import matplotlib

matplotlib.use("Agg")

__version__ = "0.1.0"
```

`LICENSE`: the MIT text with `Copyright (c) 2026 Brendon Hall`. `LICENSE-FIGURES`: the first line `Everything under results/ (figures, tables, numbers) is licensed under CC BY 4.0.` followed by the CC BY 4.0 legal code URL `https://creativecommons.org/licenses/by/4.0/legalcode`.

`README.md` stub: title, one paragraph (what the repo regenerates, from which karak output, karak pinned at 7ccd7a7), a "Status: under construction" line. Task 6.2 writes the full README.

- [ ] **Step 4: Write the failing package test**

`tests/test_package.py`:
```python
import matplotlib


def test_package_imports_and_sets_agg():
    import nwa4587_paper

    assert nwa4587_paper.__version__ == "0.1.0"
    assert matplotlib.get_backend().lower() == "agg"
```

- [ ] **Step 5: Install and run**

Run: `uv sync --group dev && uv run pytest -q`
Expected: 1 passed. `uv sync` must resolve karak from git at 7ccd7a7; if it fails on the SSH form, use the `https://` URL above.

- [ ] **Step 6: Write the CI workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install 3.12
      - run: uv sync --group dev
      - run: uv run pytest -q
```

Task 2.4 adds the smoke job.

- [ ] **Step 7: Commit, push, open PR 1 (draft; the later tasks of Part 1 go on the same branch)**

```bash
git add -A && git commit -m "chore: package skeleton, CI, licenses"
git switch -c feat/skeleton-constants-io
git push -u origin main && git push -u origin HEAD
gh pr create --draft --title "feat: skeleton, constants, synthetic data and the HDF5 reader" --body "Part 1 of the plan."
```

(The first commit lands on `main` only because the repository is empty; every later commit goes to the branch.)

### Task 1.2: `constants.py`

**Files:**
- Create: `src/nwa4587_paper/constants.py`
- Test: `tests/test_constants.py`

**Interfaces:**
- Produces: every name below, imported by later tasks exactly as spelled.

- [ ] **Step 1: Write the failing tests**

`tests/test_constants.py`:
```python
from nwa4587_paper import constants as C

NAMES = [
    "Ilmenite (FeTiO₃)", "Silica polymorph (SiO₂)", "Pyroxene (Ca,Mg,Fe)SiO₃",
    "Plagioclase (CaAl₂Si₂O₈)", "Epoxy", "Fe-Ti-Cr Spinel (Fe₂TiO₄)", "Calcite (CaCO₃)",
    "Apatite (Ca₅(PO₄)₃(OH,Cl,F))", "Fe Oxyhydroxide (FeOOH)", "Zn-bearing Phase",
    "Xenotime (YPO₄)", "Ferroan Olivine ((Fe,Mg)₂SiO₄)", "Weathering Assemblage",
    "Pigeonite (low-Ca pyroxene)", "Augite (high-Ca pyroxene)",
    "Merrillite (Ca₉NaMg(PO₄)₇)", "Chlorapatite (Ca₅(PO₄)₃(Cl,F,OH))",
]
KEYS = {n.split(" (")[0] for n in NAMES}


def test_every_palette_key_is_a_phase_key():
    assert set(C.PALETTE) <= KEYS


def test_every_mapping_key_is_a_phase_key_and_index_exists():
    indices = {p.index for p in C.TIMA_PHASES}
    for mapping in (C.TIMA_MAPPING_SPLIT, C.TIMA_MAPPING_COMBINED):
        assert set(mapping) <= KEYS
        for tima in mapping.values():
            assert set(tima) <= indices


def test_display_and_exclusion_keys():
    assert set(C.DISPLAY_NAMES) <= KEYS
    assert set(C.FIG4_SHORT_NAMES) <= KEYS
    assert set(C.EXCLUDED_KARAK_KEYS) <= KEYS
    assert set(C.FINGERPRINT_EXCLUDED_KEYS) <= KEYS
    assert set(C.LEGEND_ORDER_SPLIT) <= KEYS and set(C.LEGEND_ORDER_COMBINED) <= KEYS


def test_tima_table_has_14_unique_colours():
    assert len(C.TIMA_PHASES) == 14
    assert len({p.rgb for p in C.TIMA_PHASES}) == 14
    assert [p.index for p in C.TIMA_PHASES] == list(range(14))


def test_offsets_and_geometry():
    assert (C.TIMA_DX, C.TIMA_DY) == (1919, 1051)
    assert C.PIXEL_SIZE_UM_1X == 3.0
    assert C.FIG7_CROP_1X == (6061, 6861, 3570, 4370)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_constants.py -q`
Expected: ImportError on `constants`.

- [ ] **Step 3: Write `constants.py`**

```python
"""Every colour, name, mapping, offset, window and threshold of the paper.

Keyed by phase key: the HDF5 phase name before the first " (".
Source: the paper repository's poster/make_phase_map_1x.py (palette),
analysis/tima_comparison_1x.py and tima_comparison.py (TIMA), the figure
scripts (windows, settings) and data/sample.yaml (pixel size).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

# ----- palette (poster/make_phase_map_1x.py L39-58) -----
PALETTE: dict[str, str] = {
    "Ilmenite": "#8B4513",
    "Silica polymorph": "#F0E442",
    "Plagioclase": "#56B4E9",
    "Epoxy": "#999999",
    "Fe-Ti-Cr Spinel": "#D55E00",
    "Calcite": "#CC79A7",
    "Apatite": "#0072B2",          # pre-split panels only (Fig 1 d)
    "Fe Oxyhydroxide": "#BBBBBB",
    "Zn-bearing Phase": "#882255",
    "Xenotime": "#00FF00",
    "Ferroan Olivine": "#44AA99",
    "Weathering Assemblage": "#332288",
    "Augite": "#E6AB02",
    "Pigeonite": "#2B83BA",
    "Merrillite": "#1B9E77",
    "Chlorapatite": "#E7298A",
    "Pyroxene": "#666666",         # unresolved pyroxene, pre-split panels
}
UNASSIGNED_COLOR = "#000000"
OVERLAY_ALPHA = 0.75
PYROXENE_UNDIFF_COLOR = "#66C266"   # Fig 7 panel (a)
NYQUIST_COLOR = "#CC3333"

# ----- display names (make_fig1_workflow.py L76-92, make_fig5 _SHORT_NAMES) -----
DISPLAY_NAMES: dict[str, str] = {
    "Ilmenite": "Ilmenite",
    "Silica polymorph": "Silica polymorph",
    "Plagioclase": "Plagioclase",
    "Epoxy": "Epoxy",
    "Fe-Ti-Cr Spinel": "Fe–Ti–Cr spinel",
    "Calcite": "Calcite",
    "Apatite": "Apatite",
    "Fe Oxyhydroxide": "Fe oxyhydroxide",
    "Zn-bearing Phase": "Zn-bearing phase",
    "Xenotime": "Xenotime",
    "Ferroan Olivine": "Ferroan olivine",
    "Weathering Assemblage": "Weathering assemblage",
    "Augite": "Augite (high-Ca px)",
    "Pigeonite": "Pigeonite (low-Ca px)",
    "Merrillite": "Merrillite",
    "Chlorapatite": "Chlorapatite",
    "Pyroxene": "Pyroxene (unresolved)",
}
FIG4_SHORT_NAMES: dict[str, str] = {
    "Fe-Ti-Cr Spinel": "Spinel",
    "Ferroan Olivine": "Olivine",
    "Silica polymorph": "Silica",
    "Fe Oxyhydroxide": "Fe-OOH",
}
TABLE4_NAMES: dict[str, str] = {"Pigeonite": "Pigeonite (low-Ca)", "Augite": "Augite (high-Ca)"}

# ----- TIMA colour table (analysis/tima_comparison.py L53-68) -----
class TimaPhase(NamedTuple):
    index: int
    name: str
    rgb: tuple[int, int, int]


TIMA_PHASES: tuple[TimaPhase, ...] = (
    TimaPhase(0, "Plagioclase", (170, 0, 127)),
    TimaPhase(1, "Pyroxene low-Ca", (0, 255, 255)),
    TimaPhase(2, "Pyroxene high-Ca", (103, 70, 156)),
    TimaPhase(3, "Ilmenite Cr", (128, 170, 10)),
    TimaPhase(4, "Olivine+Ilmenite", (0, 85, 0)),
    TimaPhase(5, "Pyroxene Bi", (170, 85, 255)),
    TimaPhase(6, "Apatite", (170, 170, 0)),
    TimaPhase(7, "Calcite", (255, 170, 255)),
    TimaPhase(8, "Quartz", (85, 85, 255)),
    TimaPhase(9, "Unknown1", (85, 0, 255)),
    TimaPhase(10, "Mixture", (100, 100, 150)),
    TimaPhase(11, "Troilite", (255, 0, 0)),
    TimaPhase(12, "Unknown2", (190, 255, 242)),
    TimaPhase(13, "Hematite/Magnetite", (0, 85, 255)),
)
TIMA_NAMES: dict[int, str] = {p.index: p.name for p in TIMA_PHASES}
TIMA_COMBINED_NAME = "Pyroxene"          # TIMA 1 after combining 1, 2 and 5

# ----- TIMA to karak mapping (analysis/tima_comparison_1x.py L78-118) -----
TIMA_MAPPING_SPLIT: dict[str, tuple[int, ...]] = {
    "Plagioclase": (0,),
    "Augite": (2,),
    "Pigeonite": (1,),
    "Ferroan Olivine": (4,),
    "Ilmenite": (3, 4),
    "Fe-Ti-Cr Spinel": (3,),
    "Merrillite": (6,),
    "Chlorapatite": (6,),
    "Calcite": (7,),
    "Silica polymorph": (8,),
    "Fe Oxyhydroxide": (13,),
}
TIMA_MAPPING_COMBINED: dict[str, tuple[int, ...]] = {
    k: v for k, v in TIMA_MAPPING_SPLIT.items() if k != "Augite"
}
TIMA_MAPPING_NOTES: dict[str, str] = {
    "Plagioclase": "Direct match",
    "Augite": "Direct match",
    "Pigeonite": "Direct match",
    "Ferroan Olivine": "TIMA 4 = Olivine+Ilmenite (shared color)",
    "Ilmenite": "TIMA splits: Ilmenite Cr (3) + Olivine+Ilmenite (4)",
    "Fe-Ti-Cr Spinel": "Possible overlap with TIMA Ilmenite Cr",
    "Merrillite": "TIMA lumps phosphates into Apatite",
    "Chlorapatite": "TIMA lumps phosphates into Apatite",
    "Calcite": "Direct match",
    "Silica polymorph": "Direct match (Quartz)",
    "Fe Oxyhydroxide": "Possible match with Hematite/Magnetite",
}
COMBINED_PYROXENE_NOTE = "Augite + Pigeonite merged; TIMA low-Ca + high-Ca + Bi merged"
# in combined mode: karak Augite -> Pigeonite's label; TIMA 2 and 5 -> 1
TIMA_COMBINE_INTO = 1
TIMA_COMBINE_FROM = (2, 5)

EXCLUDED_KARAK_KEYS: tuple[str, ...] = ("Zn-bearing Phase", "Xenotime", "Weathering Assemblage")
EXCLUDED_TIMA_SPLIT: tuple[int, ...] = (5, 9, 10, 11, 12)
EXCLUDED_TIMA_COMBINED: tuple[int, ...] = (9, 10, 11, 12)
# -1 (unassigned / background) is always excluded in code.

# ----- registration (tima_comparison_1x.py L67-71) -----
TIMA_DX = 1919
TIMA_DY = 1051
TIMA_VERIFIED_PAIR = ((2160, 272), (4079, 1323))   # karak (x, y) <-> TIMA (x, y)
TIMA_LEGEND_TRIM_ROWS = 1000
TIMA_COLOR_DISTANCE_MAX = 50.0
TIMA_ACHROMATIC_RANGE = 30
TIMA_ACHROMATIC_MIN = 20

# ----- Fig 3 and Fig 4 colours (make_fig3_fig4.py L80-91, L413-414) -----
MATCHED_COLORS: dict[str, tuple[float, float, float]] = {
    "Plagioclase": (0.698, 0.133, 0.133),
    "Pigeonite": (0.357, 0.510, 0.671),      # combined mode: the merged pyroxene
    "Ferroan Olivine": (0.467, 0.533, 0.600),
    "Ilmenite": (0.133, 0.133, 0.400),
    "Fe-Ti-Cr Spinel": (0.400, 0.400, 0.133),
    "Merrillite": (0.106, 0.620, 0.467),
    "Chlorapatite": (0.906, 0.161, 0.541),
    "Calcite": (0.600, 0.800, 0.400),
    "Silica polymorph": (0.800, 0.600, 0.200),
    "Fe Oxyhydroxide": (0.200, 0.200, 0.200),
}
MATCHED_COLORS_SPLIT_OVERRIDE: dict[str, tuple[float, float, float]] = {
    "Augite": (0.902, 0.671, 0.008),
    "Pigeonite": (0.169, 0.514, 0.729),
}
LEGEND_ORDER_SPLIT: tuple[str, ...] = (
    "Plagioclase", "Augite", "Pigeonite", "Ilmenite", "Fe-Ti-Cr Spinel", "Ferroan Olivine",
    "Merrillite", "Chlorapatite", "Calcite", "Silica polymorph", "Fe Oxyhydroxide",
)
LEGEND_ORDER_COMBINED: tuple[str, ...] = tuple(k for k in LEGEND_ORDER_SPLIT if k != "Augite")
TIMA_BACKGROUND = (0.92, 0.92, 0.92)
AGREE_COLOR = (0.25, 0.68, 0.25)
DISAGREE_COLOR = (0.75, 0.20, 0.20)
EXCLUDED_COLOR = (0.88, 0.88, 0.88)
FIG5_HIGH_CA = (0.902, 0.671, 0.008)
FIG5_LOW_CA = (0.169, 0.514, 0.729)
FIG5_NON_PYX = (0.93, 0.93, 0.93)

# ----- geometry -----
PIXEL_SIZE_UM_1X = 3.0          # data/sample.yaml acquisition.pixel_size_um

# ----- fixed windows -----
FIG7_CROP_1X = (6061, 6861, 3570, 4370)     # rows r0, r1, cols c0, c1 at 1x
FIG5_WINDOW_SIZE = 520
FIG5_WINDOW_STRIDE = 100
FIG5_WINDOW_XY: tuple[int, int] | None = None   # None: search and record; Task 6.2 freezes it
XENOTIME_PAD_PX_2X = 80
FIG7_SUBSAMPLE_N = 80_000
FIG7_SUBSAMPLE_SEED = 42
FIG1_PNG_STRIDE = 16
FIG1_H5_STRIDE = 8
FIG1_ELEMENT = "Ca"


@dataclass(frozen=True)
class LamellaeSettings:
    min_grain_px: int = 500
    min_analysis_px: int = 2000
    closing_iterations: int = 2
    tensor_sigma: float = 1.5
    coherence_min: float = 0.2
    n_profiles: int = 20
    nyquist_px: float = 4.0


LAMELLAE = LamellaeSettings()


@dataclass(frozen=True)
class ColormapSettings:
    lut_n: int = 4096
    n_test: int = 100_000
    cmaps: tuple[str, ...] = ("jet", "viridis")


COLORMAP = ColormapSettings()

RARE_PHASE_THRESHOLD_PX = 500_000
FINGERPRINT_EXCLUDED_KEYS: tuple[str, ...] = ("Epoxy", "Weathering Assemblage")

# ----- Table 1: flow node params shown (label, node id, param, format) -----
TABLE1_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("Header trim (px)", "src", "header_trim_px", "{}"),
    ("Downsample factor", "src", "downsample_factor", "{}"),
    ("Minimum object size (px)", "msk", "min_object_size", "{}"),
    ("Bilateral sigma spatial (px)", "dn", "sigma_spatial", "{}"),
    ("Bilateral sigma colour", "dn", "sigma_color", "{}"),
    ("Normalisation", "nrm", "method", "{}"),
    ("PCA variance kept", "pca", "variance_threshold", "{}"),
    ("Tile size (px)", "hdb", "tile_size", "{}"),
    ("Tile merge cosine threshold", "hdb", "merge_threshold", "{}"),
    ("Minimum clusters per tile", "hdb", "min_clusters_per_tile", "{}"),
    ("HDBSCAN min cluster size", "hdb", "min_cluster_size", "{}"),
    ("HDBSCAN min samples", "hdb", "min_samples", "{}"),
    ("Rare-phase min cluster size", "rare", "min_cluster_size", "{}"),
    ("Rare-phase subsample", "rare", "subsample_n", "{:,}"),
    ("kNN neighbours", "knn", "k", "{}"),
    ("Olivine rule", "oliv", "rule", "{}"),
    ("Weathering GMM features", "weath", "features", "{}"),
    ("Pyroxene split feature (1x)", "pyx", "feature", "{}"),
    ("Phosphate GMM features", "phos", "features", "{}"),
)

# ----- Table 3 (main.tex tbl:correspondence) -----
TABLE3_ROWS: tuple[tuple[str, str, str], ...] = (
    ("Plagioclase", "Plagioclase", "Direct match"),
    ("Augite", "Pyroxene high-Ca", "Differing lamellae cut-off"),
    ("Pigeonite", "Pyroxene low-Ca", "Differing lamellae cut-off"),
    ("Ferroan Olivine", "Olivine+Ilmenite", "Shared TIMA colour"),
    ("Ilmenite", "Ilmenite Cr + Olivine+Ilmenite", "Pipeline merges"),
    ("Fe-Ti-Cr Spinel", "Ilmenite Cr?", "Possible overlap"),
    ("Merrillite", "Apatite", "TIMA lumps phosphates"),
    ("Chlorapatite", "Apatite", "TIMA lumps phosphates"),
    ("Calcite", "Calcite", "Direct match"),
    ("Silica polymorph", "Quartz", "Direct match"),
    ("Xenotime", "---", "Not detected by TIMA"),
    ("Zn-bearing Phase", "---", "Not detected by TIMA"),
    ("Fe Oxyhydroxide", "Hematite/Magnetite?", "Uncertain"),
)
```

If a Table 1 node param name (`min_clusters_per_tile`, `k`, `tile_size`) differs in `paper.json`, read `src/karak/flow/flows/paper.json` in the karak checkout and use the real names; the test in Task 2.2 checks every row resolves.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_constants.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nwa4587_paper/constants.py tests/test_constants.py
git commit -m "feat: constants module keyed by phase name"
```

### Task 1.3: `synth.py`, the synthetic HDF5, PNGs and TIMA raster; the layout listing

**Files:**
- Create: `src/nwa4587_paper/synth.py`, `expected/h5_layout.json`, `tests/conftest.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces: `synth.write_synthetic_h5(path: Path) -> Path` (a 64 by 48 map at 2x, a 128 by 96 hires map); `synth.write_synthetic_eds(dir: Path) -> Path` (jet PNGs `SYN-Ca.png`, `SYN-Mg.png`, `SYN-SEM.png` of 228 by 96 pixels: 100 banner rows plus 128 image rows, plus `mask/Valid_mask.csv`); `synth.write_synthetic_tima(path: Path) -> Path` (a 16-bit RGB PNG of 1200 rows by 200 columns: the 128 by 96 karak map painted with TIMA colours at offset (1919 - 1900, 1051 - 1000) = (19, 51) inside a 200 by 200 canvas, plus a 1000-row legend strip); `synth.SYNTH_NAMES: dict[int, str]`; `synth.SYNTH_HISTORY: list[dict]`; `synth.SYNTH_FLOW_YAML: str`.
- Pytest fixtures in `conftest.py`: `synth_h5`, `synth_eds`, `synth_tima` (session scope, built once under `tmp_path_factory`), `synth_data` -> `io.DataPaths` (Task 1.4; until then the fixture is not used).

Design of the synthetic sample (2x, 64 rows by 48 columns, mineral mask = everything except a 6-px border):
- label 3 Plagioclase: rows 6 to 30; label 2 region (pyroxene) rows 30 to 58 columns 6 to 30; label 4 Epoxy: columns 30 to 42, rows 30 to 58; label 10 Xenotime: a 4 by 4 square at rows 40 to 44, columns 36 to 40 (inside the epoxy block); label 7 Apatite: rows 50 to 58, columns 42 to 46 is outside the mask, so use rows 8 to 12, columns 36 to 44 instead.
- History (same shape as the real file): `split_threshold` oliv parent 2 -> 11 (rows 54 to 58 of the pyroxene block); `split_gmm` weath parent 2 -> 12 (rows 50 to 54); `split_hires` pyx parent 2 -> 13 Pigeonite, 14 Augite with `ratio` 2, `n_pixels_hires`, `component_means` {"14": {"Ca/(Ca+Mg)": 0.53}, "13": {"Ca/(Ca+Mg)": 0.31}}; `split_gmm` phos parent 7 -> 15, 16 (columns 36 to 40 and 40 to 44).
- Final 2x labels: 3, 4, 10, 11, 12, 13, 14, 15, 16 (no 2, no 7). Hires labels: inside the upscaled pyroxene region, stripes 4 px wide alternate 13 and 14 along the rows; -1 elsewhere; the top-left 2 by 2 block of the region is -1 (undefined feature).
- Denoised cube channels, in this order: `["Ca", "Mg", "Y", "P"]` plus whatever Table 1 needs nothing else. Ca is 0.6 inside Augite stripes (at 2x: Augite majority pixels), 0.2 in Pigeonite, 0.3 in plagioclase, 0 elsewhere; Mg the complement inside pyroxene; Y and P are 0.9 inside the xenotime square and 0.02 elsewhere. BSE is a gradient 0 to 1 by row.
- `raw/` has the same four channels at the same values; `normalized/` is skipped (the real file has it, so `h5_layout.json` lists it under `optional`).
- Root attribute `pipeline_config` is `SYNTH_FLOW_YAML`: the real `paper.json` node list converted to YAML with the params of Table 1 present (copy `src/karak/flow/flows/paper.json` from the karak checkout at 7ccd7a7 into `src/nwa4587_paper/data/paper_flow.json` and dump it with `yaml.safe_dump`; the file ships in the wheel).
- `clusters` attributes: `mineral_names` (JSON), `cluster_stats` (JSON with `n_total`), `cluster_config` (YAML with `refinement_nodes` = the four split nodes `oliv`, `weath`, `pyx`, `phos` plus `names`), `n_clusters`, `n_mineral_pixels`.

- [ ] **Step 1: Write the layout listing from the reference file (local, once)**

Run in the karak checkout (read-only on the file):
```bash
uv run python - <<'PY'
import h5py, json
out = {"datasets": {}, "attrs": {}, "optional_groups": ["normalized", "clusters/tiled"]}
with h5py.File("output/nwa4587/item2/paperflow.h5", "r") as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out["datasets"][name] = {"ndim": obj.ndim, "dtype": str(obj.dtype)}
        out["attrs"][name] = sorted(obj.attrs)
    f.visititems(visit)
    out["attrs"]["/"] = sorted(f.attrs)
json.dump(out, open("/tmp/h5_layout.json", "w"), indent=1, sort_keys=True)
PY
```
Copy `/tmp/h5_layout.json` to `expected/h5_layout.json` in the new repository. Drop the per-element `raw/<El>` entries except `raw/Ca` and `raw/Mg` (the synthetic file has fewer elements) and note that in the README of Task 6.2.

- [ ] **Step 2: Write the failing tests**

`tests/test_synth.py`:
```python
import json
import h5py
import numpy as np
from PIL import Image

from nwa4587_paper import synth


def test_layout_matches_reference_listing(synth_h5):
    layout = json.load(open("expected/h5_layout.json"))
    with h5py.File(synth_h5, "r") as f:
        for name, spec in layout["datasets"].items():
            if any(name.startswith(g) for g in layout["optional_groups"]):
                continue
            assert name in f, name
            assert f[name].ndim == spec["ndim"], name
        for name, keys in layout["attrs"].items():
            if name != "/" and any(name.startswith(g) for g in layout["optional_groups"]):
                continue
            obj = f if name == "/" else f[name]
            assert set(keys) <= set(obj.attrs), (name, set(keys) - set(obj.attrs))


def test_synthetic_labels_and_history(synth_h5):
    with h5py.File(synth_h5, "r") as f:
        labels = f["clusters/cleaned_labels"][:]
        names = json.loads(f["clusters"].attrs["mineral_names"])
        history = json.loads(f["clusters/subclustering"].attrs["history"])
        hires = f["clusters/hires/labels"]
        assert set(np.unique(labels)) == {3, 4, 10, 11, 12, 13, 14, 15, 16}
        assert names["14"].startswith("Augite") and names["13"].startswith("Pigeonite")
        assert [h["stage"] for h in history] == ["split_threshold", "split_gmm", "split_hires", "split_gmm"]
        assert hires.shape == (128, 96) and hires.attrs["ratio"] == 2
        assert set(np.unique(hires[:])) == {-1, 13, 14}


def test_synthetic_pngs_and_tima(synth_eds, synth_tima):
    ca = np.asarray(Image.open(synth_eds / "SYN-Ca.png"))
    assert ca.shape == (228, 96, 3)
    assert (synth_eds / "mask" / "Valid_mask.csv").exists()
    tima = Image.open(synth_tima)
    assert tima.size == (200, 1200)
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_synth.py -q`
Expected: fixture `synth_h5` not found.

- [ ] **Step 4: Write `conftest.py` and `synth.py`**

`tests/conftest.py`:
```python
import pytest

from nwa4587_paper import synth


@pytest.fixture(scope="session")
def synth_h5(tmp_path_factory):
    return synth.write_synthetic_h5(tmp_path_factory.mktemp("h5") / "paperflow.h5")


@pytest.fixture(scope="session")
def synth_eds(tmp_path_factory):
    return synth.write_synthetic_eds(tmp_path_factory.mktemp("eds"))


@pytest.fixture(scope="session")
def synth_tima(tmp_path_factory):
    return synth.write_synthetic_tima(tmp_path_factory.mktemp("tima") / "phases.png")
```

`src/nwa4587_paper/synth.py` (the builder; sizes and labels as designed above):
```python
"""Synthetic karak-layout data for tests and the CI smoke run."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import h5py
import numpy as np
import yaml
from matplotlib import colormaps
from PIL import Image

from . import constants as C

H2, W2 = 64, 48
RATIO = 2
ELEMENTS = ["Ca", "Mg", "Y", "P"]
SYNTH_NAMES: dict[int, str] = {
    0: "Ilmenite (FeTiO₃)", 1: "Silica polymorph (SiO₂)", 2: "Pyroxene (Ca,Mg,Fe)SiO₃",
    3: "Plagioclase (CaAl₂Si₂O₈)", 4: "Epoxy", 5: "Fe-Ti-Cr Spinel (Fe₂TiO₄)",
    6: "Calcite (CaCO₃)", 7: "Apatite (Ca₅(PO₄)₃(OH,Cl,F))", 8: "Fe Oxyhydroxide (FeOOH)",
    9: "Zn-bearing Phase", 10: "Xenotime (YPO₄)", 11: "Ferroan Olivine ((Fe,Mg)₂SiO₄)",
    12: "Weathering Assemblage", 13: "Pigeonite (low-Ca pyroxene)",
    14: "Augite (high-Ca pyroxene)", 15: "Merrillite (Ca₉NaMg(PO₄)₇)",
    16: "Chlorapatite (Ca₅(PO₄)₃(Cl,F,OH))",
}
SYNTH_FLOW_YAML = yaml.safe_dump(
    json.loads(files("nwa4587_paper").joinpath("data/paper_flow.json").read_text()),
    sort_keys=False)


def _base_labels() -> tuple[np.ndarray, np.ndarray]:
    """2x label image before any split (labels 2, 3, 4, 7, 10) and the mineral mask."""
    img = np.full((H2, W2), -1, np.int32)
    mask = np.zeros((H2, W2), bool)
    mask[6:58, 6:42] = True
    img[6:30, 6:42] = 3
    img[30:58, 6:30] = 2
    img[30:58, 30:42] = 4
    img[40:44, 36:40] = 10
    img[8:12, 36:44] = 7
    mask[8:12, 42:44] = True
    img[~mask] = -1
    return img, mask


def _hires_labels(pre: np.ndarray) -> np.ndarray:
    region = np.repeat(np.repeat(pre == 2, RATIO, 0), RATIO, 1)
    hires = np.full(region.shape, -1, np.int16)
    rows = np.arange(region.shape[0])[:, None]
    stripe = ((rows // 4) % 2 == 0)
    hires[region & np.broadcast_to(stripe, region.shape)] = 14      # Augite
    hires[region & ~np.broadcast_to(stripe, region.shape)] = 13     # Pigeonite
    r0, c0 = 60, 12                                                  # top-left of the region at 1x
    hires[r0:r0 + 2, c0:c0 + 2] = -1
    return hires


def final_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Returns (final 2x labels, mineral mask, hires labels, history)."""
    pre, mask = _base_labels()
    img = pre.copy()
    img[54:58, 6:30] = 11
    img[50:54, 6:30] = 12
    hires = _hires_labels(pre)
    hires[np.repeat(np.repeat(img != 2, RATIO, 0), RATIO, 1)] = -1
    pyx = img == 2
    for r in range(H2):
        for c in range(W2):
            if pyx[r, c]:
                block = hires[2 * r:2 * r + 2, 2 * c:2 * c + 2]
                vals = block[block >= 0]
                img[r, c] = 2 if vals.size == 0 else int(np.bincount(vals).argmax())
    img[8:12, 36:40] = 15
    img[8:12, 40:44] = 16
    n = lambda lab, arr=img: int((arr == lab).sum())
    history = [
        {"stage": "split_threshold", "parent": 2, "new_labels": [11], "names": {"11": SYNTH_NAMES[11]},
         "n_pixels": {"11": n(11)}, "method": "Fe-K > 0.6 & Ca < 0.10", "note": "synthetic"},
        {"stage": "split_gmm", "parent": 2, "new_labels": [12], "names": {"12": SYNTH_NAMES[12]},
         "n_pixels": {"12": n(12)}, "method": "GMM 2-component on Ca/(Ca+Mg)+BSE+Fe-K", "note": "synthetic",
         "component_means": {"2": {"Ca/(Ca+Mg)": 0.43}, "12": {"Ca/(Ca+Mg)": 0.62}}},
        {"stage": "split_hires", "parent": 2, "new_labels": [13, 14],
         "names": {"13": SYNTH_NAMES[13], "14": SYNTH_NAMES[14]},
         "n_pixels": {"13": n(13), "14": n(14)}, "method": "GMM 2-component on Ca/(Ca+Mg) at 2x resolution",
         "note": "synthetic", "parents": [2], "ratio": RATIO,
         "n_pixels_hires": {"13": n(13, hires), "14": n(14, hires)}, "n_undefined": 4,
         "component_means": {"14": {"Ca/(Ca+Mg)": 0.53}, "13": {"Ca/(Ca+Mg)": 0.31}}},
        {"stage": "split_gmm", "parent": 7, "new_labels": [15, 16],
         "names": {"15": SYNTH_NAMES[15], "16": SYNTH_NAMES[16]},
         "n_pixels": {"15": n(15), "16": n(16)}, "method": "GMM 2-component on Cl+Na+Mg+F", "note": "synthetic"},
    ]
    return img, mask, hires, history


def _cube(img: np.ndarray, hires: np.ndarray) -> np.ndarray:
    cube = np.zeros((H2, W2, len(ELEMENTS)), np.float32)
    ca, mg, y, p = (cube[:, :, i] for i in range(4))
    ca[img == 3] = 0.3
    ca[img == 14] = 0.6
    ca[img == 13] = 0.2
    mg[img == 14] = 0.4
    mg[img == 13] = 0.8
    y[:] = 0.02
    p[:] = 0.02
    y[img == 10] = 0.9
    p[img == 10] = 0.9
    return cube


def write_synthetic_h5(path: Path) -> Path:
    img, mask, hires, history = final_labels()
    rows, cols = np.nonzero(mask)
    labels = img[rows, cols].astype(np.int32)
    cube = _cube(img, hires)
    bse = np.linspace(0, 1, H2, dtype=np.float32)[:, None].repeat(W2, 1)
    names = {k: v for k, v in SYNTH_NAMES.items() if k not in (2, 7)}
    counts = {str(int(k)): {"n_pixels": int((labels == k).sum()),
                            "pct": float(100 * (labels == k).sum() / labels.size),
                            "mean_prob": 1.0} for k in np.unique(labels)}
    flow = yaml.safe_load(SYNTH_FLOW_YAML)
    refinement_types = ("name_phases", "split_threshold", "split_gmm", "split_hires")
    refinement_nodes = [{"id": n["id"], "type": n["type"], "params": n["params"]}
                        for n in flow["nodes"] if n["type"] in refinement_types]
    with h5py.File(path, "w") as f:
        f.attrs["pipeline_config"] = SYNTH_FLOW_YAML
        f.attrs["pipeline_version"] = "synthetic"
        f.attrs["created"] = "2026-01-01T00:00:00+00:00"
        f.attrs["library_versions"] = "{}"
        f.attrs["platform"] = "synthetic"
        f.attrs["python_version"] = "3.12"
        g = f.create_group("raw")
        g.attrs.update({"element_names": json.dumps(ELEMENTS), "height": H2, "width": W2,
                        "n_elements": len(ELEMENTS)})
        for i, el in enumerate(ELEMENTS):
            g.create_dataset(el, data=cube[:, :, i])
        b = f.create_group("bse")
        b.attrs.update({"downsample_factor": RATIO, "original_shape": [H2, W2], "stored_shape": [H2, W2]})
        b.create_dataset("image", data=bse)
        m = f.create_group("masks")
        m.attrs.update({"coverage_of_valid_pct": 100.0, "coverage_pct": float(100 * mask.mean()),
                        "mask_config": "min_object_size: 100\n", "n_connected_components": 1,
                        "n_mineral_pixels": int(mask.sum()), "n_total_pixels": int(mask.size),
                        "n_valid_pixels": int(mask.sum())})
        m.create_dataset("mineral", data=mask)
        m.create_dataset("valid", data=mask)
        d = f.create_group("denoised")
        d.attrs.update({"element_order": json.dumps(ELEMENTS), "method": "bilateral",
                        "sigma_color": "auto", "sigma_spatial": 1.0})
        d.create_dataset("cube", data=cube)
        c = f.create_group("clusters")
        c.attrs.update({
            "mineral_names": json.dumps({str(k): v for k, v in names.items()}),
            "cluster_stats": json.dumps({"n_clusters": len(counts), "n_noise": 0, "noise_pct": 0.0,
                                         "n_total": int(labels.size), "clusters": counts}),
            "cluster_config": yaml.safe_dump({"strategy": "tiled", "refinement_nodes": refinement_nodes},
                                             sort_keys=False),
            "clustering_strategy": "tiled", "n_clusters": len(counts), "n_mineral_pixels": int(labels.size),
            "n_noise": 0, "n_pca_components_used": 3, "noise_pct": 0.0,
        })
        for k, v in names.items():
            c.attrs[f"cluster_{k}_name"] = v
        c.create_dataset("cleaned_labels", data=labels)
        c.create_dataset("raw_labels", data=labels)
        c.create_dataset("probabilities", data=np.ones(labels.size, np.float32))
        c.create_dataset("mineral_indices", data=np.stack([rows, cols], 1).astype(np.int32))
        c.create_dataset("pca_variance_ratio", data=np.array([0.6, 0.3, 0.1]))
        s = c.create_group("subclustering")
        s.attrs["history"] = json.dumps(history)
        for i, rec in enumerate(history):
            s.attrs[f"split_{i:02d}"] = json.dumps(rec)
        h = c.create_group("hires").create_dataset("labels", data=hires)
        h.attrs.update({"downsample_factor": 1, "header_trim_px": 100, "left_trim_px": 0,
                        "names": json.dumps({"13": SYNTH_NAMES[13], "14": SYNTH_NAMES[14]}),
                        "ratio": RATIO})
    return path


def _jet_png(values: np.ndarray, path: Path) -> None:
    rgb = (colormaps["jet"](values)[:, :, :3] * 255).astype(np.uint8)
    banner = np.zeros((100, rgb.shape[1], 3), np.uint8)
    Image.fromarray(np.vstack([banner, rgb])).save(path)


def write_synthetic_eds(dir: Path) -> Path:
    img, mask, hires, _ = final_labels()
    ca1 = np.where(hires == 14, 0.6, np.where(hires == 13, 0.2, 0.0)).astype(np.float32)
    ca1[np.repeat(np.repeat(img == 3, RATIO, 0), RATIO, 1)] = 0.3
    mg1 = np.where(hires == 14, 0.4, np.where(hires == 13, 0.8, 0.0)).astype(np.float32)
    bse1 = np.linspace(0, 1, H2 * RATIO, dtype=np.float32)[:, None].repeat(W2 * RATIO, 1)
    dir.mkdir(parents=True, exist_ok=True)
    _jet_png(ca1, dir / "SYN-Ca.png")
    _jet_png(mg1, dir / "SYN-Mg.png")
    sem = np.vstack([np.zeros((100, W2 * RATIO, 3), np.uint8),
                     np.repeat((bse1 * 255).astype(np.uint8)[:, :, None], 3, 2)])
    Image.fromarray(sem).save(dir / "SYN-SEM.png")
    (dir / "mask").mkdir(exist_ok=True)
    # napari polygon CSV: index, shape-type, vertex-index, axis-0, axis-1 (1x pixel coordinates incl. banner)
    r0, r1, c0, c1 = 100 + 12, 100 + 116, 12, 84
    lines = ["index,shape-type,vertex-index,axis-0,axis-1"]
    for i, (r, c) in enumerate([(r0, c0), (r0, c1), (r1, c1), (r1, c0)]):
        lines.append(f"0,polygon,{i},{r},{c}")
    (dir / "mask" / "Valid_mask.csv").write_text("\n".join(lines) + "\n")
    return dir


def write_synthetic_tima(path: Path) -> Path:
    """A 16-bit RGB TIMA-style raster: the hires map in TIMA colours at the fixed
    offset minus (1900, 1000), with a legend strip of C.TIMA_LEGEND_TRIM_ROWS rows."""
    img, _, hires, _ = final_labels()
    tima_index = {14: 2, 13: 1}                      # Augite -> high-Ca, Pigeonite -> low-Ca
    lab2 = np.repeat(np.repeat(img, RATIO, 0), RATIO, 1)
    idx = np.full(hires.shape, -1, np.int16)
    idx[lab2 == 3] = 0
    idx[lab2 == 15] = 6
    idx[lab2 == 16] = 6
    idx[lab2 == 4] = -1
    idx[lab2 == 10] = 12
    for k, t in tima_index.items():
        idx[hires == k] = t
    H, W = 200, 200
    rgb = np.full((H + C.TIMA_LEGEND_TRIM_ROWS, W, 3), 40, np.uint8)   # dark grey: achromatic
    dx, dy = C.TIMA_DX - 1900, C.TIMA_DY - 1000
    for p in C.TIMA_PHASES:
        rr, cc = np.nonzero(idx == p.index)
        rgb[rr + dy, cc + dx] = p.rgb
    import imageio.v3 as iio   # Pillow cannot write 16-bit RGB; imageio can
    iio.imwrite(path, rgb.astype(np.uint16) * 257)
    return path
```

The synthetic TIMA raster is 1200 rows by 200 columns; the offset placement needs `dx + 96 <= 200` and `dy + 128 <= 200` (19 + 96 = 115 and 51 + 128 = 179).

Copy the flow: `mkdir -p src/nwa4587_paper/data && cp <karak checkout>/src/karak/flow/flows/paper.json src/nwa4587_paper/data/paper_flow.json` with the karak checkout at commit 7ccd7a7 (`git -C <karak> show 7ccd7a7:src/karak/flow/flows/paper.json > src/nwa4587_paper/data/paper_flow.json`). Add `[tool.hatch.build.targets.wheel] include = ["src/nwa4587_paper/data/*.json"]` only if hatchling does not pick the file up by default (check with `uv build` and `unzip -l dist/*.whl`).

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_synth.py tests/test_constants.py -q`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/nwa4587_paper/synth.py src/nwa4587_paper/data/paper_flow.json tests/conftest.py tests/test_synth.py expected/h5_layout.json
git commit -m "feat: synthetic karak-layout data and the HDF5 layout listing"
```

### Task 1.4: `io.py`, the `Run` reader

**Files:**
- Create: `src/nwa4587_paper/io.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Produces:
  - `phase_key(name: str) -> str` (the name before the first `" ("`).
  - `@dataclass(frozen=True) Geometry(factor: int, header_trim_px: int, left_trim_px: int, shape_2x: tuple[int, int], shape_1x: tuple[int, int], ratio: int)` with properties `pixel_size_um_1x` (= `C.PIXEL_SIZE_UM_1X`) and `pixel_size_um_2x` (= `factor * PIXEL_SIZE_UM_1X`).
  - `@dataclass(frozen=True) DataPaths(eds_dir: Path | None, tima_png: Path | None)`.
  - `class Run` with `Run.open(path) -> Run` (context manager too), and properties/methods: `names: dict[int, str]`, `keys: dict[int, str]`, `phase(key) -> int`, `has(key) -> bool`, `labels: np.ndarray` (int32 2x image, -1 background), `mask`, `mineral_indices`, `cleaned_labels_flat`, `history: list[dict]`, `split_node_ids: list[str]`, `split_record(node_id) -> dict`, `labels_before(node_id) -> np.ndarray`, `hires_labels: np.ndarray` (int16), `hires_names: dict[int, str]`, `labels_1x: np.ndarray` (int16), `bse`, `channel(name) -> np.ndarray` (2x denoised), `element_order: list[str]`, `geometry: Geometry`, `flow: dict`, `flow_param(node_id, name)`, `n_mineral_pixels: int`, `counts: dict[int, int]`.
  - `load_channel_1x(eds_dir, element, header_trim_px) -> np.ndarray` (float32 [0, 1], jet inverted via `karak.io.loaders.invert_jet_colormap`), `load_bse_1x(eds_dir, header_trim_px) -> np.ndarray` (grey float32 [0, 1]), `find_png(eds_dir, element) -> Path` (the single file matching `*-{element}.png`).
  - `class MissingPhase(KeyError)` with a message `"no phase with key 'X'; names: [...]"`.
  - `class MissingInput(FileNotFoundError)` with a message `"<what> not found at <path>; run `paper <command>` first"`.

- [ ] **Step 1: Write the failing tests**

`tests/test_io.py`:
```python
import numpy as np
import pytest

from nwa4587_paper.io import Run, phase_key, MissingPhase, load_channel_1x, find_png


def test_phase_key():
    assert phase_key("Augite (high-Ca pyroxene)") == "Augite"
    assert phase_key("Zn-bearing Phase") == "Zn-bearing Phase"
    assert phase_key("Apatite (Ca₅(PO₄)₃(OH,Cl,F))") == "Apatite"


def test_phase_lookup(synth_h5):
    with Run.open(synth_h5) as run:
        assert run.phase("Augite") == 14
        assert run.phase("Pigeonite") == 13
        assert run.has("Epoxy") and not run.has("Calcite")


def test_phase_unknown_key(synth_h5):
    with Run.open(synth_h5) as run:
        with pytest.raises(MissingPhase) as e:
            run.phase("Calcite")
        assert "Calcite" in str(e.value) and "Augite" in str(e.value)


def test_labels_and_counts(synth_h5):
    with Run.open(synth_h5) as run:
        assert run.labels.shape == (64, 48)
        assert run.labels[0, 0] == -1
        assert sum(run.counts.values()) == run.n_mineral_pixels
        assert run.counts[run.phase("Xenotime")] == 16


def test_labels_before_undoes_splits(synth_h5):
    with Run.open(synth_h5) as run:
        assert run.split_node_ids == ["oliv", "weath", "pyx", "phos"]
        before_pyx = run.labels_before("pyx")
        assert 13 not in before_pyx and 14 not in before_pyx and 2 in before_pyx
        assert 15 in before_pyx                        # later splits are undone, earlier kept
        assert 11 in before_pyx and 12 in before_pyx
        before_oliv = run.labels_before("oliv")
        assert set(np.unique(before_oliv)) == {-1, 2, 3, 4, 7, 10}


def test_labels_1x_overlays_hires(synth_h5):
    with Run.open(synth_h5) as run:
        l1 = run.labels_1x
        assert l1.shape == run.hires_labels.shape == (128, 96)
        assert l1[12, 12] == run.phase("Plagioclase")           # repeated 2x label
        assert set(np.unique(l1[60:116, 12:60])) <= {13, 14, 2, -1, 11, 12}
        assert l1[60, 12] == -1 or l1[60, 12] in (13, 14)       # undefined corner stays what the 2x map says
        assert (l1 == run.phase("Epoxy")).sum() == 0            # epoxy becomes -1 at 1x


def test_labels_1x_clips(synth_h5, tmp_path):
    import h5py, shutil
    p = shutil.copy(synth_h5, tmp_path / "odd.h5")
    with h5py.File(p, "a") as f:
        data = f["clusters/hires/labels"][:-1, :-1]
        attrs = dict(f["clusters/hires/labels"].attrs)
        del f["clusters/hires/labels"]
        d = f["clusters/hires"].create_dataset("labels", data=data)
        d.attrs.update(attrs)
    with Run.open(p) as run:
        assert run.labels_1x.shape == (127, 95)


def test_geometry_and_flow(synth_h5):
    with Run.open(synth_h5) as run:
        g = run.geometry
        assert (g.factor, g.ratio, g.header_trim_px) == (2, 2, 100)
        assert g.shape_2x == (64, 48) and g.shape_1x == (128, 96)
        assert g.pixel_size_um_2x == 6.0
        assert run.flow["name"] == "paper"
        assert run.flow_param("hdb", "tile_size") == 1024
        assert run.split_record("pyx")["component_means"]["14"]["Ca/(Ca+Mg)"] == 0.53


def test_load_channel_1x(synth_eds):
    ca = load_channel_1x(synth_eds, "Ca", header_trim_px=100)
    assert ca.shape == (128, 96) and ca.dtype == np.float32
    assert abs(float(ca[12, 12]) - 0.3) < 0.02                  # plagioclase block
    assert find_png(synth_eds, "SEM").name == "SYN-SEM.png"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_io.py -q`
Expected: ImportError on `nwa4587_paper.io`.

- [ ] **Step 3: Write `io.py`**

```python
"""The karak HDF5 reader and the 1x PNG loaders."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import yaml

from . import constants as C


class MissingPhase(KeyError):
    pass


class MissingInput(FileNotFoundError):
    pass


def phase_key(name: str) -> str:
    return name.split(" (")[0].strip()


def require(path: Path | None, what: str, command: str) -> Path:
    if path is None or not Path(path).exists():
        raise MissingInput(f"{what} not found at {path}; run `paper {command}` first")
    return Path(path)


@dataclass(frozen=True)
class Geometry:
    factor: int
    header_trim_px: int
    left_trim_px: int
    shape_2x: tuple[int, int]
    shape_1x: tuple[int, int]
    ratio: int

    @property
    def pixel_size_um_1x(self) -> float:
        return C.PIXEL_SIZE_UM_1X

    @property
    def pixel_size_um_2x(self) -> float:
        return C.PIXEL_SIZE_UM_1X * self.factor


@dataclass(frozen=True)
class DataPaths:
    eds_dir: Path | None = None
    tima_png: Path | None = None


SPLIT_TYPES = ("split_threshold", "split_gmm", "split_hires")


class Run:
    def __init__(self, f: h5py.File):
        self.f = f

    @classmethod
    def open(cls, path: str | Path) -> "Run":
        return cls(h5py.File(require(Path(path), "karak HDF5", "karak"), "r"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()

    # ----- names -----
    @cached_property
    def names(self) -> dict[int, str]:
        return {int(k): v for k, v in json.loads(self.f["clusters"].attrs["mineral_names"]).items()}

    @cached_property
    def keys(self) -> dict[int, str]:
        return {k: phase_key(v) for k, v in self.names.items()}

    def phase(self, key: str) -> int:
        for lab, k in self.keys.items():
            if k == key:
                return lab
        raise MissingPhase(f"no phase with key '{key}'; names: {sorted(self.keys.values())}")

    def has(self, key: str) -> bool:
        return key in self.keys.values()

    # ----- 2x labels -----
    @cached_property
    def mineral_indices(self) -> np.ndarray:
        return self.f["clusters/mineral_indices"][:]

    @cached_property
    def cleaned_labels_flat(self) -> np.ndarray:
        return self.f["clusters/cleaned_labels"][:].astype(np.int32)

    @cached_property
    def mask(self) -> np.ndarray:
        return self.f["masks/mineral"][:].astype(bool)

    def _to_image(self, flat: np.ndarray) -> np.ndarray:
        img = np.full(self.geometry.shape_2x, -1, np.int32)
        img[self.mineral_indices[:, 0], self.mineral_indices[:, 1]] = flat
        return img

    @cached_property
    def labels(self) -> np.ndarray:
        return self._to_image(self.cleaned_labels_flat)

    @cached_property
    def counts(self) -> dict[int, int]:
        vals, n = np.unique(self.cleaned_labels_flat, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, n)}

    @property
    def n_mineral_pixels(self) -> int:
        return int(self.cleaned_labels_flat.size)

    # ----- history -----
    @cached_property
    def history(self) -> list[dict]:
        return json.loads(self.f["clusters/subclustering"].attrs["history"])

    @cached_property
    def split_node_ids(self) -> list[str]:
        cfg = yaml.safe_load(self.f["clusters"].attrs["cluster_config"]) or {}
        ids = [n["id"] for n in cfg.get("refinement_nodes", []) if n["type"] in SPLIT_TYPES]
        if len(ids) != len(self.history):
            ids = [f"split_{i:02d}" for i in range(len(self.history))]
        return ids

    def split_record(self, node_id: str) -> dict:
        return self.history[self.split_node_ids.index(node_id)]

    def labels_before(self, node_id: str) -> np.ndarray:
        """The 2x label image before the named split node: later splits and this one undone."""
        flat = self.cleaned_labels_flat.copy()
        start = self.split_node_ids.index(node_id)
        for rec in reversed(self.history[start:]):
            for new in rec["new_labels"]:
                flat[flat == int(new)] = int(rec["parent"])
        return self._to_image(flat)

    # ----- hires -----
    @cached_property
    def hires_labels(self) -> np.ndarray:
        return self.f["clusters/hires/labels"][:].astype(np.int16)

    @cached_property
    def hires_names(self) -> dict[int, str]:
        return {int(k): v for k, v in json.loads(self.f["clusters/hires/labels"].attrs["names"]).items()}

    @cached_property
    def labels_1x(self) -> np.ndarray:
        """2x labels repeated ratio x ratio, clipped to the hires shape, with the hires
        labels on top where defined. Epoxy becomes -1 (the old PNG decode did the same)."""
        r = self.geometry.ratio
        up = np.repeat(np.repeat(self.labels, r, 0), r, 1)
        h, w = self.hires_labels.shape
        out = up[:h, :w].astype(np.int16)
        if self.has("Epoxy"):
            out[out == self.phase("Epoxy")] = -1
        defined = self.hires_labels >= 0
        out[defined] = self.hires_labels[defined]
        return out

    # ----- images -----
    @cached_property
    def bse(self) -> np.ndarray:
        return self.f["bse/image"][:].astype(np.float32)

    @cached_property
    def element_order(self) -> list[str]:
        raw = self.f["denoised"].attrs["element_order"]
        return json.loads(raw) if isinstance(raw, str) else [str(x) for x in raw]

    def channel(self, name: str) -> np.ndarray:
        return self.f["denoised/cube"][:, :, self.element_order.index(name)].astype(np.float32)

    def raw_channel(self, name: str) -> np.ndarray:
        return self.f[f"raw/{name}"][:].astype(np.float32)

    # ----- geometry and flow -----
    @cached_property
    def geometry(self) -> Geometry:
        bse = self.f["bse"]
        hires = self.f["clusters/hires/labels"]
        shape_2x = tuple(int(x) for x in self.f["bse/image"].shape)
        return Geometry(
            factor=int(bse.attrs["downsample_factor"]),
            header_trim_px=int(hires.attrs["header_trim_px"]),
            left_trim_px=int(hires.attrs["left_trim_px"]),
            shape_2x=shape_2x,
            shape_1x=tuple(int(x) for x in hires.shape),
            ratio=int(hires.attrs["ratio"]),
        )

    @cached_property
    def flow(self) -> dict:
        raw = self.f.attrs["pipeline_config"]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return yaml.safe_load(raw)

    def flow_param(self, node_id: str, name: str):
        for node in self.flow["nodes"]:
            if node["id"] == node_id:
                return node["params"][name]
        raise KeyError(f"flow has no node '{node_id}'")


# ----- 1x PNG loaders -----
def find_png(eds_dir: Path, element: str) -> Path:
    eds_dir = require(eds_dir, "elemental PNG directory", "data --from PATH")
    hits = sorted(eds_dir.glob(f"*-{element}.png"))
    if len(hits) != 1:
        raise MissingInput(f"expected one *-{element}.png in {eds_dir}, found {len(hits)}")
    return hits[0]


def load_channel_1x(eds_dir: Path, element: str, header_trim_px: int) -> np.ndarray:
    import imageio.v3 as iio
    from karak.io.loaders import invert_jet_colormap

    rgb = iio.imread(find_png(eds_dir, element))
    return invert_jet_colormap(rgb)[header_trim_px:, :].astype(np.float32)


def load_bse_1x(eds_dir: Path, header_trim_px: int) -> np.ndarray:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    arr = np.asarray(Image.open(find_png(eds_dir, "SEM")).convert("L"), dtype=np.float32) / 255.0
    return arr[header_trim_px:, :]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_io.py -q`
Expected: 10 passed. If `test_labels_1x_overlays_hires` fails on the Plagioclase assertion, check the synthetic rows: 2x row 6 to 30 is 1x row 12 to 60.

- [ ] **Step 5: Commit, mark PR 1 ready**

```bash
git add src/nwa4587_paper/io.py tests/test_io.py
git commit -m "feat: Run reader resolves phases by name and rebuilds pre-split maps"
git push && gh pr ready
```

Wait for CI (`gh pr checks --watch`), fix failures on the branch, then stop: the user merges.

---

# Part 2: the numbers bridge, `tables`, `fingerprints`, the CLI (PR 2, branch `feat/numbers-tables-fingerprints`)

### Task 2.1: `numbers.py`

**Files:**
- Create: `src/nwa4587_paper/numbers.py`
- Test: `tests/test_numbers.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Spec(kind: str, tol: float = 0.0, rel: bool = False, fmt: str = "{}")` with `kind` in `"int" | "float" | "str" | "bool"`.
  - `REGISTRY: dict[str, Spec]` keyed by dotted path with `*` wildcards for phase keys and element names, for example `"tables.phases.*.pct"`. Each stage task adds its keys here.
  - `flatten(d: dict, prefix: str = "") -> dict[str, object]` (nested dicts to dotted keys; lists stay values).
  - `spec_for(key) -> Spec` (exact match first, then `fnmatch` against wildcard keys; `KeyError` if none).
  - `validate(stage: str, numbers: dict) -> None` (every flattened key under `stage.` has a spec; raises `KeyError` listing the offenders).
  - `write_stage(out: Path, stage: str, numbers: dict) -> Path` writes `out/numbers/<stage>.json` after `validate`.
  - `merge(out: Path) -> dict` reads every `out/numbers/*.json` into `{stage: numbers}` and writes `out/numbers.json`.
  - `macro_name(key: str) -> str` (`"tables.phases.Fe-Ti-Cr Spinel.pct"` -> `\nwaTablesPhasesFeTiCrSpinelPct`; digits spelled `Zero` to `Nine`).
  - `render_tex(numbers: dict) -> str` one `\newcommand{\name}{formatted}` per leaf, sorted; lists are skipped.
  - `@dataclass Miss(key: str, expected: object, actual: object, tol: float)`.
  - `check(numbers: dict, expected: dict) -> list[Miss]` (missing on either side is a `Miss` with `None`; ints and strs exact; floats within `tol` absolute, or `tol * |expected|` when `rel`; NaN equals NaN).
  - `format_misses(misses) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/test_numbers.py`:
```python
import json
import math
import pytest

from nwa4587_paper import numbers as N


def test_flatten_and_spec_lookup():
    N.REGISTRY["demo.total"] = N.Spec("int")
    N.REGISTRY["demo.phases.*.pct"] = N.Spec("float", tol=0.001, fmt="{:.3f}")
    flat = N.flatten({"demo": {"total": 3, "phases": {"Augite": {"pct": 1.5}}}})
    assert flat == {"demo.total": 3, "demo.phases.Augite.pct": 1.5}
    assert N.spec_for("demo.phases.Augite.pct").tol == 0.001
    with pytest.raises(KeyError):
        N.spec_for("demo.unknown")


def test_validate_rejects_unlisted_key():
    with pytest.raises(KeyError) as e:
        N.validate("demo", {"total": 1, "extra": 2})
    assert "demo.extra" in str(e.value)


def test_write_merge_render(tmp_path):
    N.write_stage(tmp_path, "demo", {"total": 3, "phases": {"Fe-Ti-Cr Spinel": {"pct": 1.5}}})
    merged = N.merge(tmp_path)
    assert merged["demo"]["total"] == 3
    assert json.load(open(tmp_path / "numbers.json")) == merged
    tex = N.render_tex(merged)
    assert r"\newcommand{\nwaDemoTotal}{3}" in tex
    assert r"\newcommand{\nwaDemoPhasesFeTiCrSpinelPct}{1.500}" in tex


def test_macro_name_digits():
    assert N.macro_name("pyroxene.n_pixels_2x") == r"\nwaPyroxeneNPixelsTwoX"


def test_check_tolerances():
    N.REGISTRY["demo.f"] = N.Spec("float", tol=0.01)
    N.REGISTRY["demo.r"] = N.Spec("float", tol=0.01, rel=True)
    N.REGISTRY["demo.n"] = N.Spec("int")
    actual = {"demo": {"f": 1.005, "r": 1010.0, "n": 5}}
    expected = {"demo": {"f": 1.0, "r": 1000.0, "n": 5}}
    assert N.check(actual, expected) == []
    expected["demo"]["f"] = 1.02
    misses = N.check(actual, expected)
    assert [m.key for m in misses] == ["demo.f"]


def test_check_missing_key():
    N.REGISTRY["demo.n"] = N.Spec("int")
    N.REGISTRY["demo.m"] = N.Spec("int")
    misses = N.check({"demo": {"n": 1}}, {"demo": {"m": 1}})
    assert sorted(m.key for m in misses) == ["demo.m", "demo.n"]
    assert "demo.m" in N.format_misses(misses)


def test_check_nan_equals_nan():
    N.REGISTRY["demo.f"] = N.Spec("float", tol=0.01)
    assert N.check({"demo": {"f": math.nan}}, {"demo": {"f": math.nan}}) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_numbers.py -q`
Expected: ImportError.

- [ ] **Step 3: Write `numbers.py`**

```python
"""The numbers bridge: registry, per-stage files, merge, LaTeX macros, --check."""
from __future__ import annotations

import fnmatch
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    kind: str                 # int | float | str | bool
    tol: float = 0.0
    rel: bool = False
    fmt: str = "{}"


REGISTRY: dict[str, Spec] = {}
# Each stage task extends REGISTRY at import time of its module through
# `numbers.register({...})`, and `cli.py` imports every stage module, so the
# registry is complete whenever `paper` runs.


def register(entries: dict[str, Spec]) -> None:
    REGISTRY.update(entries)


def flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def spec_for(key: str) -> Spec:
    if key in REGISTRY:
        return REGISTRY[key]
    for pattern, spec in REGISTRY.items():
        if "*" in pattern and fnmatch.fnmatchcase(key, pattern):
            return spec
    raise KeyError(key)


def validate(stage: str, numbers: dict) -> None:
    bad = []
    for key in flatten({stage: numbers}):
        try:
            spec_for(key)
        except KeyError:
            bad.append(key)
    if bad:
        raise KeyError(f"numbers not in REGISTRY: {bad}")


def write_stage(out: Path, stage: str, numbers: dict) -> Path:
    validate(stage, numbers)
    d = Path(out) / "numbers"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stage}.json"
    path.write_text(json.dumps(numbers, indent=2, sort_keys=True, default=_jsonable))
    return path


def _jsonable(x):
    if hasattr(x, "item"):
        return x.item()
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(type(x))


def merge(out: Path) -> dict:
    out = Path(out)
    merged = {p.stem: json.loads(p.read_text()) for p in sorted((out / "numbers").glob("*.json"))}
    (out / "numbers.json").write_text(json.dumps(merged, indent=2, sort_keys=True))
    return merged


_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro_name(key: str) -> str:
    parts = re.split(r"[.\s_\-/()+,]+", key)
    words = []
    for part in parts:
        for ch in part:
            words.append(_DIGITS[int(ch)] if ch.isdigit() else ch)
        words.append(" ")
    camel = "".join(w[:1].upper() + w[1:] for w in "".join(words).split())
    camel = re.sub(r"[^A-Za-z]", "", camel)
    return "\\nwa" + camel


def _format(key: str, value) -> str:
    spec = spec_for(key)
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return spec.fmt.format(value)


def render_tex(numbers: dict) -> str:
    lines = ["% generated by `paper numbers`; do not edit"]
    for key, value in sorted(flatten(numbers).items()):
        if isinstance(value, (list, tuple)) or value is None:
            continue
        lines.append(f"\\newcommand{{{macro_name(key)}}}{{{_format(key, value)}}}")
    return "\n".join(lines) + "\n"


@dataclass
class Miss:
    key: str
    expected: object
    actual: object
    tol: float


def _close(spec: Spec, a, e) -> bool:
    if spec.kind == "float":
        a, e = float(a), float(e)
        if math.isnan(a) and math.isnan(e):
            return True
        tol = spec.tol * abs(e) if spec.rel else spec.tol
        return abs(a - e) <= tol
    return a == e


def check(numbers: dict, expected: dict) -> list[Miss]:
    got, want = flatten(numbers), flatten(expected)
    misses = []
    for key in sorted(set(got) | set(want)):
        if key not in got or key not in want:
            misses.append(Miss(key, want.get(key), got.get(key), 0.0))
            continue
        spec = spec_for(key)
        if isinstance(got[key], (list, tuple)):
            if list(got[key]) != list(want[key]):
                misses.append(Miss(key, want[key], got[key], 0.0))
        elif not _close(spec, got[key], want[key]):
            misses.append(Miss(key, want[key], got[key], spec.tol))
    return misses


def format_misses(misses: list[Miss]) -> str:
    if not misses:
        return "all numbers match"
    rows = [f"{m.key}: expected {m.expected!r}, actual {m.actual!r} (tol {m.tol})" for m in misses]
    return f"{len(misses)} numbers differ:\n  " + "\n  ".join(rows)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_numbers.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git switch main && git pull && git switch -c feat/numbers-tables-fingerprints
git add src/nwa4587_paper/numbers.py tests/test_numbers.py
git commit -m "feat: numbers registry, merge, LaTeX macros and --check"
```

### Task 2.2: `cli.py` and the `tables` stage (Tables 1 and 2)

**Files:**
- Create: `src/nwa4587_paper/cli.py`, `src/nwa4587_paper/tables.py`
- Test: `tests/test_tables.py`, `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `cli.main(argv: list[str] | None = None) -> int`. Subcommands: `tables`, `fingerprints`, `tima`, `pyroxene`, `xenotime`, `lamellae`, `colormap`, `workflow`, `figures`, `numbers`, `data`, `karak`, `all`, `synth`. Every stage subcommand takes `--h5 PATH` (default `results/paperflow.h5`), `--out DIR` (default `results`), `--data DIR` (default `data`; `eds_dir = DIR/eds`, `tima_png = DIR/tima/phases.png` if it exists, else the single `*.png` under `DIR/tima`), `--check` (compare with `expected/numbers.json`, exit 1 on a miss) and `--expected PATH`.
  - `cli.STAGES: dict[str, Callable[[Run, Path, DataPaths], dict]]` filled by the stage modules in this order: `tables, fingerprints, tima, pyroxene, xenotime, lamellae, colormap, workflow`. Stages added in later tasks register themselves; until then `figures` runs the ones that exist.
  - `cli.run_stage(name, args) -> int`: opens the `Run` (the `colormap` stage gets `None`), calls the stage, writes `numbers/<name>.json`, prints the stage's files, runs `--check` on that stage's subtree.
  - `paper synth DIR` writes the synthetic HDF5 to `DIR/results/paperflow.h5`, the PNGs to `DIR/data/eds`, the raster to `DIR/data/tima/phases.png` (for the CI smoke run).
  - `tables.run(run, out, data) -> dict` writes `out/tables/table1_params.tex`, `table2_phases.tex`, `table2_phases.csv`; returns `{"n_mineral_pixels": int, "n_phases_with_pixels": int, "phases": {key: {"label": int, "pixels": int, "pct": float}}, "params": {row label: value}}`.
  - A `MissingInput` raised by a stage exits with code 2 and prints the message; a `MissingPhase` exits 3.

- [ ] **Step 1: Write the failing tests**

`tests/test_tables.py`:
```python
import csv
from pathlib import Path

from nwa4587_paper import numbers as N
from nwa4587_paper import tables
from nwa4587_paper.io import Run, DataPaths


def test_tables_writes_files_and_numbers(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = tables.run(run, tmp_path, DataPaths())
    N.validate("tables", nums)
    assert (tmp_path / "tables" / "table1_params.tex").exists()
    tex = (tmp_path / "tables" / "table2_phases.tex").read_text()
    assert "Augite" in tex and r"\toprule" in tex
    rows = list(csv.DictReader(open(tmp_path / "tables" / "table2_phases.csv")))
    assert rows[0]["phase"] == "Plagioclase"                   # sorted by area, descending
    assert sum(int(r["pixels"]) for r in rows) == nums["n_mineral_pixels"]
    assert abs(sum(float(r["pct"]) for r in rows) - 100.0) < 0.01
    assert nums["phases"]["Xenotime"]["pixels"] == 16
    assert nums["params"]["Tile size (px)"] == 1024
    assert nums["n_phases_with_pixels"] == 9


def test_tables_is_deterministic(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        a = tables.run(run, tmp_path / "a", DataPaths())
        b = tables.run(run, tmp_path / "b", DataPaths())
    assert a == b
```

`tests/test_cli.py`:
```python
import json
import subprocess
import sys

from nwa4587_paper import cli


def test_tables_subcommand_writes_numbers(synth_h5, tmp_path):
    rc = cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    assert rc == 0
    nums = json.load(open(tmp_path / "numbers" / "tables.json"))
    assert "phases" in nums


def test_check_fails_on_mismatch(synth_h5, tmp_path):
    cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path)])
    exp = {"tables": json.load(open(tmp_path / "numbers" / "tables.json"))}
    exp["tables"]["n_mineral_pixels"] += 1
    p = tmp_path / "expected.json"
    p.write_text(json.dumps(exp))
    assert cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path), "--check", "--expected", str(p)]) == 1
    exp["tables"]["n_mineral_pixels"] -= 1
    p.write_text(json.dumps(exp))
    assert cli.main(["tables", "--h5", str(synth_h5), "--out", str(tmp_path), "--check", "--expected", str(p)]) == 0


def test_missing_h5_exits_2(tmp_path):
    assert cli.main(["tables", "--h5", str(tmp_path / "none.h5"), "--out", str(tmp_path)]) == 2


def test_synth_subcommand(tmp_path):
    assert cli.main(["synth", str(tmp_path)]) == 0
    assert (tmp_path / "results" / "paperflow.h5").exists()
    assert (tmp_path / "data" / "tima" / "phases.png").exists()


def test_console_script_runs():
    out = subprocess.run([sys.executable, "-m", "nwa4587_paper.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "tables" in out.stdout
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tables.py tests/test_cli.py -q`
Expected: ImportError on `tables` and `cli`.

- [ ] **Step 3: Write `tables.py`**

```python
"""Table 1 (flow parameters) and Table 2 (phase abundances)."""
from __future__ import annotations

import csv
from pathlib import Path

from . import constants as C
from . import numbers as N
from .io import DataPaths, Run

N.register({
    "tables.n_mineral_pixels": N.Spec("int"),
    "tables.n_phases_with_pixels": N.Spec("int"),
    "tables.phases.*.label": N.Spec("int"),
    "tables.phases.*.pixels": N.Spec("int"),
    "tables.phases.*.pct": N.Spec("float", tol=0.0005, fmt="{:.2f}"),
    "tables.params.*": N.Spec("str"),
})


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def table1_rows(run: Run) -> list[tuple[str, str]]:
    rows = []
    for label, node, param, fmt in C.TABLE1_ROWS:
        value = run.flow_param(node, param)
        rows.append((label, fmt.format("auto" if value is None else value)))
    return rows


def write_table1(path: Path, rows: list[tuple[str, str]]) -> None:
    lines = [r"\begin{tabular}{ll}", r"\toprule", r"Parameter & Value \\", r"\midrule"]
    lines += [f"{_tex_escape(k)} & {_tex_escape(v)} \\\\" for k, v in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))


def table2_rows(run: Run) -> list[dict]:
    total = run.n_mineral_pixels
    rows = [{"label": lab, "phase": run.keys[lab], "name": run.names[lab],
             "pixels": n, "pct": 100.0 * n / total}
            for lab, n in run.counts.items() if n > 0]
    rows.sort(key=lambda r: (-r["pixels"], r["label"]))
    return rows


def _pct_text(pct: float) -> str:
    return "<0.01" if pct < 0.01 else f"{pct:.2f}"


def write_table2(dir: Path, rows: list[dict]) -> None:
    with open(dir / "table2_phases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "phase", "name", "pixels", "pct"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "pct": f"{r['pct']:.4f}"})
    lines = [r"\begin{tabular}{rlrr}", r"\toprule", r"Label & Phase & Area \% & Pixels \\", r"\midrule"]
    for r in rows:
        pixels = f"{r['pixels']:,}".replace(",", "{,}")
        lines.append(f"{r['label']} & {_tex_escape(r['phase'])} & {_pct_text(r['pct'])} & {pixels} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (dir / "table2_phases.tex").write_text("\n".join(lines))


def run(run: Run, out: Path, data: DataPaths) -> dict:
    d = Path(out) / "tables"
    d.mkdir(parents=True, exist_ok=True)
    t1 = table1_rows(run)
    write_table1(d / "table1_params.tex", t1)
    t2 = table2_rows(run)
    write_table2(d, t2)
    return {
        "n_mineral_pixels": run.n_mineral_pixels,
        "n_phases_with_pixels": len(t2),
        "phases": {r["phase"]: {"label": r["label"], "pixels": r["pixels"], "pct": r["pct"]} for r in t2},
        "params": {k: v for k, v in t1},
    }
```

- [ ] **Step 4: Write `cli.py`**

```python
"""The `paper` command."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from . import numbers as N
from .io import DataPaths, MissingInput, MissingPhase, Run

STAGES: dict[str, Callable] = {}
STAGE_ORDER = ["tables", "fingerprints", "tima", "pyroxene", "xenotime", "lamellae", "colormap", "workflow"]
NEEDS_RUN = {"colormap": False}


def _import_stages() -> None:
    from . import tables  # noqa: F401
    STAGES.setdefault("tables", tables.run)
    for name in STAGE_ORDER[1:]:
        try:
            mod = __import__(f"nwa4587_paper.{name}", fromlist=["run"])
            STAGES.setdefault(name, mod.run)
        except ImportError:
            pass


def data_paths(data_dir: Path) -> DataPaths:
    eds = data_dir / "eds"
    tima_dir = data_dir / "tima"
    tima = None
    if (tima_dir / "phases.png").exists():
        tima = tima_dir / "phases.png"
    elif tima_dir.is_dir():
        pngs = sorted(tima_dir.glob("*.png"))
        tima = pngs[0] if len(pngs) == 1 else None
    return DataPaths(eds_dir=eds if eds.is_dir() else None, tima_png=tima)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--h5", default="results/paperflow.h5")
    p.add_argument("--out", default="results")
    p.add_argument("--data", default="data")
    p.add_argument("--check", action="store_true")
    p.add_argument("--expected", default="expected/numbers.json")


def run_stage(name: str, args) -> int:
    fn = STAGES[name]
    out = Path(args.out)
    data = data_paths(Path(args.data))
    try:
        if NEEDS_RUN.get(name, True):
            with Run.open(args.h5) as run:
                nums = fn(run, out, data)
        else:
            nums = fn(None, out, data)
    except MissingInput as e:
        print(f"{name}: {e}", file=sys.stderr)
        return 2
    except MissingPhase as e:
        print(f"{name}: {e}", file=sys.stderr)
        return 3
    path = N.write_stage(out, name, nums)
    print(f"{name}: wrote {path}")
    if args.check:
        return _check({name: nums}, args.expected)
    return 0


def _check(numbers: dict, expected_path: str) -> int:
    expected = json.loads(Path(expected_path).read_text())
    expected = {k: v for k, v in expected.items() if k in numbers}
    misses = N.check(numbers, expected)
    print(N.format_misses(misses))
    return 1 if misses else 0


def cmd_numbers(args) -> int:
    out = Path(args.out)
    merged = N.merge(out)
    (out / "numbers.tex").write_text(N.render_tex(merged))
    print(f"numbers: wrote {out / 'numbers.json'} and numbers.tex")
    if args.check:
        return _check(merged, args.expected)
    return 0


def cmd_figures(args) -> int:
    worst = 0
    for name in STAGE_ORDER:
        if name in STAGES:
            worst = max(worst, run_stage(name, args))
    return worst


def cmd_synth(args) -> int:
    from . import synth
    root = Path(args.dir)
    (root / "results").mkdir(parents=True, exist_ok=True)
    synth.write_synthetic_h5(root / "results" / "paperflow.h5")
    synth.write_synthetic_eds(root / "data" / "eds")
    (root / "data" / "tima").mkdir(parents=True, exist_ok=True)
    synth.write_synthetic_tima(root / "data" / "tima" / "phases.png")
    print(f"synthetic data under {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    _import_stages()
    p = argparse.ArgumentParser(prog="paper", description="Regenerate the NWA 4587 paper results.")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in STAGE_ORDER:
        sp = sub.add_parser(name, help=f"run the {name} stage")
        _add_common(sp)
        sp.set_defaults(func=lambda a, n=name: run_stage(n, a))
    sp = sub.add_parser("figures", help="run every stage")
    _add_common(sp)
    sp.set_defaults(func=cmd_figures)
    sp = sub.add_parser("numbers", help="merge numbers, render numbers.tex, optionally --check")
    _add_common(sp)
    sp.set_defaults(func=cmd_numbers)
    sp = sub.add_parser("synth", help="write synthetic data for a smoke run")
    sp.add_argument("dir")
    sp.set_defaults(func=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd in STAGE_ORDER and args.cmd not in STAGES:
        print(f"{args.cmd}: stage not implemented yet", file=sys.stderr)
        return 4
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

Task 6.1 adds the `data`, `karak` and `all` subcommands.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: all pass (Part 1 tests plus 7 new).

- [ ] **Step 6: Commit**

```bash
git add src/nwa4587_paper/cli.py src/nwa4587_paper/tables.py tests/test_tables.py tests/test_cli.py
git commit -m "feat: paper CLI and the tables stage (Tables 1 and 2)"
```

### Task 2.3: the `fingerprints` stage (Fig 2)

**Files:**
- Create: `src/nwa4587_paper/fingerprints.py`
- Test: `tests/test_fingerprints.py`

**Interfaces:**
- Produces: `fingerprints.run(run, out, data) -> dict` writing `out/figures/fig2_fingerprints.pdf` and `.png`; returns `{"element_order": [str], "phases": {key: {"n_pixels": int, "area_pct": float, "mean": {element: float}, "std": {element: float}}}}`.
- `fingerprints.compute(run) -> dict[int, dict]` (per label `mean`, `std`, `n_pixels`, `area_pct` with `area_pct` over all mineral pixels, excluded keys removed) and `element_order(fps) -> list[str]` (elements by descending variance of the per-phase means over the kept phases, as `R/paper/figures/make_fig5_fingerprints.py` L160 to L175 does).

Port notes (from `make_fig5_fingerprints.py`): exclude `C.FINGERPRINT_EXCLUDED_KEYS` and phases with 0 pixels; use `karak.identification.fingerprint.compute_fingerprints(run.f["denoised/cube"][:], run.cleaned_labels_flat, run.mineral_indices, run.element_order, accumulate="float32")` for the means and stds (float32 reproduces the paper's values), then recompute the element ranking on the kept phases only. Layout: three columns, `n_rows = ceil(n/3)`, figure width 17 cm, height `n_rows * 1.30 + 0.60` in; bars `#4878a8` edge `#345a80` width 0.70; error bars `#444444` elinewidth 0.45 capsize 1.0; y `MultipleLocator(0.25)`, ylim `(0, min(1, 1.08 * max(mean + std)))`; title `f"{display}  ({pct})"` bold 7 pt left, with `pct` formatted `.1f` at >= 1, `.2f` at >= 0.1, `.3f` at >= 0.01, else `<0.01`; x tick labels only on the bottom-visible panels, rotated 50 degrees anchored right at 6 pt; "Mean intensity" on column 0; panels ordered by pixel count descending; `subplots_adjust(left=0.07, right=0.98, bottom=0.60/fig_h, top=0.98, hspace=0.55, wspace=0.30)`; unused axes hidden; `pad_inches=0.04`. Display names from `C.DISPLAY_NAMES`.

- [ ] **Step 1: Write the failing tests**

`tests/test_fingerprints.py`:
```python
from nwa4587_paper import fingerprints, numbers as N
from nwa4587_paper.io import Run, DataPaths


def test_fingerprints_excludes_and_orders(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = fingerprints.run(run, tmp_path, DataPaths())
    N.validate("fingerprints", nums)
    assert "Epoxy" not in nums["phases"] and "Weathering Assemblage" not in nums["phases"]
    assert "Augite" in nums["phases"] and "Xenotime" in nums["phases"]
    assert set(nums["element_order"]) == {"Ca", "Mg", "Y", "P"}
    assert nums["element_order"][0] in ("Y", "P", "Mg", "Ca")
    assert abs(nums["phases"]["Augite"]["mean"]["Ca"] - 0.6) < 1e-6
    assert (tmp_path / "figures" / "fig2_fingerprints.pdf").exists()
    assert (tmp_path / "figures" / "fig2_fingerprints.png").exists()


def test_area_pct_uses_all_mineral_pixels(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = fingerprints.run(run, tmp_path, DataPaths())
        total = run.n_mineral_pixels
        assert abs(nums["phases"]["Xenotime"]["area_pct"] - 100 * 16 / total) < 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_fingerprints.py -q`
Expected: ImportError.

- [ ] **Step 3: Write `fingerprints.py`**

Register:
```python
N.register({
    "fingerprints.element_order": N.Spec("str"),
    "fingerprints.phases.*.n_pixels": N.Spec("int"),
    "fingerprints.phases.*.area_pct": N.Spec("float", tol=0.0005, fmt="{:.2f}"),
    "fingerprints.phases.*.mean.*": N.Spec("float", tol=0.002, fmt="{:.3f}"),
    "fingerprints.phases.*.std.*": N.Spec("float", tol=0.002, fmt="{:.3f}"),
})
```

Core:
```python
def compute(run: Run) -> dict[int, dict]:
    from karak.identification.fingerprint import compute_fingerprints

    cube = run.f["denoised/cube"][:]
    res = compute_fingerprints(cube, run.cleaned_labels_flat, run.mineral_indices,
                               run.element_order, accumulate="float32")
    fps = res["fingerprints"]
    drop = {run.phase(k) for k in C.FINGERPRINT_EXCLUDED_KEYS if run.has(k)}
    return {lab: fp for lab, fp in fps.items() if lab not in drop and fp["n_pixels"] > 0}


def element_order(fps: dict[int, dict], elements: list[str]) -> list[str]:
    means = np.array([fp["mean"] for fp in fps.values()])
    order = np.argsort(np.var(means, axis=0))[::-1]
    return [elements[i] for i in order]
```
`run()` draws the figure per the port notes, saves both files, and returns the dict with `float(...)` conversions. Use `C.DISPLAY_NAMES.get(key, key)` for titles.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_fingerprints.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/nwa4587_paper/fingerprints.py tests/test_fingerprints.py
git commit -m "feat: fingerprints stage (Fig 2)"
```

### Task 2.4: CI smoke job, `figures` and `numbers` end to end

**Files:**
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_cli.py` (append)

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/test_cli.py`:
```python
def test_figures_then_numbers_on_synthetic(tmp_path):
    assert cli.main(["synth", str(tmp_path)]) == 0
    common = ["--h5", str(tmp_path / "results" / "paperflow.h5"), "--out", str(tmp_path / "results"),
              "--data", str(tmp_path / "data")]
    assert cli.main(["figures", *common]) == 0
    assert cli.main(["numbers", *common]) == 0
    merged = json.load(open(tmp_path / "results" / "numbers.json"))
    assert {"tables", "fingerprints"} <= set(merged)
    tex = (tmp_path / "results" / "numbers.tex").read_text()
    assert r"\nwaTablesNMineralPixels" in tex
```

- [ ] **Step 2: Run, verify, then add the smoke job**

Run: `uv run pytest tests/test_cli.py -q`. Expected: pass (the stages that exist run; later tasks make the test cover more).

Append to `ci.yml`:
```yaml
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv python install 3.12
      - run: uv sync
      - run: uv run paper synth smoke
      - run: uv run paper figures --h5 smoke/results/paperflow.h5 --out smoke/results --data smoke/data
      - run: uv run paper numbers --out smoke/results
      - run: test -s smoke/results/numbers.tex && ls smoke/results/figures smoke/results/tables
```

- [ ] **Step 3: Commit, push, open PR 2**

```bash
git add .github/workflows/ci.yml tests/test_cli.py
git commit -m "ci: smoke run of paper figures and numbers on synthetic data"
git push -u origin HEAD
gh pr create --title "feat: numbers bridge, tables and fingerprints stages, paper CLI" --body "Part 2 of the plan. Tables 1 and 2, Fig 2, numbers.json, numbers.tex, --check, CI smoke run."
gh pr checks --watch
```

---

# Part 3: the TIMA comparison (PR 3, branch `feat/tima`)

### Task 3.1: `tima_core.py`, decode, align, combine

**Files:**
- Create: `src/nwa4587_paper/tima_core.py`
- Test: `tests/test_tima_core.py`

**Interfaces:**
- Produces:
  - `decode_tima(png: Path, legend_trim: int = C.TIMA_LEGEND_TRIM_ROWS, distance_max: float = C.TIMA_COLOR_DISTANCE_MAX) -> np.ndarray` (int8, TIMA index 0 to 13, -1 elsewhere). The PNG may be 8-bit or 16-bit RGB; 16-bit values are divided by 257 and rounded to 8-bit before matching.
  - `align(karak: np.ndarray, tima: np.ndarray, dx: int = C.TIMA_DX, dy: int = C.TIMA_DY) -> tuple[np.ndarray, np.ndarray, dict]`, the verbatim port of `align_translation` (`R/analysis/tima_comparison_1x.py` L181 to L222) with the prints removed; `info` keys `dx, dy, tima_window_xyxy, karak_window_xyxy, shape_hw`.
  - `combine_pyroxene(run_phase: Callable[[str], int], karak: np.ndarray, tima: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: karak `Augite` -> `Pigeonite` label; TIMA `C.TIMA_COMBINE_FROM` -> `C.TIMA_COMBINE_INTO`.
  - `Mode = Literal["split", "combined"]`; `build_mapping(run, mode) -> dict[int, dict]` with entries `{"key": str, "name": str, "tima_labels": list[int], "notes": str}` keyed by karak label, in the order of `C.LEGEND_ORDER_*`, skipping keys the run lacks; in combined mode the `Pigeonite` entry gets `name = "Pyroxene"`, `tima_labels = [1]` and `notes = C.COMBINED_PYROXENE_NOTE`.
  - `exclusions(run, mode) -> tuple[set[int], set[int]]`: karak `{-1} | {labels of C.EXCLUDED_KARAK_KEYS present}`, TIMA `{-1} | set(C.EXCLUDED_TIMA_SPLIT or _COMBINED)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tima_core.py`:
```python
import numpy as np
import pytest
import imageio.v3 as iio

from nwa4587_paper import constants as C, tima_core as T
from nwa4587_paper.io import Run


def _png(tmp_path, rgb8, bits=8):
    p = tmp_path / f"t{bits}.png"
    iio.imwrite(p, (rgb8.astype(np.uint16) * 257) if bits == 16 else rgb8.astype(np.uint8))
    return p


def test_decode_exact_near_far_grey_and_trim(tmp_path):
    img = np.zeros((6, 4, 3), np.uint8)
    img[0, 0] = C.TIMA_PHASES[0].rgb                       # exact -> 0
    img[0, 1] = np.array(C.TIMA_PHASES[7].rgb) - 10        # within 50 -> 7
    img[0, 2] = (200, 60, 60)                              # far from every colour -> -1
    img[0, 3] = (120, 125, 118)                            # achromatic -> -1
    img[1, 0] = (10, 10, 10)                               # dark, max <= 20: not achromatic rule; nearest wins or -1 by distance
    img[5, :] = C.TIMA_PHASES[0].rgb                       # legend row, trimmed away
    for bits in (8, 16):
        out = T.decode_tima(_png(tmp_path, img, bits), legend_trim=1)
        assert out.shape == (5, 4) and out.dtype == np.int8
        assert out[0, 0] == 0 and out[0, 1] == 7 and out[0, 2] == -1 and out[0, 3] == -1
        assert out[1, 0] == -1                              # (10,10,10) is 30+ from (0,85,0): over the limit


def test_align_windows():
    karak = np.zeros((10, 8), np.int16)
    tima = np.zeros((30, 30), np.int8)
    k, t, info = T.align(karak, tima, dx=5, dy=7)
    assert k.shape == t.shape == (10, 8)
    assert info["tima_window_xyxy"] == (5, 7, 13, 17)
    assert info["karak_window_xyxy"] == (0, 0, 8, 10)


def test_align_crops_karak_when_tima_is_short():
    karak = np.zeros((10, 8), np.int16)
    tima = np.zeros((12, 30), np.int8)
    k, t, info = T.align(karak, tima, dx=5, dy=7)
    assert k.shape == (5, 8) and info["shape_hw"] == (5, 8)


def test_align_no_overlap():
    with pytest.raises(ValueError, match="no overlap"):
        T.align(np.zeros((4, 4)), np.zeros((4, 4)), dx=10, dy=10)


def test_combine_and_mapping(synth_h5):
    with Run.open(synth_h5) as run:
        karak = np.array([[run.phase("Augite"), run.phase("Pigeonite"), -1]], np.int16)
        tima = np.array([[2, 5, 1]], np.int8)
        k, t = T.combine_pyroxene(run.phase, karak, tima)
        assert list(k[0]) == [run.phase("Pigeonite")] * 2 + [-1]
        assert list(t[0]) == [1, 1, 1]
        split = T.build_mapping(run, "split")
        assert split[run.phase("Augite")]["tima_labels"] == [2]
        assert "Calcite" not in {v["key"] for v in split.values()}    # absent from the synthetic run
        comb = T.build_mapping(run, "combined")
        assert run.phase("Augite") not in comb
        assert comb[run.phase("Pigeonite")]["name"] == "Pyroxene"
        ek, et = T.exclusions(run, "split")
        assert ek == {-1, run.phase("Xenotime"), run.phase("Weathering Assemblage")}
        assert et == {-1, 5, 9, 10, 11, 12}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tima_core.py -q`
Expected: ImportError.

- [ ] **Step 3: Write `tima_core.py` (part 1)**

```python
"""TIMA raster decode, registration, pyroxene merging, confusion metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np

from . import constants as C
from .io import Run

Mode = Literal["split", "combined"]
_TIMA_RGB = np.array([p.rgb for p in C.TIMA_PHASES], dtype=np.float32)


def read_rgb8(png: Path) -> np.ndarray:
    """RGB uint8 (H, W, 3) from an 8-bit or 16-bit PNG."""
    import imageio.v3 as iio

    arr = iio.imread(png)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, 2)
    arr = arr[:, :, :3]
    if arr.dtype == np.uint16:
        arr = np.rint(arr.astype(np.float32) / 257.0).astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8))


def decode_tima(png: Path, legend_trim: int = C.TIMA_LEGEND_TRIM_ROWS,
                distance_max: float = C.TIMA_COLOR_DISTANCE_MAX) -> np.ndarray:
    arr = read_rgb8(png)
    if legend_trim > 0:
        arr = arr[:-legend_trim, :, :]
    H, W, _ = arr.shape
    arr_f = arr.astype(np.float32)
    channel_range = arr_f.max(axis=2) - arr_f.min(axis=2)
    achromatic = (channel_range < C.TIMA_ACHROMATIC_RANGE) & (arr_f.max(axis=2) > C.TIMA_ACHROMATIC_MIN)
    flat = arr_f.reshape(-1, 3)
    achromatic_flat = achromatic.reshape(-1)
    n = flat.shape[0]
    out = np.empty(n, dtype=np.int8)
    chunk = 8_000_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = flat[start:end, None, :] - _TIMA_RGB[None, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        best = dists.argmin(axis=1)
        best_dist = dists[np.arange(end - start), best]
        seg = best.astype(np.int8)
        seg[best_dist > distance_max] = -1
        seg[achromatic_flat[start:end]] = -1
        out[start:end] = seg
    return out.reshape(H, W)


def align(karak: np.ndarray, tima: np.ndarray, dx: int = C.TIMA_DX, dy: int = C.TIMA_DY):
    h_k, w_k = karak.shape
    h_t, w_t = tima.shape
    x0_t, y0_t = max(0, dx), max(0, dy)
    x1_t, y1_t = min(w_t, dx + w_k), min(h_t, dy + h_k)
    if x1_t <= x0_t or y1_t <= y0_t:
        raise ValueError("Translation produces no overlap")
    x0_k, y0_k, x1_k, y1_k = x0_t - dx, y0_t - dy, x1_t - dx, y1_t - dy
    karak_crop = karak[y0_k:y1_k, x0_k:x1_k]
    tima_crop = tima[y0_t:y1_t, x0_t:x1_t]
    assert karak_crop.shape == tima_crop.shape
    info = {"dx": int(dx), "dy": int(dy),
            "tima_window_xyxy": (int(x0_t), int(y0_t), int(x1_t), int(y1_t)),
            "karak_window_xyxy": (int(x0_k), int(y0_k), int(x1_k), int(y1_k)),
            "shape_hw": (int(karak_crop.shape[0]), int(karak_crop.shape[1]))}
    return karak_crop, tima_crop, info


def combine_pyroxene(run_phase: Callable[[str], int], karak: np.ndarray, tima: np.ndarray):
    k, t = karak.copy(), tima.copy()
    k[k == run_phase("Augite")] = run_phase("Pigeonite")
    for src in C.TIMA_COMBINE_FROM:
        t[t == src] = C.TIMA_COMBINE_INTO
    return k, t


def build_mapping(run: Run, mode: Mode) -> dict[int, dict]:
    order = C.LEGEND_ORDER_SPLIT if mode == "split" else C.LEGEND_ORDER_COMBINED
    table = C.TIMA_MAPPING_SPLIT if mode == "split" else C.TIMA_MAPPING_COMBINED
    out: dict[int, dict] = {}
    for key in order:
        if not run.has(key):
            continue
        name, tima_labels, notes = C.DISPLAY_NAMES.get(key, key), list(table[key]), C.TIMA_MAPPING_NOTES[key]
        if mode == "combined" and key == "Pigeonite":
            name, tima_labels, notes = C.TIMA_COMBINED_NAME, [C.TIMA_COMBINE_INTO], C.COMBINED_PYROXENE_NOTE
        out[run.phase(key)] = {"key": key, "name": name, "tima_labels": tima_labels, "notes": notes}
    return out


def exclusions(run: Run, mode: Mode) -> tuple[set[int], set[int]]:
    karak = {-1} | {run.phase(k) for k in C.EXCLUDED_KARAK_KEYS if run.has(k)}
    tima = {-1} | set(C.EXCLUDED_TIMA_SPLIT if mode == "split" else C.EXCLUDED_TIMA_COMBINED)
    return karak, tima
```

`DISPLAY_NAMES["Augite"]` is `"Augite (high-Ca px)"`, which matches the old `name` field exactly; the old split mapping used the plain names for everything else, as `DISPLAY_NAMES` gives for the other keys except `Fe-Ti-Cr Spinel` (`Fe–Ti–Cr spinel` with en dashes) and `Fe Oxyhydroxide`/`Ferroan Olivine` (lower case). The TIMA figure legends and Table 4 use the plain key for those three: add `MAPPING_NAMES = {**{k: k for k in TIMA_MAPPING_SPLIT}, "Augite": "Augite (high-Ca px)", "Pigeonite": "Pigeonite (low-Ca px)"}` to `constants.py` and use it instead of `DISPLAY_NAMES` in `build_mapping`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tima_core.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git switch main && git pull && git switch -c feat/tima
git add src/nwa4587_paper/tima_core.py src/nwa4587_paper/constants.py tests/test_tima_core.py
git commit -m "feat: TIMA decode, registration and mapping"
```

### Task 3.2: `tima_core.py`, the metrics

**Files:**
- Modify: `src/nwa4587_paper/tima_core.py`
- Test: `tests/test_tima_core.py` (append)

**Interfaces:**
- Produces:
  - `confusion(pipeline: np.ndarray, tima: np.ndarray, mapping: dict[int, dict], excluded_pipeline: set[int], excluded_tima: set[int]) -> Confusion` with `Confusion(cm: np.ndarray, per_phase: list[dict], overall_accuracy: float, pipeline_ids: list[int], tima_ids: list[int], n_included: int)`; `per_phase` entries have `pipeline_label, key, name, tp, fp, fn, precision, recall, f1, pixel_count`. The verbatim port of `R/analysis/tima_comparison.py` L864 to L1017 (`compute_confusion_matrix`) with prints removed and `key` added from `mapping[label]["key"]`.
  - `abundance(pipeline, tima, mapping) -> tuple[list[dict], float, float]`: the verbatim port of L1020 to L1113 (`compute_abundance_correlation`); table entries `pipeline_label, key, name, pipeline_count, pipeline_pct, tima_count, tima_pct, notes`; Spearman from `scipy.stats.spearmanr`; the denominator is every pixel where both labels are >= 0.
  - `rare_phases(pipeline, tima, run, threshold: int = C.RARE_PHASE_THRESHOLD_PX, top_n: int = 5) -> dict` with `karak_phase_counts`, `tima_phase_counts` (string keys), `karak_rare` and `tima_rare` lists of `{label, name, pixel_count, pct_of_mineral, top_destinations: [{tgt_label, tgt_name, count, pct_of_src}]}` as in `R/analysis/tima_rare_phases.py` (no exclusions; -1 counts as a destination named "Background"; names via `run.keys` and `C.TIMA_NAMES`, with the combined pyroxene named "Pyroxene (combined)" when `combined` is true).
  - `agreement_masks(pipeline, tima, mapping, excluded_pipeline, excluded_tima) -> tuple[np.ndarray, np.ndarray]` (`agree`, `included`) as Fig 4 computes them.

Important behaviours to keep (they fix the published numbers): `pixel_count` is the row total after exclusions; FN subtracts the TP of other pipeline phases that claim the same TIMA label; the LUT indexing assumes -1 never survives the mask.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tima_core.py`:
```python
def _toy_mapping():
    return {3: {"key": "Plagioclase", "name": "Plagioclase", "tima_labels": [0], "notes": ""},
            0: {"key": "Ilmenite", "name": "Ilmenite", "tima_labels": [3, 4], "notes": ""},
            11: {"key": "Ferroan Olivine", "name": "Ferroan Olivine", "tima_labels": [4], "notes": ""}}


def test_confusion_counts_and_shared_label_fn():
    pipe = np.array([[3, 3, 3, 0, 0, 11, 11, 9, -1]])
    tima = np.array([[0, 0, 7, 3, 4, 4, 0, 0, 0]])
    res = T.confusion(pipe, tima, _toy_mapping(), {-1, 9}, {-1})
    assert res.n_included == 7
    by = {p["pipeline_label"]: p for p in res.per_phase}
    assert (by[3]["tp"], by[3]["fp"], by[3]["pixel_count"]) == (2, 1, 3)
    assert by[0]["tp"] == 2 and by[0]["fp"] == 0
    # TIMA 4 column total 2: ilmenite own tp 1, olivine's tp 1 is subtracted -> fn 0 for ilmenite from column 4
    assert by[0]["fn"] == 0
    assert by[11]["tp"] == 1 and by[11]["fp"] == 1 and by[11]["fn"] == 0
    assert abs(res.overall_accuracy - 5 / 7) < 1e-12
    assert res.cm.sum() == 7


def test_abundance_denominator_and_double_count():
    pipe = np.array([[3, 3, 0, 11, -1]])
    tima = np.array([[0, 4, 4, 4, 0]])
    table, r, p = T.abundance(pipe, tima, _toy_mapping())
    by = {t["pipeline_label"]: t for t in table}
    assert by[3]["pipeline_pct"] == 50.0                    # 2 of 4 overlapping pixels
    assert by[0]["tima_count"] == 3 and by[11]["tima_count"] == 3   # TIMA 4 counted for both
    assert isinstance(r, float) and isinstance(p, float)


def test_rare_phases_and_agreement(synth_h5):
    with Run.open(synth_h5) as run:
        pipe = run.labels_1x
        tima = np.full(pipe.shape, -1, np.int8)
        tima[pipe == run.phase("Plagioclase")] = 0
        tima[pipe == run.phase("Augite")] = 2
        tima[pipe == run.phase("Pigeonite")] = 1
        rare = T.rare_phases(pipe, tima, run, threshold=100, combined=False)
        xen = next(r for r in rare["karak_rare"] if r["label"] == run.phase("Xenotime"))
        assert xen["top_destinations"][0]["tgt_name"] == "Background"
        mapping = T.build_mapping(run, "split")
        ek, et = T.exclusions(run, "split")
        agree, included = T.agreement_masks(pipe, tima, mapping, ek, et)
        assert agree.sum() == included.sum() == ((tima >= 0) & ~np.isin(pipe, list(ek))).sum()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tima_core.py -q`
Expected: AttributeError on `confusion`.

- [ ] **Step 3: Append the metrics to `tima_core.py`**

```python
from dataclasses import dataclass


@dataclass
class Confusion:
    cm: np.ndarray
    per_phase: list[dict]
    overall_accuracy: float
    pipeline_ids: list[int]
    tima_ids: list[int]
    n_included: int


def confusion(pipeline: np.ndarray, tima: np.ndarray, mapping: dict[int, dict],
              excluded_pipeline: set[int], excluded_tima: set[int]) -> Confusion:
    if pipeline.shape != tima.shape:
        raise ValueError(f"Shape mismatch: pipeline {pipeline.shape} vs TIMA {tima.shape}")
    mask = np.ones(pipeline.shape, dtype=bool)
    for lbl in excluded_pipeline:
        mask &= pipeline != lbl
    for lbl in excluded_tima:
        mask &= tima != lbl
    pl_flat, t_flat = pipeline[mask], tima[mask]
    n_included = int(mask.sum())
    pipeline_ids = sorted(int(x) for x in np.unique(pl_flat))
    tima_ids = sorted(int(x) for x in np.unique(t_flat))
    pl_idx = {lbl: i for i, lbl in enumerate(pipeline_ids)}
    t_idx = {lbl: i for i, lbl in enumerate(tima_ids)}
    cm = np.zeros((len(pipeline_ids), len(tima_ids)), dtype=np.int64)
    if n_included:
        pl_lut = np.full(max(pipeline_ids) + 2, -1, np.int32)
        t_lut = np.full(max(tima_ids) + 2, -1, np.int32)
        for lbl, i in pl_idx.items():
            pl_lut[lbl] = i
        for lbl, i in t_idx.items():
            t_lut[lbl] = i
        np.add.at(cm, (pl_lut[pl_flat.astype(np.intp)], t_lut[t_flat.astype(np.intp)]), 1)
    per_phase = []
    for pl_lbl in pipeline_ids:
        row_i = pl_idx[pl_lbl]
        total = int(cm[row_i, :].sum())
        mapped = mapping.get(pl_lbl, {}).get("tima_labels", [])
        tp = sum(int(cm[row_i, t_idx[t]]) for t in mapped if t in t_idx)
        fp = total - tp
        fn = 0
        for t in mapped:
            if t in t_idx:
                col_j = t_idx[t]
                col_total = int(cm[:, col_j].sum())
                own = int(cm[row_i, col_j])
                other = sum(int(cm[pl_idx[o], col_j]) for o, info in mapping.items()
                            if o != pl_lbl and t in info.get("tima_labels", []) and o in pl_idx)
                fn += col_total - own - other
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        info = mapping.get(pl_lbl, {})
        per_phase.append({"pipeline_label": pl_lbl, "key": info.get("key", f"label_{pl_lbl}"),
                          "name": info.get("name", f"Label {pl_lbl}"), "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1, "pixel_count": total})
    n_agree = sum(int(cm[pl_idx[p], t_idx[t]]) for p in pipeline_ids
                  for t in mapping.get(p, {}).get("tima_labels", []) if t in t_idx)
    acc = n_agree / n_included if n_included > 0 else 0.0
    return Confusion(cm, per_phase, acc, pipeline_ids, tima_ids, n_included)


def abundance(pipeline: np.ndarray, tima: np.ndarray, mapping: dict[int, dict]):
    from scipy.stats import spearmanr

    if pipeline.shape != tima.shape:
        raise ValueError("Shape mismatch")
    overlap = (pipeline >= 0) & (tima >= 0)
    n = int(overlap.sum())
    pl_flat, t_flat = pipeline[overlap], tima[overlap]
    table, p_pcts, t_pcts = [], [], []
    for pl_lbl, info in sorted(mapping.items(), key=lambda kv: kv[0]):
        pl_count = int((pl_flat == pl_lbl).sum())
        t_count = sum(int((t_flat == t).sum()) for t in info["tima_labels"])
        pl_pct = pl_count / n * 100 if n else 0.0
        t_pct = t_count / n * 100 if n else 0.0
        table.append({"pipeline_label": pl_lbl, "key": info["key"], "name": info["name"],
                      "pipeline_count": pl_count, "pipeline_pct": pl_pct,
                      "tima_count": t_count, "tima_pct": t_pct, "notes": info.get("notes", "")})
        p_pcts.append(pl_pct)
        t_pcts.append(t_pct)
    if len(p_pcts) >= 3:
        res = spearmanr(p_pcts, t_pcts)
        return table, float(res.statistic), float(res.pvalue)
    return table, float("nan"), float("nan")


def agreement_masks(pipeline, tima, mapping, excluded_pipeline, excluded_tima):
    included = ~np.isin(pipeline, list(excluded_pipeline)) & ~np.isin(tima, list(excluded_tima))
    agree = np.zeros(pipeline.shape, bool)
    for p, info in mapping.items():
        agree |= (pipeline == p) & np.isin(tima, info["tima_labels"])
    return agree & included, included


def _top_destinations(src, tgt, src_lbl, name_fn, top_n):
    m = src == src_lbl
    n_src = int(m.sum())
    if n_src == 0:
        return []
    vals, counts = np.unique(tgt[m], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return [{"tgt_label": int(vals[i]), "tgt_name": name_fn(int(vals[i])), "count": int(counts[i]),
             "pct_of_src": counts[i] / n_src * 100} for i in order[:top_n]]


def rare_phases(pipeline, tima, run: Run, threshold: int = C.RARE_PHASE_THRESHOLD_PX,
                top_n: int = 5, combined: bool = False) -> dict:
    def kname(lbl):
        if lbl == -1:
            return "Background"
        if combined and lbl == run.phase("Pigeonite"):
            return "Pyroxene (combined)"
        return run.keys.get(lbl, f"karak {lbl}")

    def tname(lbl):
        if lbl == -1:
            return "Background"
        if combined and lbl == C.TIMA_COMBINE_INTO:
            return "Pyroxene (combined)"
        return C.TIMA_NAMES.get(lbl, f"TIMA {lbl}")

    def counts(a):
        v, c = np.unique(a, return_counts=True)
        return {int(x): int(y) for x, y in zip(v, c)}

    def table(src_counts, name_fn, src, other, other_name_fn):
        n_mineral = sum(c for l, c in src_counts.items() if l != -1)
        rare = sorted(((l, c) for l, c in src_counts.items() if l != -1 and 0 < c < threshold),
                      key=lambda lc: -lc[1])
        return [{"label": l, "name": name_fn(l), "pixel_count": c, "pct_of_mineral": c / n_mineral * 100,
                 "top_destinations": _top_destinations(src, other, l, other_name_fn, top_n)} for l, c in rare]

    kc, tc = counts(pipeline), counts(tima)
    return {"threshold": threshold, "combine_pyroxene": combined,
            "karak_phase_counts": {str(k): v for k, v in kc.items()},
            "tima_phase_counts": {str(k): v for k, v in tc.items()},
            "karak_rare": table(kc, kname, pipeline, tima, tname),
            "tima_rare": table(tc, tname, tima, pipeline, kname)}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_tima_core.py -q`
Expected: 8 passed. (Check the toy numbers by hand before changing the test: included pixels are the 7 whose pipeline label is in {3, 0, 11}; agreements are 3-0, 3-0, 0-3, 0-4, 11-4 = 5.)

- [ ] **Step 5: Commit**

```bash
git add src/nwa4587_paper/tima_core.py tests/test_tima_core.py
git commit -m "feat: TIMA confusion, abundance, agreement and rare-phase metrics"
```

### Task 3.3: the `tima` stage: Figs 3, 4, 5, Tables 3, 4, numbers

**Files:**
- Create: `src/nwa4587_paper/tima.py`
- Test: `tests/test_tima.py`

**Interfaces:**
- Produces: `tima.run(run, out, data) -> dict`. Raises `MissingInput` when `data.tima_png` is None (the CLI turns that into exit 2; `paper figures` must treat exit 2 from `tima` as a skip: in `cli.cmd_figures`, print `"tima: skipped (no TIMA raster under data/tima)"` and do not count it in `worst`). Writes:
  - `figures/fig3_phase_comparison.{pdf,png}`, `fig4_agreement_map.{pdf,png}`, `fig5_pyroxene_tima.{pdf,png}`;
  - `tables/table3_correspondence.tex`, `table4_concordance.tex`, `table4_concordance.csv`, `tima_confusion_split.csv`, `tima_confusion_combined.csv`, `tima_rare_phases.json`.
  - Returns:
    ```
    {"alignment": {"dx", "dy", "shape_hw": [h, w]},
     "split": {"overall_accuracy", "spearman_r", "spearman_p", "n_included",
               "per_phase": {key: {"precision", "recall", "f1", "pixel_count", "tp", "fp", "fn"}},
               "abundance": {key: {"pipeline_pct", "tima_pct"}}},
     "combined": {same keys},
     "rare": {key: {"top_tima_name": str, "top_pct": float}}   # karak rare phases, combined mode
     "fig5_window": {"x0", "y0", "size", "n_both_pyroxene"}}
    ```

Pipeline (port of `tima_comparison_1x.main`, `make_fig3_fig4`, `make_fig_pyroxene_tima`, `tima_rare_phases`):
1. `karak = run.labels_1x`; `tima = decode_tima(data.tima_png)`; `k, t, info = align(karak, tima)`.
2. Split: `mapping = build_mapping(run, "split")`, `ek, et = exclusions(run, "split")`, `conf = confusion(k, t, mapping, ek, et)`, `table, r, p = abundance(k, t, mapping)`.
3. Combined: `kc, tc = combine_pyroxene(run.phase, k, t)` and the same with `"combined"`.
4. Rare: `rare_phases(kc, tc, run, combined=True)`; the numbers keep, per karak rare phase, the first destination's name and percentage.
5. Fig 3 and Fig 4 in split mode (the paper's choice; commit 73e3baa of R). Fig 4's annotation uses the split `overall_accuracy` computed here, not a JSON.
6. Fig 5 from the split maps: window `C.FIG5_WINDOW_XY` or, when None, `find_densest_window(k, t, run, C.FIG5_WINDOW_SIZE, C.FIG5_WINDOW_STRIDE)` (verbatim port of `find_densest_pyroxene_window`, `R/analysis/tima_pyroxene_zoom.py` L61 to L89, with the karak labels from `run.phase("Augite")`/`run.phase("Pigeonite")` and TIMA 2/1); log the window.
7. Table 3 from `C.TABLE3_ROWS`, rows whose key the run lacks dropped. Table 4 from the split `per_phase`, sorted by pixel count descending, names from `C.TABLE4_NAMES.get(key, key)`, values `.2f`, pixels with `{,}` separators.
8. Confusion CSVs: rows pipeline keys, columns TIMA names, integer counts.

Figure specifications (from the port cards; sizes in inches, fonts sans-serif 8.5 pt base, `pdf.fonttype 42`):
- Fig 3: `aspect = h / w`; `panel_h = 3.15 * aspect`; `fig_w = 6.69`; `fig_h = panel_h + 0.85 + 0.35`; two panels `wspace 0.04`, `subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.85 / fig_h)`; karak coloured by label with `C.MATCHED_COLORS` overridden by `C.MATCHED_COLORS_SPLIT_OVERRIDE` in split mode; TIMA coloured by the first mapping entry that lists its index (`setdefault` in legend order); background `C.TIMA_BACKGROUND`; titles "(a) Pipeline" and "(b) TIMA (aligned)" bold 9 pt left pad 4; legend entries in `C.LEGEND_ORDER_SPLIT` as `f"{name} [{tima names joined by '/'}]"` (bracket dropped when equal or empty) plus "Background / excluded"; `fig.legend(loc="lower center", ncol=3, fontsize=7, framealpha=0.95, edgecolor="0.7", borderaxespad=0.3, handlelength=1.2, handletextpad=0.5, columnspacing=1.0)`.
- Fig 4: `fig_w = 6.69`, `img_w = 6.69 * 0.65`, `fig_h = img_w * aspect + 0.5`; gridspec `width_ratios=[2.5, 1]`, `wspace 0.08`, `left 0.01 right 0.98 top 0.95 bottom 0.06`; panel (a) RGB image: `C.EXCLUDED_COLOR` everywhere, `C.AGREE_COLOR` on agree, `C.DISAGREE_COLOR` on included and not agree; text box top right `f"Overall agreement\n{acc*100:.1f}%"` bold 9 pt in a white round box; legend lower left 7 pt: `f"Agree ({n:,} px, {pct:.1f}%)"`, `f"Disagree ({n:,} px, {100-pct:.1f}%)"`, "Excluded"; panel (b) stacked `barh` height 0.65 per mapping entry with pixels, sorted by total descending, y inverted, short names from `C.FIG4_SHORT_NAMES`, `f"{agree/total*100:.0f}%"` at `total * 1.02` 6 pt colour 0.3, `xlim(0, max * 1.18)`, scientific x ticks, title "(b) Per-phase breakdown".
- Fig 5: `fig_w = 6.69`, `panel_w = (6.69 - 0.1) / 2`, `fig_h = panel_w + 0.7`; two panels; karak panel: `C.FIG5_NON_PYX` fill, Augite `C.FIG5_HIGH_CA`, Pigeonite `C.FIG5_LOW_CA`; TIMA panel: index 2 high, 1 low; titles "(a) Pipeline (unsupervised)" and "(b) TIMA"; legend "Augite / pyroxene high-Ca", "Pigeonite / pyroxene low-Ca", "Other phase / background", `ncol=3, fontsize=7.5`.

- [ ] **Step 1: Write the failing tests**

`tests/test_tima.py`:
```python
import json
import numpy as np
import pytest

from nwa4587_paper import numbers as N, tima
from nwa4587_paper.io import Run, DataPaths, MissingInput


def test_tima_stage_on_synthetic(synth_h5, synth_tima, tmp_path):
    with Run.open(synth_h5) as run:
        nums = tima.run(run, tmp_path, DataPaths(tima_png=synth_tima))
    N.validate("tima", nums)
    for f in ("fig3_phase_comparison", "fig4_agreement_map", "fig5_pyroxene_tima"):
        assert (tmp_path / "figures" / f"{f}.pdf").exists()
    for f in ("table3_correspondence.tex", "table4_concordance.tex", "table4_concordance.csv",
              "tima_confusion_split.csv", "tima_confusion_combined.csv", "tima_rare_phases.json"):
        assert (tmp_path / "tables" / f).exists()
    # the synthetic raster paints the karak map in TIMA colours, so agreement is perfect
    assert nums["split"]["overall_accuracy"] == pytest.approx(1.0)
    assert nums["combined"]["overall_accuracy"] == pytest.approx(1.0)
    assert nums["split"]["per_phase"]["Augite"]["f1"] == pytest.approx(1.0)
    assert nums["alignment"]["shape_hw"] == [128, 96]
    assert nums["fig5_window"]["size"] <= 128


def test_tima_missing_raster(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        with pytest.raises(MissingInput):
            tima.run(run, tmp_path, DataPaths(tima_png=None))


def test_tima_window_search_is_capped_to_map(synth_h5, synth_tima, tmp_path):
    with Run.open(synth_h5) as run:
        k = run.labels_1x
        x0, y0, n = tima.find_densest_window(k, np.where(k >= 0, 1, -1).astype(np.int8), run, size=40, stride=10)
        assert 0 <= x0 <= k.shape[1] - 40 and 0 <= y0 <= k.shape[0] - 40
```

For the synthetic raster the registration must place the painted map so that `align` recovers it exactly: `synth.write_synthetic_tima` paints at `(C.TIMA_DX - 1900, C.TIMA_DY - 1000)`; the stage therefore needs the offset to be a parameter of `run()`, defaulting to `(C.TIMA_DX, C.TIMA_DY)`. Give `tima.run` keyword arguments `dx: int = C.TIMA_DX, dy: int = C.TIMA_DY` and have the test call `tima.run(run, tmp_path, DataPaths(tima_png=synth_tima), dx=C.TIMA_DX - 1900, dy=C.TIMA_DY - 1000)`. The CLI never passes them. Also cap the Fig 5 window: `size = min(C.FIG5_WINDOW_SIZE, h, w)`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tima.py -q`
Expected: ImportError.

- [ ] **Step 3: Write `tima.py`**

Register the numbers:
```python
N.register({
    "tima.alignment.dx": N.Spec("int"), "tima.alignment.dy": N.Spec("int"),
    "tima.alignment.shape_hw": N.Spec("int"),
    "tima.*.overall_accuracy": N.Spec("float", tol=0.0005, fmt="{:.4f}"),
    "tima.*.spearman_r": N.Spec("float", tol=0.005, fmt="{:.3f}"),
    "tima.*.spearman_p": N.Spec("float", tol=0.5, rel=True, fmt="{:.2e}"),
    "tima.*.n_included": N.Spec("int"),
    "tima.*.per_phase.*.precision": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "tima.*.per_phase.*.recall": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "tima.*.per_phase.*.f1": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "tima.*.per_phase.*.pixel_count": N.Spec("int"),
    "tima.*.per_phase.*.tp": N.Spec("int"), "tima.*.per_phase.*.fp": N.Spec("int"),
    "tima.*.per_phase.*.fn": N.Spec("int"),
    "tima.*.abundance.*.pipeline_pct": N.Spec("float", tol=0.005, fmt="{:.1f}"),
    "tima.*.abundance.*.tima_pct": N.Spec("float", tol=0.005, fmt="{:.1f}"),
    "tima.rare.*.top_tima_name": N.Spec("str"),
    "tima.rare.*.top_pct": N.Spec("float", tol=0.05, fmt="{:.1f}"),
    "tima.fig5_window.x0": N.Spec("int"), "tima.fig5_window.y0": N.Spec("int"),
    "tima.fig5_window.size": N.Spec("int"), "tima.fig5_window.n_both_pyroxene": N.Spec("int"),
})
```
Then `find_densest_window` (verbatim port), `_metrics(run, k, t, mode) -> tuple[Confusion, list, float, float]`, `_fig3`, `_fig4`, `_fig5`, `_tables`, `_csvs` and `run()` as specified above. Keep each figure function under 80 lines; take layout numbers from this task's text, not from new literals elsewhere.

- [ ] **Step 4: Run the tests, then the full suite**

Run: `uv run pytest tests/test_tima.py -q && uv run pytest -q`
Expected: pass. The CLI smoke test now also exercises `tima` because `paper synth` writes the raster; `cmd_figures` must call `run_stage("tima", ...)` with the synthetic offset only through a hidden `--tima-offset DX,DY` option (add it to `_add_common`, default `None` meaning the constants), and `paper synth` prints the command line to use: `--tima-offset 19,51`. Update `test_figures_then_numbers_on_synthetic` to pass it.

- [ ] **Step 5: Commit, push, open PR 3**

```bash
git add src/nwa4587_paper/tima.py src/nwa4587_paper/cli.py tests/test_tima.py tests/test_cli.py
git commit -m "feat: tima stage (Figs 3-5, Tables 3-4, confusion and rare-phase numbers)"
git push -u origin HEAD
gh pr create --title "feat: TIMA comparison stage" --body "Part 3 of the plan. Decoder, registration, metrics, Figs 3 to 5, Tables 3 and 4."
gh pr checks --watch
```

---

# Part 4: `pyroxene`, `xenotime`, `workflow` (PR 4, branch `feat/pyroxene-xenotime-workflow`)

### Task 4.1: the `pyroxene` stage (Fig 7 and the section 3.4 numbers)

**Files:**
- Create: `src/nwa4587_paper/pyroxene.py`
- Test: `tests/test_pyroxene.py`

**Interfaces:**
- Produces: `pyroxene.run(run, out, data) -> dict` writing `figures/fig7_pyroxene_refinement.{pdf,png}`; needs `data.eds_dir` (raises `MissingInput` otherwise). Returns:
  ```
  {"n_pixels_2x": int,                       # Augite + Pigeonite pixels at 2x (the paper's 4.64 M)
   "pct_of_mineral_2x": float,
   "n_pixels_1x": {"Augite": int, "Pigeonite": int},   # from the pyx history record n_pixels_hires
   "pct_1x": {"Augite": float, "Pigeonite": float},
   "gmm_mean": {"Augite": float, "Pigeonite": float},  # history component_means of Ca/(Ca+Mg)
   "ratio_mean": {"Augite": float, "Pigeonite": float},# per-pixel Ca/(Ca+Mg) class means from the 1x PNGs (the old panel c print)
   "n_undefined_1x": int}
  ```
- `pyroxene.load_1x(run, data) -> dict` with `labels` (= `run.labels_1x`), `bse` (`load_bse_1x`), `ca`, `mg` (`load_channel_1x`) cropped to the common shape, `aug_mask`, `pig_mask`, `ratio_aug`, `ratio_pig`, `ratio_all`, and the subsampled `ca_aug, mg_aug, ca_pig, mg_pig` (`rng = np.random.default_rng(C.FIG7_SUBSAMPLE_SEED)`, augite drawn first, `n = C.FIG7_SUBSAMPLE_N`).

Port of `R/paper/figures/make_fig6_pyroxene.py`:
- Panel (a): `pre = run.labels_before("pyx")`, crop at 2x `(r0 // f, r1 // f, c0 // f, c1 // f)` from `C.FIG7_CROP_1X` with `f = run.geometry.factor`; RGB `clip(bse / 0.5, 0, 1)` grey; pixels of the pyroxene parent label (the `pyx` record's `parent`) blended `0.5 * grey + 0.5 * C.PYROXENE_UNDIFF_COLOR`; title "2x downsample" bold 8; one legend patch "Pyroxene (undiff.)".
- Panel (b): the 1x crop `C.FIG7_CROP_1X` of `bse` with Augite and Pigeonite blended at alpha 0.55 in `C.PALETTE`; title "1x native resolution"; two legend patches.
- Panel (c): histograms of the ratio, bins `linspace(0.10, 0.75, 110)`, density, Pigeonite then Augite at alpha 0.5, "All pyroxene" as a black step of width 0.7, dashed class-mean lines; xlabel "Ca / (Ca + Mg)", ylabel "Density", top and right spines hidden.
- Panel (d): `gaussian_kde(vstack([mg, ca]), bw_method=0.06)` per class on a 150 by 150 grid Mg 0.06 to 0.36, Ca 0.04 to 0.42; `contourf` at 5 levels `linspace(0.05 max, 0.95 max, 5)` with alphas `linspace(0.08, 0.40, 5)`, `contour` lines at `levels[1::2]` width 0.7; centroid `+` markers white then class colour; xlabel "Mg intensity", ylabel "Ca intensity".
- Figure 6.693 by 5.689 in, `gridspec_kw=dict(hspace=0.28, wspace=0.32, top=0.95, bottom=0.08, left=0.08, right=0.97)`, panel labels "(a)" to "(d)" at `(-0.06, 1.04)` bold 9.
- Numbers: `n_pixels_2x = counts[Augite] + counts[Pigeonite]` (2x); the 1x counts and the GMM means come from `run.split_record("pyx")`; `ratio_mean` from the 1x arrays.

- [ ] **Step 1: Write the failing tests**

`tests/test_pyroxene.py`:
```python
import pytest

from nwa4587_paper import numbers as N, pyroxene
from nwa4587_paper.io import Run, DataPaths, MissingInput


def test_pyroxene_numbers_and_figure(synth_h5, synth_eds, tmp_path):
    with Run.open(synth_h5) as run:
        nums = pyroxene.run(run, tmp_path, DataPaths(eds_dir=synth_eds))
        rec = run.split_record("pyx")
        aug, pig = run.phase("Augite"), run.phase("Pigeonite")
        assert nums["n_pixels_2x"] == run.counts[aug] + run.counts[pig]
        assert nums["n_pixels_1x"]["Augite"] == rec["n_pixels_hires"][str(aug)]
        assert nums["gmm_mean"] == {"Augite": 0.53, "Pigeonite": 0.31}
        assert nums["ratio_mean"]["Augite"] == pytest.approx(0.6, abs=0.03)     # Ca 0.6, Mg 0.4
        assert nums["ratio_mean"]["Pigeonite"] == pytest.approx(0.2, abs=0.03)  # Ca 0.2, Mg 0.8
    N.validate("pyroxene", nums)
    assert (tmp_path / "figures" / "fig7_pyroxene_refinement.pdf").exists()


def test_pyroxene_needs_eds(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        with pytest.raises(MissingInput):
            pyroxene.run(run, tmp_path, DataPaths(eds_dir=None))
```

The synthetic crop: `C.FIG7_CROP_1X` lies outside a 128 by 96 map. `run()` clips the crop to the map (`r1 = min(r1, h)` and so on) and, when the clipped crop is empty, uses the pyroxene region's bounding box at 1x. Add a test assertion that the figure exists in that case (the first test already covers it).

- [ ] **Step 2: Run to verify failure, write `pyroxene.py`, run again**

Run: `uv run pytest tests/test_pyroxene.py -q` (ImportError). Register:
```python
N.register({
    "pyroxene.n_pixels_2x": N.Spec("int"),
    "pyroxene.pct_of_mineral_2x": N.Spec("float", tol=0.005, fmt="{:.1f}"),
    "pyroxene.n_pixels_1x.*": N.Spec("int"),
    "pyroxene.pct_1x.*": N.Spec("float", tol=0.05, fmt="{:.1f}"),
    "pyroxene.gmm_mean.*": N.Spec("float", tol=0.0005, fmt="{:.2f}"),
    "pyroxene.ratio_mean.*": N.Spec("float", tol=0.002, fmt="{:.2f}"),
    "pyroxene.n_undefined_1x": N.Spec("int"),
})
```
Write the module per the port above. Expected after: 2 passed.

- [ ] **Step 3: Commit**

```bash
git switch main && git pull && git switch -c feat/pyroxene-xenotime-workflow
git add src/nwa4587_paper/pyroxene.py tests/test_pyroxene.py
git commit -m "feat: pyroxene stage (Fig 7 and the section 3.4 numbers)"
```

### Task 4.2: the `xenotime` stage (Fig 6)

**Files:**
- Create: `src/nwa4587_paper/xenotime.py`
- Test: `tests/test_xenotime.py`

**Interfaces:**
- Produces: `xenotime.run(run, out, data) -> dict` writing `figures/fig6_xenotime_callout.{pdf,png}`; returns `{"n_pixels_2x": int, "n_components": int, "largest_component_px": int, "bbox_2x": [r0, c0, r1, c1], "roi_2x": [r0, c0, r1, c1], "centroid_2x": [row, col]}`.
- `xenotime.region(labels: np.ndarray, label: int, pad: int) -> tuple[dict, slice, slice]`: components with `scipy.ndimage.label` (4-connectivity, the old default), the largest one, its bounding box padded by `pad` and squared as `R/paper/figures/make_fig7_xenotime.py` does (expand the shorter side by `diff // 2` before and `diff - diff // 2` after, clipped).

Port of `make_fig7_xenotime.py` with the canonical palette instead of tab20: figure `(3.54, 4.0)` in; `GridSpec(3, 2, height_ratios=[1, 1, 0.03], hspace=0.22, wspace=0.08, left=0.02, right=0.98, top=0.93, bottom=0.18)`; (a) BSE grey with a gold `#FFD700` contour of the dilated xenotime mask (width 1.0), title "BSE", scale bar in white from the verbatim `_add_scalebar` (pixel size `run.geometry.pixel_size_um_2x`, nice lengths `[10, 20, 25, 50, 100, 200, 250, 500, 1000]`, 20 % of the view width); (b) the labels coloured with `C.PALETTE` by key (`C.UNASSIGNED_COLOR` for -1, light grey `(0.85, 0.85, 0.85)` for a label without a palette entry), white contour, title "Phase map"; (c) Y and (d) P from `run.channel`, cmap "hot", shared `vmax = max(y.max(), p.max())`, cyan contour, titles "Y-L intensity" and "P-K intensity"; two horizontal colourbars labelled "Relative intensity"; panel letters white on a black box at `(0.04, 0.96)`; figure legend of the keys present in the region, `ncol = min(n, 5)`, fontsize 4.5, lower center.

- [ ] **Step 1: Write the failing tests**

`tests/test_xenotime.py`:
```python
import numpy as np

from nwa4587_paper import numbers as N, xenotime
from nwa4587_paper.io import Run, DataPaths


def test_region_is_square_and_padded():
    img = np.full((100, 100), -1)
    img[40:44, 50:60] = 10
    img[90, 90] = 10
    info, rs, cs = xenotime.region(img, 10, pad=5)
    assert info["n_components"] == 2 and info["largest_component_px"] == 40
    assert (rs.stop - rs.start) == (cs.stop - cs.start)
    assert rs.start <= 35 and cs.start <= 45 and cs.stop >= 65


def test_xenotime_stage(synth_h5, tmp_path):
    with Run.open(synth_h5) as run:
        nums = xenotime.run(run, tmp_path, DataPaths())
    N.validate("xenotime", nums)
    assert nums["n_pixels_2x"] == 16 and nums["n_components"] == 1
    assert nums["largest_component_px"] == 16
    assert (tmp_path / "figures" / "fig6_xenotime_callout.png").exists()
```

- [ ] **Step 2: Run (ImportError), write `xenotime.py`, run again (2 passed)**

Register:
```python
N.register({
    "xenotime.n_pixels_2x": N.Spec("int"), "xenotime.n_components": N.Spec("int"),
    "xenotime.largest_component_px": N.Spec("int"),
    "xenotime.bbox_2x": N.Spec("int"), "xenotime.roi_2x": N.Spec("int"),
    "xenotime.centroid_2x": N.Spec("float", tol=0.5, fmt="{:.0f}"),
})
```
`run()` uses `run.phase("Xenotime")`, `run.labels`, `run.bse`, `run.channel("Y")`, `run.channel("P")`, `C.XENOTIME_PAD_PX_2X`. The Y and P channels may be absent in another sample: raise `MissingInput("channel Y not in denoised cube")` when `"Y"` or `"P"` is not in `run.element_order`.

- [ ] **Step 3: Commit**

```bash
git add src/nwa4587_paper/xenotime.py tests/test_xenotime.py
git commit -m "feat: xenotime stage (Fig 6) with the canonical palette"
```

### Task 4.3: the `workflow` stage (Fig 1)

**Files:**
- Create: `src/nwa4587_paper/workflow.py`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Produces: `workflow.run(run, out, data) -> dict` writing `figures/fig1_workflow.{pdf,png}`; needs `data.eds_dir` for the jet PNG; returns `{"n_clusters_before_pyx": int, "n_phases_final": int}`.

Port of `R/paper/figures/make_fig1_workflow.py`:
- (a) the jet `C.FIG1_ELEMENT` PNG, banner rows = `run.geometry.header_trim_px` removed, stride `C.FIG1_PNG_STRIDE`; title "(a) Jet RGB input".
- (b) `run.raw_channel("Ca")[::8, ::8]` grey with `vmax` = 99.5th percentile of the positive values; "(b) Inverted scalar".
- (c) `where(mask, clip(denoised Ca / vmax, 0, 1), 0.95)` as grey RGB; "(c) Mask + denoise".
- (d) `labels_before("pyx")[::8, ::8]` coloured by `C.PALETTE` through the run's keys (the pyroxene parent uses `PALETTE["Pyroxene"]`, apatite `PALETTE["Apatite"]`); title `f"(d) {n} clusters"` where `n` is the number of distinct labels >= 0 in that map.
- (e) the final labels coloured the same way; "(e) Named phases".
- Layout in absolute inches: `FIG_W = 17 / 2.54`, `margin 0.06`, `gap 0.20`, `panel_w = (FIG_W - 0.12 - 0.8) / 5`, `panel_h = panel_w * aspect`, `title_h 0.22`, `note_h 0.16`, `legend_h 0.44`; four `FancyArrowPatch` arrows `"->,head_width=2.5,head_length=4"` colour `#444444` width 1.1 between panels at mid height; the note "colormap inversion: only for false-colour exports" italic 5.5 pt `#555555` under the first gap; legend of every final key ordered by descending count in the thumbnail, `ncol 5`, 5.5 pt, no frame; strides `C.FIG1_H5_STRIDE`; background grey 0.95; saved without `bbox_inches`.

- [ ] **Step 1: Write the failing test**

`tests/test_workflow.py`:
```python
from nwa4587_paper import numbers as N, workflow
from nwa4587_paper.io import Run, DataPaths


def test_workflow_figure(synth_h5, synth_eds, tmp_path):
    with Run.open(synth_h5) as run:
        nums = workflow.run(run, tmp_path, DataPaths(eds_dir=synth_eds))
    N.validate("workflow", nums)
    assert nums["n_clusters_before_pyx"] == 7       # 3, 4, 10, 11, 12, 2, 15/16 -> {2,3,4,10,11,12,15,16} minus none = 8? see note
    assert (tmp_path / "figures" / "fig1_workflow.pdf").exists()
```
Compute the expected count from the synthetic design before running: `labels_before("pyx")` undoes `pyx` and `phos`, so the labels are {2, 3, 4, 7, 10, 11, 12}, which is 7. Keep `7` and delete the trailing comment.

- [ ] **Step 2: Run (ImportError), write `workflow.py`, run again (1 passed)**

Register `{"workflow.n_clusters_before_pyx": N.Spec("int"), "workflow.n_phases_final": N.Spec("int")}`. The stride on the synthetic 64 by 48 map gives an 8 by 6 thumbnail; that is fine for the test.

- [ ] **Step 3: Commit, push, open PR 4**

```bash
git add src/nwa4587_paper/workflow.py tests/test_workflow.py
git commit -m "feat: workflow stage (Fig 1)"
git push -u origin HEAD
gh pr create --title "feat: pyroxene, xenotime and workflow stages" --body "Part 4 of the plan. Figs 7, 6 and 1 and the section 3.4 numbers."
gh pr checks --watch
```

---

# Part 5: `lamellae` and `colormap` (PR 5, branch `feat/lamellae-colormap`)

### Task 5.1: `lamellae_core.py`

**Files:**
- Create: `src/nwa4587_paper/lamellae_core.py`
- Test: `tests/test_lamellae.py`

**Interfaces:**
- Produces (verbatim ports of `output/nwa4587/item2/paperflow/acceptance/lamellae_from_hires.py` in the karak checkout, lines 53 to 265, prints removed, settings from `C.LAMELLAE`):
  - `binary_map(hires: np.ndarray, augite: int, pigeonite: int) -> np.ndarray` (int8: augite 1, pigeonite 0, else -1).
  - `segment_grains(label_1x, min_size) -> tuple[np.ndarray, int]`.
  - `measure_orientation(augite_binary, grain_mask, sigma) -> tuple[float, float]`.
  - `measure_spacing(augite_binary, grain_mask, orientation_deg, n_profiles) -> tuple[float, float, int]`.
  - `rayleigh_test(angles_deg) -> tuple[float, float]`.
  - `analyze(hires, augite, pigeonite, pixel_size_um, settings=C.LAMELLAE) -> tuple[dict, list[dict]]`: the summary dict and the grain records with exactly the keys of the old script (`grain_id, area_px, area_um2, equiv_diameter_um, centroid_row, centroid_col, bbox, augite_fraction, orientation_deg, coherence, spacing_px, spacing_um, spacing_iqr_um, n_profiles, near_nyquist`); the summary has `n_grains_total, n_grains_analyzed, pixel_size_um, nyquist_limit_um, orientation{circular_mean_deg, rayleigh_R_bar, rayleigh_p_value, uniform}, spacing{mean_um, median_um, std_um, q25_um, q75_um, min_um, max_um, n_near_nyquist}, grain_size_vs_spacing{spearman_rho, spearman_p}, coherence{mean, min, max}`. With zero analyzed grains every float is `nan`, `uniform` is `None`, and nothing raises.

Core code to port (settings threaded through; the body is the old code):

```python
def segment_grains(label_1x, min_size):
    from scipy.ndimage import label as ndlabel, binary_closing
    from skimage.morphology import remove_small_objects
    pyx_mask = label_1x >= 0
    struct = np.ones((3, 3), dtype=bool)
    pyx_closed = binary_closing(pyx_mask, structure=struct, iterations=C.LAMELLAE.closing_iterations)
    grain_labels, n_raw = ndlabel(pyx_closed, structure=struct)
    grain_labels = remove_small_objects(grain_labels, max_size=min_size - 1, connectivity=2)
    unique = np.unique(grain_labels)
    return grain_labels, int((unique > 0).sum())
```
(`remove_small_objects(max_size=min_size - 1)` keeps objects of size >= `min_size`, the old `min_size=` semantics without the deprecation warning. Check `skimage.__version__ >= 0.26`; otherwise pass `min_size=min_size`.)

`measure_orientation`, `measure_spacing` and `rayleigh_test`: copy the bodies from the acceptance script lines 89 to 265 unchanged except `sigma=C.LAMELLAE.tensor_sigma` and `n_profiles` from the argument. `analyze` is the body of the old `main()` from "Analyzing grains" to the `summary` dict (lines 300 to 437): `regionprops` on the grain labels, keep `area >= settings.min_analysis_px`, skip augite fraction < 0.01 or > 0.99, skip NaN orientation or coherence < `settings.coherence_min`, skip NaN spacing or `n_valid < 2`; per-grain conversions `spacing_um = spacing_px * pixel_size_um`, `area_um2 = area * pixel_size_um ** 2`, `near_nyquist = spacing_px < settings.nyquist_px`; summary exactly as the old code, with the guarded empty case.

- [ ] **Step 1: Write the failing tests**

`tests/test_lamellae.py`:
```python
import math
import numpy as np

from nwa4587_paper import lamellae_core as L


def striped_grain(h=300, w=300, period=20, angle_deg=30.0):
    """A disc of radius 120 filled with augite/pigeonite stripes of the given period and angle."""
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < 120 ** 2
    th = math.radians(angle_deg)
    coord = xx * math.sin(th) - yy * math.cos(th)         # distance across the stripes
    aug = ((coord // (period / 2)) % 2 == 0)
    hires = np.full((h, w), -1, np.int16)
    hires[disc & aug] = 14
    hires[disc & ~aug] = 13
    return hires


def test_binary_map_and_segmentation():
    hires = striped_grain()
    b = L.binary_map(hires, augite=14, pigeonite=13)
    assert set(np.unique(b)) == {-1, 0, 1}
    grains, n = L.segment_grains(b, min_size=500)
    assert n == 1


def test_orientation_and_spacing_recovered():
    hires = striped_grain(period=20, angle_deg=30.0)
    b = L.binary_map(hires, 14, 13)
    grain = b >= 0
    orient, coh = L.measure_orientation(b == 1, grain, sigma=1.5)
    assert coh > 0.2
    assert min(abs(orient - 30.0), abs(orient - 210.0), abs(orient - 120.0), abs(orient + 60.0)) < 6.0 \
        or min(abs(orient - 30.0), abs(abs(orient - 30.0) - 180.0)) < 6.0
    spacing_px, iqr, n = L.measure_spacing(b == 1, grain, orient, n_profiles=20)
    assert n >= 2 and abs(spacing_px - 20.0) <= 2.0


def test_rayleigh_uniform_and_concentrated():
    r_u, p_u = L.rayleigh_test(np.linspace(0, 180, 36, endpoint=False))
    r_c, p_c = L.rayleigh_test(np.full(30, 40.0))
    assert p_u > 0.5 and r_c > 0.99 and p_c < 1e-6


def test_analyze_summary_keys_and_values():
    summary, grains = L.analyze(striped_grain(), augite=14, pigeonite=13, pixel_size_um=3.0)
    assert summary["n_grains_total"] == 1 and summary["n_grains_analyzed"] == 1
    assert summary["nyquist_limit_um"] == 6.0
    assert set(summary) == {"n_grains_total", "n_grains_analyzed", "pixel_size_um", "nyquist_limit_um",
                            "orientation", "spacing", "grain_size_vs_spacing", "coherence"}
    assert abs(summary["spacing"]["median_um"] - 60.0) <= 6.0
    assert grains[0]["near_nyquist"] is False and grains[0]["area_px"] > 40000


def test_lamellae_no_grains():
    hires = np.full((50, 50), -1, np.int16)
    hires[10:14, 10:14] = 14                                  # 16 px: below min_grain_px
    summary, grains = L.analyze(hires, 14, 13, pixel_size_um=3.0)
    assert summary["n_grains_total"] == 0 and summary["n_grains_analyzed"] == 0 and grains == []
    assert math.isnan(summary["spacing"]["median_um"]) and summary["orientation"]["uniform"] is None
```

The orientation convention: the old script reports the lamellae direction (`theta_lam = theta_grad + pi/2`) modulo 180. For stripes whose normal is at 30 degrees the lamellae direction is 120 degrees. Run the test once, read the value, and keep the single assertion that matches the convention (expected: within 6 degrees of 120). Replace the two-line `or` assertion with that one.

- [ ] **Step 2: Run to verify failure, write the module, run again**

Run: `uv run pytest tests/test_lamellae.py -q` (ImportError), write `lamellae_core.py`, run again. Expected: 5 passed. The spacing test allows 2 px because the autocorrelation peak sits on an integer lag.

- [ ] **Step 3: Commit**

```bash
git switch main && git pull && git switch -c feat/lamellae-colormap
git add src/nwa4587_paper/lamellae_core.py tests/test_lamellae.py
git commit -m "feat: lamellae segmentation and measurements ported from the published analysis"
```

### Task 5.2: the `lamellae` stage (Fig 8)

**Files:**
- Create: `src/nwa4587_paper/lamellae.py`
- Test: `tests/test_lamellae.py` (append)

**Interfaces:**
- Produces: `lamellae.run(run, out, data) -> dict` writing `figures/fig8_lamellae.{pdf,png}`, `tables/lamellae_grain_table.csv` (the record keys minus `bbox`), `tables/lamellae_summary.json`; returns the summary dict exactly (keys above).

Fig 8, the port of `R/paper/figures/make_fig_lamellae.py` with the factor from the run: width 6.693 in, height 0.85 times that; `subplots(2, 2)`, `hspace 0.35 wspace 0.35`; colours `C.PALETTE["Augite"]`, `C.PALETTE["Pigeonite"]`, `C.NYQUIST_COLOR`, median line in the Augite colour.
- (a) BSE close-up: `f = run.geometry.factor`, `ps = run.geometry.pixel_size_um_1x`; centroids are 1x pixels, so the 2x position is `centroid / f` (the old script divided by a wrong 4); median centroid, an 800 by 800 2x window around it, `imshow(clip(bse / 0.5, 0, 1), gray, bilinear, extent=[c0 * f * ps / 1000, c1 * f * ps / 1000, r1 * f * ps / 1000, r0 * f * ps / 1000])` in mm; per grain inside the window a line of half-length 0.3 mm along the orientation, width 1.5, viridis by spacing (`Normalize` 5th to 95th percentile), centroid dot size 3 white edge; colourbar "Spacing (µm)" 7 pt; a 0.5 mm white scale bar labelled "500 µm" bold 6 pt.
- (b) rose diagram on a polar axes placed over the hidden rectangular one: 18 bins of 10 degrees weighted by `area_um2`, normalised, mirrored at +pi, Pigeonite colour alpha 0.7, theta zero north, direction -1, `rlabel_position 225`, mean direction as a dashed Augite-coloured line at mu and mu + pi, stats box `"R̄ = {:.2f}\n{p}\nμ = {:.0f}°"` with `p` as "p < 0.001" or `p = {:.3f}`, title `n = {n}`.
- (c) spacing histogram: clip at `1.3 * P98`, `n_bins = min(25, max(10, n // 3))`, Nyquist dashed line labelled "Nyquist (6 µm)", median solid labelled `"Median ({median:.0f} µm)"`, IQR band alpha 0.15 labelled `"IQR ({q25:.0f}–{q75:.0f} µm)"`, legend 5.5 pt, clipped-count note.
- (d) area against spacing, log both axes, hsv by orientation (0 to 180), `yerr = iqr / 2`, colourbar "Orientation (°)", text `"ρ = {rho:.2f}, {p}"`.
- Panel letters at `(-0.06, 1.04)` bold 10.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_lamellae.py`:
```python
def test_lamellae_stage(synth_h5, tmp_path):
    from nwa4587_paper import lamellae, numbers as N
    from nwa4587_paper.io import Run, DataPaths
    with Run.open(synth_h5) as run:
        nums = lamellae.run(run, tmp_path, DataPaths())
    N.validate("lamellae", nums)
    assert nums["pixel_size_um"] == 3.0 and nums["nyquist_limit_um"] == 6.0
    assert (tmp_path / "figures" / "fig8_lamellae.pdf").exists()
    assert (tmp_path / "tables" / "lamellae_grain_table.csv").exists()
```
The synthetic pyroxene region is 56 by 48 px at 1x with 4-px stripes: one grain of 2688 px, so `n_grains_analyzed` is 0 or 1 depending on coherence; the figure must render in both cases (empty panels with the `n = 0` title when nothing is analyzed).

- [ ] **Step 2: Run (ImportError), write `lamellae.py`, run again (6 passed)**

Register:
```python
N.register({
    "lamellae.n_grains_total": N.Spec("int"), "lamellae.n_grains_analyzed": N.Spec("int"),
    "lamellae.pixel_size_um": N.Spec("float", tol=0.0, fmt="{:.1f}"),
    "lamellae.nyquist_limit_um": N.Spec("float", tol=0.0, fmt="{:.1f}"),
    "lamellae.orientation.circular_mean_deg": N.Spec("float", tol=0.5, fmt="{:.1f}"),
    "lamellae.orientation.rayleigh_R_bar": N.Spec("float", tol=0.005, fmt="{:.2f}"),
    "lamellae.orientation.rayleigh_p_value": N.Spec("float", tol=0.01, fmt="{:.2f}"),
    "lamellae.orientation.uniform": N.Spec("bool"),
    "lamellae.spacing.*": N.Spec("float", tol=0.5, fmt="{:.0f}"),
    "lamellae.spacing.n_near_nyquist": N.Spec("int"),
    "lamellae.grain_size_vs_spacing.spearman_rho": N.Spec("float", tol=0.01, fmt="{:.2f}"),
    "lamellae.grain_size_vs_spacing.spearman_p": N.Spec("float", tol=0.5, rel=True, fmt="{:.3f}"),
    "lamellae.coherence.*": N.Spec("float", tol=0.01, fmt="{:.2f}"),
})
```
`run()` calls `lamellae_core.analyze(run.hires_labels, run.phase("Augite"), run.phase("Pigeonite"), run.geometry.pixel_size_um_1x)`, writes the CSV and JSON, draws Fig 8 with `run.bse` and `run.geometry`, returns the summary.

- [ ] **Step 3: Commit**

```bash
git add src/nwa4587_paper/lamellae.py tests/test_lamellae.py
git commit -m "feat: lamellae stage (Fig 8) with the factor from the HDF5"
```

### Task 5.3: the `colormap` stage

**Files:**
- Create: `src/nwa4587_paper/colormap.py`
- Test: `tests/test_colormap.py`

**Interfaces:**
- Produces: `colormap.run(run, out, data, settings=C.COLORMAP) -> dict` (the `run` argument is ignored and may be None; `cli.NEEDS_RUN["colormap"] = False`); writes no figure (the paper dropped it); returns `{cmap: {"mae", "median_ae", "max_ae", "rmse", "p90", "p99", "n_exact", "n_unique_rgb"} for cmap in settings.cmaps}` plus `"jet_regions": {region: {"n", "mae", "max_ae"}}` and `"lut_step": 1 / (lut_n - 1)`.
- `build_lut(cmap_name, lut_n) -> np.ndarray` (256 cubed float32, black = 0), `forward_map(scalars, cmap_name) -> uint8 (N, 3)`, `invert_via_lut(rgb, lut)`, `analyse(cmap_name, n_test, lut) -> dict`, `classify_jet_region(scalar) -> str`: verbatim from `R/paper/colormap_roundtrip_analysis.py` (region thresholds 0.12 dark blue, 0.35 cyan, 0.55 green, 0.75 yellow, 0.88 orange-red, else dark red). `build_lut` may delegate to `karak.io.loaders.get_full_lut(f"cmap:{name}")` when `lut_n == 4096` because karak builds the same table; keep the own implementation as the reference and test that both agree on 1000 random colours.

- [ ] **Step 1: Write the failing tests**

`tests/test_colormap.py`:
```python
import numpy as np

from nwa4587_paper import colormap as CM, constants as C, numbers as N


def test_roundtrip_small_lut_is_accurate():
    settings = C.ColormapSettings(lut_n=256, n_test=2000, cmaps=("jet",))
    nums = CM.run(None, None, None, settings=settings)
    N.validate("colormap", nums)
    assert nums["jet"]["mae"] < 0.01 and nums["jet"]["max_ae"] < 0.05
    assert set(nums["jet_regions"]) == {"dark blue", "cyan", "green", "yellow", "orange-red", "dark red"}
    assert nums["lut_step"] == 1 / 255


def test_lut_matches_karak_on_samples():
    from karak.io.loaders import get_full_lut
    ours = CM.build_lut("jet", 4096)
    theirs = get_full_lut("cmap:jet")
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 256, size=(1000, 3))
    assert np.allclose(ours[idx[:, 0], idx[:, 1], idx[:, 2]], theirs[idx[:, 0], idx[:, 1], idx[:, 2]], atol=1e-6)
```
The 4096 LUT takes about 20 s to build; mark the second test `@pytest.mark.slow` and register the marker in `pyproject.toml` (`markers = ["slow: builds the full 256^3 LUT"]`). CI runs it.

- [ ] **Step 2: Run (ImportError), write `colormap.py`, run again (2 passed)**

Register:
```python
N.register({
    "colormap.lut_step": N.Spec("float", tol=1e-9, fmt="{:.6f}"),
    "colormap.*.mae": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.median_ae": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.max_ae": N.Spec("float", tol=1e-4, fmt="{:.2e}"),
    "colormap.*.rmse": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.p90": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.p99": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.*.n_exact": N.Spec("int"), "colormap.*.n_unique_rgb": N.Spec("int"),
    "colormap.jet_regions.*.n": N.Spec("int"),
    "colormap.jet_regions.*.mae": N.Spec("float", tol=1e-5, fmt="{:.2e}"),
    "colormap.jet_regions.*.max_ae": N.Spec("float", tol=1e-4, fmt="{:.2e}"),
})
```
`run()` writes `out/numbers/colormap.json` through the CLI like every stage (it returns the dict; `out` may be None in the unit test, so `run()` itself writes nothing). The paper's quoted values are `mae` about 0.0014 (0.14 %) and `max_ae` about 0.016 (1.6 %) for jet at `lut_n` 4096 and `n_test` 100 000.

- [ ] **Step 3: Commit, push, open PR 5**

```bash
git add src/nwa4587_paper/colormap.py tests/test_colormap.py pyproject.toml
git commit -m "feat: colormap round-trip numbers"
git push -u origin HEAD
gh pr create --title "feat: lamellae and colormap stages" --body "Part 5 of the plan. Fig 8 from clusters/hires/labels with the factor from the HDF5, and the colormap round-trip numbers."
gh pr checks --watch
```

---

# Part 6: data access, real-data acceptance, README (PR 6, branch `feat/data-acceptance`)

### Task 6.1: `paper data`, `paper karak`, `paper all`

**Files:**
- Create: `src/nwa4587_paper/data.py`
- Modify: `src/nwa4587_paper/cli.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `data.copy_inputs(src: Path, data_dir: Path) -> dict` copies `*.png` elemental maps and `mask/Valid_mask.csv` from `src` into `data_dir/eds/`, and a TIMA raster into `data_dir/tima/phases.png` when `src/TIMA results/*Phases.png` or `src/tima/*.png` exists. Returns `{"n_png": int, "mask": bool, "tima": bool}`. Existing files are left in place unless `--force`.
  - `data.zenodo_stub(doi: str) -> int` prints "Zenodo download is not available yet; sub-project 5 deposits the data. Place the files by hand: ..." and returns 2.
  - `data.run_karak(data_dir, out_dir, workers: int = 0, extra: list[str] = ()) -> int` runs `karak run --builtin paper --input data_dir/eds --out out_dir/paperflow --workers <n> --plain` through `subprocess.run` with the `karak` console script of the current environment (`shutil.which("karak")`, else `python -m karak.flow run`), streams output, and returns its exit code. The HDF5 lands at `out_dir/paperflow.h5`, which is the default `--h5`.
  - CLI: `paper data --from PATH | zenodo:DOI [--data DIR] [--force]`; `paper karak [--data DIR] [--out DIR] [--workers N] [-- EXTRA...]`; `paper all` = `data` (only when `--from` is given) then `karak` then `figures` then `numbers`.

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
from pathlib import Path
from unittest import mock

from nwa4587_paper import cli, data


def test_copy_inputs(tmp_path, synth_eds):
    src = tmp_path / "src"
    src.mkdir()
    for p in synth_eds.glob("*.png"):
        (src / p.name).write_bytes(p.read_bytes())
    (src / "mask").mkdir()
    (src / "mask" / "Valid_mask.csv").write_text("index,shape-type,vertex-index,axis-0,axis-1\n")
    (src / "TIMA results").mkdir()
    (src / "TIMA results" / "X-Phases.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    info = data.copy_inputs(src, tmp_path / "data")
    assert info == {"n_png": 3, "mask": True, "tima": True}
    assert (tmp_path / "data" / "eds" / "SYN-Ca.png").exists()
    assert (tmp_path / "data" / "tima" / "phases.png").exists()


def test_zenodo_stub_exits_2(capsys):
    assert data.zenodo_stub("10.5281/zenodo.0000000") == 2
    assert "not available yet" in capsys.readouterr().out


def test_run_karak_builds_the_command(tmp_path):
    with mock.patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert data.run_karak(tmp_path / "data", tmp_path / "results", workers=4) == 0
        argv = run.call_args.args[0]
        assert "--builtin" in argv and "paper" in argv and "--workers" in argv and "4" in argv
        assert str(tmp_path / "results" / "paperflow") in argv


def test_cli_data_from(tmp_path, synth_eds):
    assert cli.main(["data", "--from", str(synth_eds), "--data", str(tmp_path / "d")]) == 0
    assert cli.main(["data", "--from", "zenodo:10.5281/zenodo.1", "--data", str(tmp_path / "d")]) == 2
```

- [ ] **Step 2: Run (ImportError), write `data.py` and the CLI additions, run again (4 passed)**

In `cli.py` add the three subparsers; `all` calls `cmd_data` (when `--from`), `data.run_karak`, `cmd_figures`, `cmd_numbers` in order and stops at the first non-zero code except the `tima` skip.

- [ ] **Step 3: Commit**

```bash
git switch main && git pull && git switch -c feat/data-acceptance
git add src/nwa4587_paper/data.py src/nwa4587_paper/cli.py tests/test_data.py
git commit -m "feat: paper data, paper karak and paper all"
```

### Task 6.2: real-data acceptance, expected numbers, Fig 5 window, README, citation (local)

**Files:**
- Create: `expected/numbers.json`, `CITATION.cff`, `docs/acceptance-2026.md`
- Modify: `README.md`, `src/nwa4587_paper/constants.py` (`FIG5_WINDOW_XY`), `tests/test_constants.py`

This task runs on the user's machine with the real data. It changes no algorithm.

- [ ] **Step 1: Stage the inputs**

```bash
cd ~/Projects/science/nwa4587-karak-paper
uv run paper data --from /home/brendon/Dropbox/Projects/izawa/NWA_4587_data
mkdir -p data/tima && cp "/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/data/TIMA results/NAW 4587-2_S3858-Phases.png" data/tima/phases.png
mkdir -p results && cp /home/brendon/Projects/science/karak/output/nwa4587/item2/paperflow.h5 results/paperflow.h5
```
(`cp` reads the paper repository; it writes nothing there. The copied HDF5 is the 2026-10-07 reference run, so `paper karak` is not rerun here; a second acceptance with a fresh `paper karak` run is a follow-up once sub-project 4 pins a release.)

- [ ] **Step 2: Run every stage and record the time**

```bash
time uv run paper figures 2>&1 | tee results/acceptance.log
uv run paper numbers
```
Expected: exit 0; `results/figures/fig1_workflow.pdf` to `fig8_lamellae.pdf`, `results/tables/table1_params.tex` to `table4_concordance.tex`, `results/numbers.json`, `results/numbers.tex`. The `tima` stage logs the Fig 5 window it chose.

- [ ] **Step 3: Freeze the Fig 5 window**

Set `FIG5_WINDOW_XY = (x0, y0)` in `constants.py` to the logged values, add `assert C.FIG5_WINDOW_XY is not None` to `test_offsets_and_geometry`, rerun `uv run paper tima` and confirm `numbers/tima.json` carries the same window.

- [ ] **Step 4: Check the numbers against the known values**

Compare `results/numbers.json` by hand with these values from the reference run and the paper; every one must hold before the expected file is committed:

| key | expected |
|---|---|
| `tables.n_mineral_pixels` | 12495787 |
| `tables.n_phases_with_pixels` | 15 |
| `tables.phases.Plagioclase.pct` | 28.923 |
| `tables.phases.Pigeonite.pct` | 19.320 |
| `tables.phases.Augite.pct` | 17.760 |
| `tables.phases.Weathering Assemblage.pct` | 14.140 |
| `tables.phases.Epoxy.pct` | 8.725 |
| `tables.phases.Xenotime.pixels` | 213 |
| `pyroxene.n_pixels_2x` | 4633456 |
| `pyroxene.n_pixels_1x.Augite` / `.Pigeonite` | 8904186 / 9629508 |
| `pyroxene.pct_1x.Augite` / `.Pigeonite` | 48.0 / 52.0 |
| `pyroxene.gmm_mean.Augite` / `.Pigeonite` | 0.5336 / 0.3056 |
| `lamellae.n_grains_total` / `n_grains_analyzed` | 272 / 154 |
| `lamellae.spacing.median_um` | 45.0 |
| `lamellae.orientation.rayleigh_p_value` | 0.80 (within 0.01) |
| `xenotime.n_pixels_2x` / `largest_component_px` | 213 / 200 |
| `colormap.jet.mae` / `max_ae` | about 0.0014 / 0.016 |
| `tima.alignment.shape_hw` | [12685, 7980] |
| `tima.split.overall_accuracy` | record; compare with the published 0.8407 |
| `tima.combined.overall_accuracy` | record; compare with the published 0.9910 |
| `tima.split.n_included` | record; compare with the published 37,640,743 |

The TIMA values come from `clusters/hires/labels` for the first time. Differences from the published numbers are expected to be small because the 1x composition matches the published split to 0.1 pp; report them in the PR and in `docs/acceptance-2026.md` with the published value beside each.

- [ ] **Step 5: Commit the expected numbers and check**

```bash
cp results/numbers.json expected/numbers.json
uv run paper numbers --check          # expected: "all numbers match", exit 0
uv run pytest -q
```

- [ ] **Step 6: Compare the figures with the published PDFs**

Open each `results/figures/figN_*.pdf` beside `R/paper/figures/<old name>.pdf` (fig1_workflow, fig5_fingerprints, fig3_phase_comparison, fig4_agreement_map, fig_pyroxene_tima, fig7_xenotime_callout, fig6_pyroxene_refinement, fig8_lamellae). Note every visible difference in `docs/acceptance-2026.md`: expected ones are the 13/14 colour assignment (by name now), the xenotime palette, the Fig 8 panel (a) extent (correct factor), and the lamellae statistics.

- [ ] **Step 7: README and CITATION**

`README.md` sections: what the repository regenerates (the paper's Table 1 to 4 and Fig 1 to 8 numbering with the output file names); requirements (Python 3.12, uv, about 4 GB RAM for the TIMA stage, the 2.2 GB HDF5); quick start (`uv sync`, `paper data --from PATH`, either `paper karak` (45 min CPU) or copy the deposited `paperflow.h5` into `results/`, `paper figures`, `paper numbers --check`); the data (not in the repository; raw PNGs and HDF5 from the authors until the Zenodo deposit; the TIMA raster under a TESCAN agreement, optional); the numbers bridge (`numbers.json`, `numbers.tex`, `expected/numbers.json`, tolerances in `numbers.py`); known differences from the published paper (15 phases with pixels, the lamellae correction with the old and new values, the TIMA metrics recomputed from the hires labels); licensing (MIT code, CC BY 4.0 results); how to cite.

`CITATION.cff`: `cff-version 1.2.0`, `title`, `authors` (Hall, Brendon; Izawa, Matthew), `repository-code`, `license: MIT`, and a `preferred-citation` of type `article` for the paper with the journal and year as the user supplies them (ask for the final reference; use "in revision, 2026" until then).

- [ ] **Step 8: Commit, push, open PR 6**

```bash
git add expected/numbers.json src/nwa4587_paper/constants.py tests/test_constants.py README.md CITATION.cff docs/acceptance-2026.md
git commit -m "feat: real-data acceptance, expected numbers, README and citation"
git push -u origin HEAD
gh pr create --title "feat: real-data acceptance and documentation" --body-file docs/acceptance-2026.md
gh pr checks --watch
```

---

## Self-review notes

- Spec coverage: section 1 (layout, `paper data`, `paper karak`, optional TIMA) is Tasks 1.1, 2.2 and 6.1; section 2 (`Run`, the eight stages, the numbers bridge, the four rulings) is Tasks 1.4, 2.1 to 5.3; section 3 (constants) is Task 1.2 with `MAPPING_NAMES` added in Task 3.1; section 4 (fixture, tests, CI, acceptance, licensing, delivery order) is Tasks 1.3, 2.4, 6.2 and the six PR boundaries.
- Deviation from the spec, with reason: `labels_before` is keyed by node id as the spec says, but the id comes from `cluster_config.refinement_nodes`, because the history records carry no node id. The `pipeline_config` attribute is YAML, not JSON; `Run.flow` parses either. The spec's colormap "seed 0" is not needed: the analysis is deterministic (`linspace`), so `ColormapSettings` has no seed.
- Type consistency: every stage is `run(run: Run, out: Path, data: DataPaths) -> dict`; `tima.run` adds keyword-only `dx`, `dy`; `colormap.run` adds `settings`. `numbers.Spec`, `register`, `validate`, `write_stage`, `merge`, `render_tex`, `check` are used with the same names in Tasks 2.2 to 6.1. `tima_core.confusion` returns `Confusion`; `tima.py` reads `.per_phase`, `.overall_accuracy`, `.n_included`, `.cm`, `.pipeline_ids`, `.tima_ids`.
- Review Focus items are pinned: 1 in Task 1.4 (`test_phase_unknown_key`), 2 in Task 3.1 (`test_align_no_overlap`), 3 in Task 1.4 (`test_labels_1x_clips`), 4 in Task 5.1 (`test_lamellae_no_grains`), 5 in Task 2.1 (`test_check_missing_key`).
- Known soft spots for the implementer: the synthetic TIMA raster must be painted at the offset the test passes (`dx = C.TIMA_DX - 1900`, `dy = C.TIMA_DY - 1000`); `imageio.v3.imwrite` of a `uint16` array is the 16-bit PNG writer; `remove_small_objects` changed its keyword in scikit-image 0.26.
