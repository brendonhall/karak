# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Karak is an automated mineralogy pipeline for SEM-EDS (Scanning Electron Microscopy with Energy Dispersive Spectroscopy) elemental map stacks. It transforms jet-colormapped elemental map PNGs into mineral phase maps with chemical fingerprints. Python >=3.12, built with Hatchling.

## Commands

```bash
pip install -e .                    # install for development
karak                               # run full pipeline (expects config.yaml in data/ dir)
karak --test-mode                   # fast validation: 4 elements, 4x downsample
karak --from-stage denoise          # resume from a specific stage
karak --no-qc                       # skip QC figure generation
karak --clean                       # delete previous HDF5 and start fresh
karak -c path/to/config.yaml        # use specific config file
karak -v                            # enable debug logging
```

No unit test suite exists yet. QC validation is done via diagnostic plots generated at each pipeline stage.

## Architecture

### Pipeline Stages

The pipeline runs 5 sequential stages with HDF5 checkpoint/resume. Each stage reads from and writes to an HDF5 file, enabling resume from any prior stage via `--from-stage`.

1. **Load** — Read PNG element maps, invert jet colormap to scalar values (cached 256³ LUT), build compositional cube, apply downsampling
2. **Mask** — Create mineral mask from elemental data, apply optional polygon mask, compute statistics
3. **Denoise** — Bilateral or anisotropic diffusion filtering on cube channels
4. **Normalize** — Per-channel z-score normalization across mineral pixels
5. **Cluster** — PCA dimensionality reduction + HDBSCAN clustering, with optional tiled progressive strategy and two-pass (major + rare phases) workflow

### Source Layout (`src/karak/`)

- **`cli/runner.py`** — Main entry point and stage orchestrator. All stages are wired here with Rich progress UI.
- **`config.py`** — Pydantic models for YAML config. `PipelineConfig` is the root; sub-configs for each stage. Config is embedded in HDF5 for provenance.
- **`io/`** — `loaders.py` (PNG loading, jet inversion, downsampling), `masks.py` (background masking, polygon masks), `storage.py` (HDF5 read/write with provenance metadata)
- **`preprocessing/`** — `denoise.py` (bilateral/anisotropic filters), `compositional.py` (z-score normalization)
- **`clustering/`** — `pca.py`, `hdbscan_cluster.py`, `noise_assign.py` (k-NN reassignment), `tiling.py` (spatial tiling + progressive phase registry unification), `refinement.py` (post-clustering GMM splits)
- **`identification/`** — `fingerprint.py` (per-cluster chemical signatures)
- **`qc/`** — `figures.py` and `tiled_figures.py` (diagnostic visualizations per stage)

### Key Design Patterns

- **Config-driven**: All parameters live in YAML config validated by Pydantic. No hardcoded values for processing parameters.
- **Pure functions**: Processing modules take NumPy arrays + config, return arrays. Side effects (I/O) are isolated in `io/` and `cli/runner.py`.
- **HDF5 provenance**: Every run stores full config YAML, library versions, and timestamps as HDF5 attributes.
- **Checkpoint/resume**: Stage completion is tracked in HDF5. The `--from-stage` flag reloads prior results and continues.
- **Headless rendering**: Matplotlib uses Agg backend; all UI is Rich-based terminal output.
