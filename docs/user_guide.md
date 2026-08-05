# Karak User Guide

Karak is an unsupervised mineral phase mapping pipeline for SEM-EDS element
map stacks. This guide covers input data expectations, the five pipeline
stages, the full configuration schema, the HDF5 output layout, and how to
resume interrupted runs.

For installation and a quickstart, see the [README](../README.md).

## Contents

1. [Input data](#input-data)
2. [Pipeline stages](#pipeline-stages)
3. [Configuration reference](#configuration-reference)
4. [HDF5 output layout](#hdf5-output-layout)
5. [Checkpointing and resume](#checkpointing-and-resume)
6. [CLI reference](#cli-reference)
7. [Reproducibility](#reproducibility)
8. [Computational requirements](#computational-requirements)

---

## Input data

### Element maps (required)

One image file per element channel, all with identical dimensions, placed in
`input_dir` and matched by `loader.file_glob` (default `*.png`).

- **False-color maps** (the common case): RGB images produced by SEM-EDS
  vendor software using a colormap such as jet. Karak inverts these back to
  scalar relative intensities in [0, 1] using a precomputed RGB-to-scalar
  lookup table (see `loader.colormap`). Pure black pixels `[0, 0, 0]` are
  interpreted as below-detection / no-data and map to 0.0.
- **Raw scalar maps**: grayscale images can be ingested via a custom linear
  grayscale LUT (`loader.colormap: "lut:PATH"`), or by any matplotlib
  colormap name (`"cmap:gray"`).

Recovered values are *relative intensities*, not quantified wt% compositions.
Karak therefore uses per-channel z-score normalization rather than
compositional (CLR/ILR) transforms.

**Element names** are extracted from filenames:

- Default (legacy TIMA convention): the basename is split on `-` and parts
  from index 2 onward are joined, so `NAW 4587-2_S3858-Fe-K.png` yields
  `Fe-K`.
- Custom naming: set `loader.filename_pattern` with an `{element}`
  placeholder (and optional `{sample}` wildcard), e.g.
  `"Map_NWA-5218_eds_{element}.bmp"`.

### BSE / SEM image (recommended)

A backscatter electron image, identified by matching the parsed element name
to `bse_channel` (default `SEM`) or given explicitly via
`loader.bse_filename`. The BSE image must be **true grayscale**
(R == G == B); it is kept separate from the element cube and is not
colormap-inverted.

### Valid-region polygon mask (optional)

A [napari](https://napari.org) shapes CSV export
(`mask.valid_mask_path`) restricting processing to the sample region
(excluding epoxy, labels, and mount edges). Expected columns:

```
index, shape-type, vertex-index, axis-0, axis-1
```

Only `polygon` shapes are rasterized; open `path` shapes are skipped.
Vertex coordinates are in **original (pre-trim, pre-downsample) image
space** — Karak scales and offsets them automatically to match the working
resolution.

---

## Pipeline stages

The pipeline runs five sequential stages, each checkpointed to HDF5:

| Stage | What it does |
|-------|--------------|
| `load` | Discover element map files, trim annotation strips, downsample, invert the colormap to scalar [0, 1] intensities, stack into an (H, W, C) cube, and load the BSE image. |
| `mask` | Rasterize the valid-region polygon (if provided), then flag pixels that are zero across *all* element channels as background/epoxy. The mineral mask is the intersection; components smaller than `mask.min_object_size` are removed. |
| `denoise` | Edge-aware smoothing of the raw [0, 1] cube, channel by channel — bilateral filter (default) or anisotropic (Perona-Malik) diffusion. Operating on raw intensities preserves grain boundaries and physical signal. |
| `normalize` | Per-channel z-score normalization: mean and standard deviation are computed over **mineral pixels only**, then `z = (x - mean) / std`. Non-mineral pixels are set to 0. |
| `cluster` | PCA dimensionality reduction, HDBSCAN density-based clustering (global or tiled strategy), kNN reassignment of noise pixels, and optional post-clustering refinement. Per-cluster chemical fingerprints are computed from the denoised cube. |

Cluster labels are anonymous phases (0, 1, 2, ...). Assigning mineral names
is a human-in-the-loop step: inspect the per-cluster chemical fingerprints
and QC figures, then record names with
`karak.io.storage.save_mineral_names()`.

---

## Configuration reference

All parameters live in a single YAML file validated by Pydantic
(`karak.config.PipelineConfig`). The CLI auto-detects
`data/pipeline_config.yaml` or `../data/pipeline_config.yaml` relative to
the working directory, or accepts an explicit path via `karak -c
path/to/config.yaml`. If no config is found, a default one is created at
`data/pipeline_config.yaml`.

Relative paths (`input_dir`, `hdf5_output`, `figure_dir`,
`mask.valid_mask_path`) are resolved **relative to the config file's
directory**, so a config can travel with its data.

### Top level (`PipelineConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `input_dir` | `"../data"` | Directory containing raw element map images. |
| `hdf5_output` | `"../data/eds_pipeline.h5"` | Path for the output HDF5 file. |
| `figure_dir` | `"../data/figures"` | Directory for QC diagnostic figures. |
| `exclude_elements` | `["Fe-L"]` | Element channels to exclude from the cube (e.g. redundant lines). |
| `bse_channel` | `"SEM"` | Name of the BSE/SEM channel (kept separate from the cube). |

### `loader` (`LoaderConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `file_glob` | `"*.png"` | Glob pattern (relative to `input_dir`) for element map files. |
| `filename_pattern` | `null` | Pattern with `{element}` placeholder (optional `{sample}` wildcard) to extract element names. `null` = legacy heuristic: split basename on `-`, join parts[2:]. |
| `bse_filename` | `null` | Exact filename of the BSE channel when it does not match `file_glob`. `null` = find BSE within the glob by matching `bse_channel`. |
| `colormap` | `"cmap:jet"` | Colormap spec for inversion: `cmap:NAME` (matplotlib colormap) or `lut:PATH` ((N, 3) uint8 `.npy` LUT; relative paths resolve against `input_dir`). Bare `jet` is accepted as shorthand. |

The first run with a given colormap builds a 256³ RGB-to-scalar lookup
table (30–60 s for a 4096-entry palette). It is cached on disk under the
package's `.cache/` directory and reused by all later runs.

### `downsample` (`DownsampleConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `header_trim_px` | `100` | Pixels trimmed from the top of each image (scale bar region). |
| `bottom_trim_px` | `0` | Pixels trimmed from the bottom (annotation strip). |
| `left_trim_px` | `0` | Pixels trimmed from the left edge. |
| `right_trim_px` | `0` | Pixels trimmed from the right edge (annotation/colorbar strip). |
| `downsample_factor` | `2` | Integer downsampling factor applied to all images (BSE and EDS alike). |

### `mask` (`MaskConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `min_object_size` | `100` | Remove connected components smaller than this (pixels) from the mineral mask. |
| `valid_mask_path` | `null` | Path to a napari shapes CSV defining the sample boundary polygon (coordinates in original image space). `null` = no polygon restriction. |

### `normalize` (`NormalizeConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `method` | `"zscore"` | Normalization method. Only `zscore` (per-channel zero-mean unit-variance on mineral pixels) is supported. |

### `denoise` (`DenoiseConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `method` | `"bilateral"` | `bilateral` or `anisotropic_diffusion`. |
| `sigma_color` | `null` | Bilateral color sigma (`null` = auto from data range). |
| `sigma_spatial` | `1.0` | Bilateral spatial sigma. |
| `niter` | `10` | Anisotropic diffusion iterations. |
| `kappa` | `50` | Conductance coefficient for diffusion. |
| `gamma` | `0.1` | Diffusion speed (0–0.25 stable). |
| `option` | `2` | Perona-Malik option (1 = favours high contrast, 2 = wide regions). |

### `cluster` (`ClusterConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `strategy` | `"global"` | `global` (single HDBSCAN run) or `tiled` (per-tile HDBSCAN with phase registry merging). |

#### `cluster.pca` (`PCAConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `n_components` | `null` | PCA components to retain. `null` = keep all; the researcher picks a cutoff from the scree plot at the checkpoint. |
| `subsample_fraction` | `null` | Fraction of mineral pixels subsampled for PCA fitting. `null` = use all. |
| `random_state` | `42` | Seed for the subsample draw and the PCA solver. |

#### `cluster.hdbscan` (`HDBSCANConfig`)

| Field | Default | Meaning |
|-------|---------|---------|
| `min_cluster_size` | `1000` | Minimum HDBSCAN cluster size (pixels). |
| `min_samples` | `null` | HDBSCAN `min_samples`. `null` = defaults to `min_cluster_size`. |
| `subsample_n` | `null` | Max pixels for HDBSCAN fitting; the rest are assigned via `approximate_predict`. `null` = use all (set to e.g. 500000 for large images). |
| `noise_reassign_k` | `5` | Number of neighbors for kNN reassignment of noise pixels. |
| `random_state` | `42` | Seed for the subsample draw. |

#### `cluster.tiled` (`TiledConfig`) — used when `strategy: tiled`

| Field | Default | Meaning |
|-------|---------|---------|
| `tile_size` | `512` | Tile side length in pixels. |
| `merge_threshold` | `0.92` | Cosine similarity threshold for matching tile clusters to the cross-tile phase registry. |
| `min_tile_pixels` | `null` | Minimum mineral pixels for a tile to be processed. `null` = `2 * hdbscan.min_cluster_size`. |
| `min_clusters_per_tile` | `3` | Tiles with fewer HDBSCAN clusters are deferred to the final kNN pass (avoids single-class tile artifacts). |

#### `cluster.rare_phase` (`RarePhaseConfig`) — optional Pass 2

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Enable the two-pass workflow (Pass 1 major phases, Pass 2 rare phases on unassigned pixels). |
| `min_cluster_size` | `50` | Minimum cluster size for the rare-phase HDBSCAN (much smaller than the main pass). |
| `min_samples` | `null` | `min_samples` for Pass 2. `null` = defaults to `min_cluster_size`. |
| `subsample_n` | `500000` | Max unassigned pixels for Pass 2 fitting; rest via `approximate_predict`. `null` = all. |
| `merge_threshold` | `null` | Cosine threshold for matching rare clusters to the registry. `null` = same as `tiled.merge_threshold`. |

#### `cluster.refinement` (`RefinementConfig`) — optional post-clustering split

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Enable post-clustering refinement (applied after kNN noise reassignment). |
| `target_phase` | `2` | Cluster label of the phase to refine. |

`cluster.refinement.olivine` (`OlivineExtractionConfig`):

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Extract ferroan olivine from the target phase by Fe/Ca thresholds. |
| `fe_threshold` | `0.6` | Minimum denoised Fe-K intensity for olivine pixels. |
| `ca_threshold` | `0.10` | Maximum denoised Ca intensity for olivine pixels. |

`cluster.refinement.gmm_split` (`GMMSplitConfig`):

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Split the target phase into sub-phases with a Gaussian Mixture Model. |
| `n_components` | `2` | Number of GMM components (e.g. pigeonite + augite). |
| `features` | `["Ca", "Mg", "Fe-K", "BSE"]` | Feature channels: element names plus `BSE`. A `Ca/(Ca+Mg)` ratio feature is added automatically when both Ca and Mg are present. |
| `bse_weight` | `1.0` | Weight multiplier for the BSE feature. |
| `subsample_n` | `500000` | Max pixels for GMM fitting; rest assigned via `predict`. `null` = all. |
| `random_state` | `42` | Seed for the subsample draw and GMM initialization (`n_init=5`). |

---

## HDF5 output layout

All results are written to a single HDF5 file (`hdf5_output`) with gzip
compression:

```
/
├── raw/          Per-element (H, W) float32 arrays; attrs: element_names
│                 (JSON), n_elements, height, width
├── bse/          image dataset; attrs: original_shape, downsample_factor
├── masks/        mineral (bool), valid (bool, if provided); mask statistics
│                 as attrs
├── denoised/     cube (H, W, C) float32; attrs: method parameters,
│                 element_order (JSON)
├── normalized/   cube (H, W, C) float32; means, stds datasets; attrs:
│                 element_order (JSON)
├── clusters/     raw_labels, cleaned_labels, probabilities,
│   │             pca_variance_ratio, mineral_indices, cluster_stats;
│   │             attrs: mineral_names (JSON), cluster_N_name
│   └── tiled/    per-tile summaries and phase registry (tiled strategy)
└── attrs:        pipeline_config (full YAML), created (UTC),
                  pipeline_version, python_version, platform,
                  library_versions (JSON)
```

Each group receives a `stage_completed` UTC-timestamp attribute when its
stage finishes. Matching `load_*` functions in `karak.io.storage` read every
group back for downstream analysis or resume.

## Checkpointing and resume

Because each stage checkpoints to HDF5, an interrupted or crashed run can
resume without recomputing earlier stages:

```bash
karak                          # skips stages already marked complete
karak --from-stage denoise     # force re-run from denoise onward
karak --clean                  # delete the HDF5 and start fresh
```

`--from-stage STAGE` loads all prior stages' data from the HDF5 file and
re-executes `STAGE` and everything after it. Valid stage names: `load`,
`mask`, `denoise`, `normalize`, `cluster`. Typical use: tweak clustering
parameters in the config, then `karak --from-stage cluster` to re-cluster
without re-running preprocessing.

## CLI reference

```
karak [-c CONFIG] [--from-stage STAGE] [--no-qc] [--test-mode] [--clean] [-v]
```

| Flag | Meaning |
|------|---------|
| `-c, --config` | Path to `pipeline_config.yaml` (default: auto-detect `data/pipeline_config.yaml` or `../data/pipeline_config.yaml`; creates a default if none found). |
| `--from-stage` | Force re-run from this stage onward (loads prior data from HDF5). |
| `--no-qc` | Skip QC figure generation. |
| `--test-mode` | Fast validation: 4 elements (Fe-K, Ca, Mg, Si) at 4x downsample. |
| `--clean` | Delete the previous HDF5 file and start fresh. |
| `-v` | Debug logging. |

## Reproducibility

- **Seeded randomness** — every stochastic step (PCA subsampling and solver,
  HDBSCAN fit subsampling, GMM fitting) is seeded through a `random_state`
  config field (default 42). Two runs with the same config, data, and
  library versions produce identical outputs.
- **Embedded provenance** — the output HDF5 file records the full pipeline
  config YAML, library versions, Python version, platform string, and
  per-stage completion timestamps, making every result file
  self-documenting.
- **Locked dependencies** — the repository ships a `uv.lock` file;
  `uv sync` reproduces the exact tested environment.

## Computational requirements

Tested on an AMD Ryzen AI 5 340 with 32 GB RAM (Linux).

- A full-resolution run on a ~7400 x 5400 px, 21-channel dataset uses
  **~10–12 GB RAM** (raw float32 cube ~3.2 GB plus working copies).
- For large images, bound memory with `cluster.hdbscan.subsample_n`
  (e.g. 500000) or the `tiled` strategy, which keeps per-tile memory
  constant regardless of image size.
- `karak --test-mode` (4 elements, 4x downsample) validates an installation
  in minutes on a laptop.
- No GPU is required; all computation is CPU-based.
