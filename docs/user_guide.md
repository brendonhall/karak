# Karak User Guide

Karak is an unsupervised mineral phase mapping pipeline for SEM-EDS element
map stacks. This guide covers input data expectations, the pipeline stages,
flows and their parameters, the HDF5 output layout, and caching.

For installation and a quickstart, see the [README](../README.md).

## Contents

1. [Input data](#input-data)
2. [Pipeline stages](#pipeline-stages)
3. [Payloads](#payloads)
4. [Flows](#flows)
5. [Parameters](#parameters)
6. [HDF5 output layout](#hdf5-output-layout)
7. [Caching and re-runs](#caching-and-re-runs)
8. [CLI reference](#cli-reference)
9. [Reproducibility](#reproducibility)
10. [Computational requirements](#computational-requirements)

Full per-stage port and parameter tables: [stage reference](stage_reference.md).

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

Each processing step is a stage: a small module with typed parameters and
named input/output ports. Ports carry immutable payloads (see
[Payloads](#payloads)); cubes are tagged with a space (`raw`, `denoised`,
`normalized`) and label arrays with a state (`raw`, `cleaned`), written
below as e.g. `cube:raw`. The **[stage reference](stage_reference.md)**
documents every stage's full port and parameter contract, generated from
the registry; `karak schema` prints the same contract as JSON.

| Stage | Inputs → Outputs | What it does |
|-------|------------------|--------------|
| `load_elements` | – → `cube:raw`, `bse` | Discover element map files, trim annotation strips, downsample, invert the colormap to scalar [0, 1] intensities, stack into an (H, W, C) cube, and load the BSE image. |
| `mask` | `cube:raw` → `masks` | Rasterize the valid-region polygon (if provided), then flag pixels that are zero across *all* element channels as background/epoxy. The mineral mask is the intersection; components smaller than `min_object_size` are removed. |
| `denoise` | `cube:raw`, `masks` → `cube:denoised` | Edge-aware smoothing of the raw [0, 1] cube, channel by channel — bilateral filter (default) or anisotropic (Perona-Malik) diffusion. Operating on raw intensities preserves grain boundaries and physical signal. |
| `normalize` | `cube:denoised`, `masks` → `cube:normalized` | Per-channel z-score normalization: mean and standard deviation are computed over **mineral pixels only**, then `z = (x - mean) / std`. Non-mineral pixels are set to 0. |
| `pca` | `cube:normalized`, `masks` → `features` | PCA fit + projection of mineral pixels. `n_components: 0` auto-selects the first count reaching 95% cumulative variance (minimum 5). |
| `hdbscan_global` | `features` → `labels:raw` | Single HDBSCAN run over all mineral-pixel features. |
| `hdbscan_tiled` | `features`, `cube:denoised` → `labels:raw`, `tiles` | Per-tile HDBSCAN with cosine-similarity phase-registry merging across tiles. |
| `rare_phase` | `labels:raw`, `features`, `cube:denoised`, `tiles` → `labels:raw`, `tiles` | Recluster still-unassigned pixels with more sensitive parameters (Pass 2 of the two-pass workflow). Including this stage in a flow is what enables the workflow. |
| `noise_assign` | `labels:raw`, `features` → `labels:cleaned` | Distance-weighted k-NN reassignment of every remaining unlabeled pixel. |
| `refine` | `labels:cleaned`, `cube:denoised`, `bse` → `labels:cleaned` | Composite-phase splitting: threshold-based olivine extraction, then a GMM split of the target phase. |
| `cluster_stats` | `labels:cleaned` → `stats` | Cluster counts, sizes, and noise fraction. |
| `fingerprints` | `labels:cleaned`, `cube:denoised` → `fingerprints` | Per-cluster mean/std element intensities from the denoised cube, with cosine-similar pairs flagged. |
| `export_h5` | ten optional inputs → sink | Writes the provenance HDF5 file; only connected groups are written (see [HDF5 output layout](#hdf5-output-layout)). |
| `qc_mask` | `bse`, `masks` (+`cube:raw`) → sink | Mask coverage overlay with optional TIMA reference panel. |
| `qc_denoise` | `cube:raw`, `cube:denoised`, `bse`, `masks` → sink | Before/after denoising comparison panels. |
| `qc_normalize` | `cube:normalized`, `masks` → sink | Z-score histograms and channel correlation matrix. |
| `qc_scree` | `features` → sink | PCA explained-variance scree plot with the selection cutoff. |
| `qc_phase_map` | `labels:raw`, `labels:cleaned`, `bse`, `stats` → sink | Raw vs cleaned phase map over the BSE image. |
| `qc_cluster_summary` | `stats` → sink | Cluster size and probability summary chart. |
| `qc_tiled` | `bse`, `tiles`, `features` → sink | Tile grid overlay and phase discovery chart. |
| `qc_fingerprints` | `fingerprints` → sink | Per-cluster chemical fingerprint chart. |
| `qc_named_phase_map` | `labels:cleaned`, `bse` → sink | Final phase map with researcher-assigned mineral names. |

Cluster labels are anonymous phases (0, 1, 2, ...). Assigning mineral names
is a human-in-the-loop step: inspect the per-cluster chemical fingerprints
and QC figures, then record names with
`karak.io.storage.save_mineral_names()` and render a named map with the
`qc_named_phase_map` stage. `preprocessing.denoise.compare_denoisers()` is
a notebook helper for side-by-side denoiser comparison.

---

## Payloads

Stages exchange immutable dataclasses (`karak.stages.payloads`). Each
payload serializes to HDF5 for the node cache, and `apply()` returns new
payloads via `.replace()` instead of mutating inputs.

| Payload | Carried by ports | Contents |
|---------|------------------|----------|
| `ElementCube` | `cube`, `cube_raw`, `cube_denoised`, `cube_normalized` | (H, W, C) float32 pixels, element names, a `space` tag, per-channel means/stds when normalized, and the downsample/trim geometry (so polygon masks rasterize without a side channel). |
| `BseImage` | `bse` | (H, W) float32 backscatter-electron image. |
| `MaskSet` | `masks` | Boolean mineral mask, optional valid-region mask, and mask statistics. |
| `PCAFeatures` | `features` | (N_mineral, n_kept) float32 features, (N_mineral, 2) pixel coordinates, image shape, full explained-variance ratios, and the kept component count. |
| `Labels` | `labels`, `labels_raw` | (N_mineral,) int32 labels, optional membership probabilities, pixel coordinates, image shape, and a `state` tag (`raw` may contain -1; `cleaned` never does). |
| `TiledArtifacts` | `tiles` | Per-tile diagnostic results and the global phase registry from tiled clustering. |
| `ClusterStats` | `stats` | Cluster counts, sizes, and noise fraction as a dict. |
| `Fingerprints` | `fingerprints` | Per-cluster mean/std element spectra, element ranking, and flagged similar pairs. |

---

## Flows

A flow is a JSON DAG of stages: `nodes` (stage `type` + `params`) connected
by `edges` (output port to input port). Four builtins ship with karak:

| Flow | Clustering path |
|------|-----------------|
| `global` | `hdbscan_global → noise_assign` |
| `tiled` | `hdbscan_tiled → noise_assign` |
| `tiled-rare` | `hdbscan_tiled → rare_phase → noise_assign` |
| `stepwise` | none yet: load and mask steps only |

`stepwise` grows one step at a time as steps join the dashboard work;
today it runs the load step (`src`) and the mask step (`msk`). The
builtin leaves `msk.valid_mask_path` at `null` (no polygon); set it in your
own copy (`karak flow init --builtin stepwise -o FILE`), for example to
`"{input}/mask/Valid_mask.csv"`.

```bash
karak run --builtin global --input data/ --out output/sample
karak run my_flow.json --input data/ --out output/sample
karak run --builtin global --set hdb.min_cluster_size=500 --set pca.n_components=8
karak validate my_flow.json
```

To author a custom flow, start from a builtin with `karak flow init
--builtin NAME -o my_flow.json` and edit it. Editors can check a flow file
against [`flow.schema.json`](flow.schema.json), a JSON Schema generated from
the stage registry (one node variant per stage, every parameter required,
with types, choices, bounds, and template values). String params accept run-scoped tokens: `{input}` (the
`--input` directory), `{out}` (the `--out` basename), `{work}` (the work
directory). `karak validate` reports structural errors — unknown stages,
bad parameter values, unconnected required inputs, port type mismatches
(for example wiring a raw cube into a stage that expects a denoised one),
and cycles — before anything runs.

---

## Parameters

A flow JSON lists every parameter of every node. A run takes no parameter
value from code: `karak validate` and `karak run` reject a flow with a
missing parameter, and name it. The [stage reference](stage_reference.md)
lists each stage's parameters with their template values, bounds, and help
text. Template values fill a node only when you ask for it:

- `karak flow init --builtin NAME -o FILE` writes a builtin flow with every
  parameter spelled out.
- `karak flow complete FILE [-o OUT]` fills missing parameters of an older
  or hand-written flow from the stage templates, and prints what it added.
  It also upgrades flow format version 1 (sparse parameters) to version 2.
- `karak run ... --set NODE.PARAM=VALUE` changes one value for one run. The
  run record keeps the complete flow after the change (see
  [Reproducibility](#reproducibility)).

Conventions:

- `null` is a value only for parameters whose template value is `null`
  (for example `denoise.sigma_color`, where `null` means "derive from the
  data range", or `mask.valid_mask_path`, where it means "no polygon").
  Anywhere else `null` is an error, never a stand-in for a default.
- Some integer parameters use `0` for "automatic", for example
  `hdbscan_*.min_samples = 0` (same as `min_cluster_size`) and
  `hdbscan_*.subsample_n = 0` (use all pixels). The stage reference says so
  in each parameter's help text.
- `load_elements.colormap` selects the inversion palette: `tima:jet` (the
  256-entry palette TIMA renders with; level k inverts to exactly k/255),
  `cmap:NAME` (a matplotlib colormap), or `lut:PATH` (an (N, 3) uint8 `.npy`
  palette; relative paths resolve against `input_dir`). The first run with
  a palette builds a 256³ RGB-to-scalar lookup table (about 10 s for
  `tima:jet`, 30–60 s for a 4096-entry palette), cached on disk and reused.
- `load_elements.header_trim_px` and the other trims remove edge strips,
  in original-image pixels. TIMA element-map exports have no header band,
  so the builtins use `0`.
- `mask.valid_mask_path` is a napari shapes CSV with polygon coordinates in
  original image space; it aligns with any downsample factor and trim.

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
└── attrs:        pipeline_config (the complete flow, as YAML), created (UTC),
                  pipeline_version, python_version, platform,
                  library_versions (JSON)
```

The `export_h5` sink writes the file; the executing flow JSON is embedded
in its root attributes. Matching `load_*` functions in `karak.io.storage`
read every group back for downstream analysis.

## Caching and re-runs

Every stage output is cached under `{work}/cache/` (default: `work/` next
to the output), keyed by a recipe hash of the stage type, its parameters,
its upstream results, and — for `load_elements` — the input files' sizes
and modification times. Re-running a flow reuses everything that did not
change:

```bash
karak run --builtin global --input data/ --out output/s1   # first run: all stages
karak run --builtin global --input data/ --out output/s1   # warm: only sinks re-run
karak run --builtin global --set refine.target_phase=3 ... # only refine + downstream
karak run ... --no-cache                                   # force a clean run
```

An interrupted run resumes the same way: completed stage outputs are
already in the cache, so the next invocation continues from the crash
point. To start from scratch, delete the work directory
(`<out dir>/work/`).

## CLI reference

```
karak flow init --builtin NAME -o FLOW.json [--force]
karak flow complete FLOW.json [-o OUT]
karak run (FLOW.json | --builtin global|tiled|tiled-rare|stepwise)
          --input DIR --out BASE [--work DIR] [--no-cache] [--no-qc]
          [--set NODE.PARAM=VALUE ...] [--device cpu|cuda] [--workers N] [--plain]
karak validate (FLOW.json | --builtin NAME)
karak schema
karak bench ...
karak view BASE
```

`karak flow init` writes a builtin flow with every parameter listed;
`karak flow complete` fills missing parameters from the stage templates
(see [Parameters](#parameters)).

On a terminal, `karak run` shows a live dashboard: the run context
(input, output, workers, device, cache, memory), a table of steps with
their status and time, the running step's parameters with changed values
first, a progress bar where the stage reports progress (the load step
reports one tick per file), output summaries such as
`ElementCube 6525×3990×19 float32 1.98 GB space=raw`, and the last five
log lines. A cached step shows the first 8 characters of its recipe hash.
When the run ends, the last frame stays on screen as the summary.

With `--plain`, or when stdout is not a terminal (a pipe, `nohup`, a CI
log), the same information prints as plain lines.

`karak run` exits with 1 when a step fails and with 130 on Ctrl-C;
completed steps stay cached.

To inspect the load step's output, open it in napari (install the extra
once with `uv sync --extra view`):

```
uv run --extra view karak view BASE [--show Fe-K,Si] [--mask mask/Valid_mask.csv]
```

`BASE` is the run's `--out` value, its work or cache directory, or a single
cached `.h5` file copied from another machine. Given an `--out` value, the
command opens the outputs listed in the latest run record
(`{out}/runs/latest/run.json`): the load step's element cube and the BSE
image from the same step. Without a record (or if its cache files are
gone) it opens the newest cached element cube instead. It shows one gray
layer per element (only `--show` elements visible), placed in full-resolution
coordinates, so the cursor position matches the original exports and a
napari shapes CSV such as the valid-area mask lines up. It loads the whole
cube into memory (about 2 GB for NWA 4587).

## Reproducibility

- **Seeded randomness** — every stochastic step (PCA subsampling and solver,
  HDBSCAN fit subsampling, GMM fitting) is seeded through a `random_state`
  parameter (42 in the builtins). Two runs with the same flow, data, and
  library versions produce identical outputs.
- **Complete flows** — a flow file lists every parameter of every node, so
  the file alone specifies a run; no value comes from code defaults.
- **Run records** — every `karak run` writes `{out}/runs/<UTC time>/`
  (`{out}` is the `--out` value) with `flow.json`, the complete flow exactly
  as executed after any `--set`/`--device` changes (rerun it with
  `karak run {out}/runs/<time>/flow.json --input ... --out ...`), and
  `run.json`: status (ok, failed, or interrupted, with the error), start and
  finish times, the command line, input/output/work paths, workers and
  cache settings, the `--set` overrides, the karak version and git commit
  (with a dirty flag), library versions, host, and per node the resolved
  parameters, recipe hash, whether it ran or came from the cache, its time,
  and its output files with one-line summaries. `{out}/runs/latest` points
  at the newest record. Records are never overwritten or pruned.
- **Embedded provenance** — the output HDF5 file records the complete
  executing flow, library versions, Python version, and
  platform string, making every result file self-documenting.
- **Locked dependencies** — the repository ships a `uv.lock` file;
  `uv sync` reproduces the exact tested environment.

## Computational requirements

Tested on an AMD Ryzen AI 5 340 with 32 GB RAM (Linux).

- A full-resolution run on a ~7400 x 5400 px, 21-channel dataset uses
  **~10–12 GB RAM** (raw float32 cube ~3.2 GB plus working copies).
- For large images, bound memory with `hdb.subsample_n` (e.g. 500000) or
  the `tiled` flow, which keeps per-tile memory
  constant regardless of image size.
- A run with `--set src.downsample_factor=4 --set
  src.include_elements=Fe-K,Ca,Mg,Si` checks an installation in minutes on
  a laptop.
- No GPU is required; all computation is CPU-based.
