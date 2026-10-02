# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Cache writes run on a background thread: the next step starts while the
  previous outputs are written; the run drains the writer before it
  returns, also on failure and Ctrl-C. Log lines report each write.
- Cache files default to the HDF5 `lzf` filter (NWA 4587 denoise output:
  written in 8.8 s and 972 MB, against about 35 s and 842 MB for the
  previous gzip file); `karak run --cache-compression {lzf,gzip,none}`.
  Datasets are chunked (512, 512). Existing gzip cache files stay
  readable.
- A total RAM budget for outputs held between steps (`--ram-budget GB`,
  default half of the available memory) replaces the fixed 256 MB
  per-payload spill threshold; spills are logged. Writes waiting in the
  cache writer's queue are bounded by the same amount: a step that
  produces faster than the disk writes waits. The writer drops each
  payload as soon as it is written.
- A failed cache write (a full disk) stops the run at the next step with
  an `error: cache writer: ...` line; runs delete `.tmp` files left in the
  cache by killed runs.
- `karak flow init --builtin NAME -o FILE` writes a builtin flow with every
  parameter listed; `karak flow complete FILE [-o OUT]` fills missing
  parameters from the stage templates, prints what it added, and upgrades
  format version 1 flows.
- `Stage.template()` returns a stage's template parameter values.
- `docs/flow.schema.json`: a JSON Schema for flow files, generated from the
  stage registry (`uv run python -m karak.flow.schema`; a test pins it).
  One node variant per stage, every parameter required, with types,
  choices, bounds, nullability, and template values as `default`
  annotations. Groundwork for a visual pipeline composer.
- Run records: every `karak run` writes `{out}/runs/<UTC time>/flow.json`
  (the complete flow as executed, after overrides) and `run.json` (status,
  times, argv, paths, settings, overrides, karak version and git commit,
  library versions, host, and per node the resolved parameters, recipe
  hash, cache status, time, and output files). Failed and interrupted runs
  are recorded too; `{out}/runs/latest` points at the newest. The dashboard
  and `--plain` output show the record path.

- `karak run` shows a live dashboard on a terminal: run context, the
  running step's parameters (changed ones first), per-file progress for
  the load step, output summaries, cache status with recipe hashes,
  memory use, and the latest log lines. The final frame stays on screen
  as the run summary. `--plain` (automatic when stdout is not a terminal)
  prints the same information as lines.
- `stepwise` builtin flow, which grows one step at a time; it currently
  runs the load, mask, and denoise steps.
- Bilateral filter core (`karak.preprocessing.bilateral`): a numpy
  reference and a CuPy kernel for the bilateral and the joint (guided)
  bilateral filter. `--device cuda` on the denoise step now runs this
  kernel for every bilateral method (cuCIM is no longer used there).
- Denoise `method` choices `bilateral_sym` (symmetric Gaussian kernel),
  `joint_bilateral_total` (range weight from the summed channels) and
  `joint_bilateral_bse` (range weight from the BSE image, wired to the
  stage's new optional `bse` input port). `bilateral` is unchanged and
  stays scikit-image on the CPU. For the new methods the colour lookup
  table covers the differences to the zero padding, so border pixels are
  weighted correctly, and a constant guide (a flat BSE image or a constant
  channel sum) still applies the spatial kernel.
- Finding: scikit-image's `denoise_bilateral` (0.19 through 0.26) applies
  an off-centre spatial kernel. `_compute_spatial_lut` builds an
  (n+1) x (n+1) table for an n x n window (`np.arange(-n // 2, ...)`) and
  the Cython loop reads it with stride n, so for the default 7 x 7 window
  the unit weight sits at offset (+2, -2) and the top row has zero weight.
  karak keeps that behaviour under `bilateral` because the published
  baseline used it; the CUDA `bilateral` reproduces it to 1e-5.
- The denoise step reports progress per element channel to the dashboard,
  also with `--workers` (channels complete in any order) and on CUDA;
  `denoise_cube` and the per-method functions take an `on_channel` callback.
- Payloads have `summary()`; the cache stores it next to each payload.
- `karak view PATH` opens the newest cached element cube and its BSE image
  in napari (optional `karak[view]` extra): one gray layer per element,
  placed in full-resolution coordinates, plus an optional napari shapes
  CSV (`--mask`) such as the valid-area mask. `PATH` is a run's `--out`
  base, its work or cache directory, or a single cached `.h5` file. With
  an `--out` base it opens the outputs of the latest run record.
- `karak view` also opens the mask step's output: the mineral mask as a
  labels layer, and the valid mask as a hidden labels layer when the flow
  set one. Cached payloads record the recipes of the outputs they consumed
  (an `upstream` attribute), so a cache scan overlays only masks computed
  from the cube it opens.
- `karak view` also opens the denoise step's cube as `dn: <element>`
  layers next to the raw elements, visible for the same `--show` elements.
  Cubes are told apart by their space tag, so a cache scan never mistakes
  a denoised cube for a raw one, also for files written before upstream
  recipes were recorded. A cache scan shows a denoised cube with the masks
  it consumed, never a newer mask from an interrupted rerun.
- Exit codes for `karak run`: 1 when a step fails, 130 on Ctrl-C.
- Warnings raised while loading maps (for example PIL's
  `DecompressionBombWarning` on 104 Mpx exports) are logged once through
  the `karak` logger, in serial and `--workers` runs alike, so they show
  in the dashboard log panel instead of breaking the live display.

- `colormap: tima:jet`, the 256-entry palette TIMA renders element maps
  with, recovered from the NWA 4587 exports (all 256 entries occur there).
  Level k inverts to exactly k/255.

### Changed

- **Flows are complete (format version 2).** A flow JSON lists every
  parameter of every node, and a run takes no value from code: `karak
  validate` and `karak run` reject a flow with a missing parameter or a
  version-1 (sparse) flow, and name the fix (`karak flow complete`). The
  shipped flows list every parameter and are the source of truth; the code
  that built them is gone. Recipe hashes are unchanged, so existing caches
  stay valid.
- JSON `null` is a value only for parameters whose template value is
  `null`; elsewhere it is an error instead of meaning "use the default".
- `--set` of a parameter the stage does not declare is an error.
- Node `ui` metadata (canvas positions) no longer affects flow equality.
- `rare_phase.merge_threshold` is an explicit value (template 0.92, range
  0.5 to 1). Its old `0` meant "reuse the tiled threshold" but actually took
  0.92 from code, whatever the tiled node said; a flow that still says 0
  now fails validation. In `tiled-rare` the recipe hashes of `rare` and its
  downstream nodes change once; results are the same.
- The core argument bundles moved from `karak.config` (Pydantic, with
  default values) to `karak.core_params` (frozen dataclasses with no
  defaults), built by the stages from their params (`downsample_config`,
  `loader_config`, `denoise_config`, `pca_config`, `hdbscan_config`,
  `tiled_config`, `rare_phase_config`, `refinement_config`). Core functions
  no longer default their data arguments either (for example
  `create_mineral_mask(min_object_size=...)`, `assign_noise_pixels(k=...)`,
  `load_element_maps(bse_channel=..., include_elements=...,
  loader_config=...)`). `run_tiled_hdbscan` and `recluster_unassigned` take
  explicit bundles in place of `ClusterConfig`.
- The export sink records each group's parameters from the complete
  executing flow, and fails when the producing node is missing, instead of
  filling values from code.

- The default colormap is now `tima:jet` instead of `cmap:jet`.
  Matplotlib's jet cuts the cyan and yellow corners of TIMA's ramp and
  places its segment breakpoints differently, so on TIMA exports it merged
  22 of 255 levels and shifted intensities by 1.2 levels on average (max 8
  levels, 0.032).
- The default `header_trim_px` is now 0 instead of 100. TIMA element-map
  exports carry no header band; the old trim removed 100 rows of sample.
  Valid-mask CSVs are in full-resolution coordinates and still align.
- To reproduce results from 0.2.0 and earlier exactly, set
  `colormap: cmap:jet` and `header_trim_px: 100`
  (`--set src.colormap=cmap:jet --set src.header_trim_px=100`).

### Fixed

- The GPU bilateral path never ran: cucim (25.6 through 26.8) ships no
  `denoise_bilateral`. `--device cuda` on the denoise step now uses
  karak's own CuPy kernel (see Added).
- `load_valid_mask` failed with a string path (every flow run with
  `mask.valid_mask_path` set) after the shapes-CSV refactor.
- The mask step no longer triggers scikit-image's `min_size` deprecation
  warning; `max_size = min_object_size - 1` keeps the same objects.

### Removed

- The legacy YAML mode: `karak -c config.yaml`, `--test-mode`,
  `--emit-flow`, `--clean`, `--from-stage`, `load_config`, `save_config`,
  `PipelineConfig`, and the YAML-to-flow shim.
- `rare_phase.noise_reassign_k`, which had no effect.
- The `pydantic` dependency. Write the pipeline as a flow
  JSON instead (`karak flow init`). `karak` with no command prints usage.

## [0.2.0] - 2026-08-22

Modular re-architecture into three layers (numeric core, stages, flow),
emulating the chainable-modules + JSON-flow-graph style. The science is
unchanged; parity tests pin every stage to the core function it wraps.

### Added

- `karak.stages`: one self-describing class per processing step with typed,
  bounded `Param`s and named `Port`s carrying immutable payloads
  (`ElementCube`, `MaskSet`, `PCAFeatures`, `Labels`, ...). Auto-registry;
  `karak schema` prints the full palette as JSON.
- `karak.flow`: JSON flow graphs (nodes + edges), a standalone structural
  validator, and a headless executor with content-addressed per-node
  caching, refcounted memory eviction, and spill-to-cache for large arrays.
- Builtin flows `global`, `tiled`, and `tiled-rare`, shipped both as code
  and as JSON under `karak/flow/flows/`.
- New CLI: `karak run (FLOW.json | --builtin NAME) --input DIR --out BASE
  [--set NODE.PARAM=VALUE ...]`, `karak validate`, `karak schema`, mirrored
  by `python -m karak.flow`.
- `karak --emit-flow -c config.yaml` prints the flow equivalent of a legacy
  YAML config for migration.
- The cluster stage split into `pca`, `hdbscan_global`/`hdbscan_tiled`,
  `rare_phase`, `noise_assign`, `refine`, `cluster_stats`, and
  `fingerprints` (chemical fingerprinting is now part of the standard
  flows).
- QC figures are now sink stages (`qc_*`), including `qc_named_phase_map`
  for researcher-assigned mineral names.
- Test suite grew from 4 to 20 files: Param coercion, registry, payload
  HDF5 round-trips, per-stage parity, graph JSON round-trip, one test per
  validation rule, executor caching/invalidation/eviction, and end-to-end
  flow runs on a synthetic scene.

### Changed

- The provenance HDF5 is now a product written by the `export_h5` sink; the
  executing flow JSON is embedded in its root attributes. The group layout
  is unchanged.
- `karak -c config.yaml` converts the YAML to a flow via
  `flow_from_config()` and runs on the flow engine. Results are identical.
- `io.storage` save functions take plain dicts instead of config objects.
- The PCA component heuristic (95% cumulative variance, minimum 5) moved
  from the runner into `clustering.pca.auto_n_components()`.

### Deprecated

- `--from-stage`: per-node caching resumes automatically from whatever
  changed. The flag prints a warning and is otherwise ignored.

### Removed

- The `_run_*` stage functions, shared `ctx` dict, and HDF5
  `stage_completed` checkpoint attributes
  (`mark_stage_complete`/`get_completed_stages`).

## [0.1.0] - 2026-08-05

Initial release accompanying the manuscript *"Unsupervised mineral phase
mapping from SEM-EDS element maps: a density-based clustering pipeline with
multi-resolution refinement"* (submitted to Computers & Geosciences, 2026).

### Added

- Five-stage checkpointed pipeline (`load -> mask -> denoise -> normalize ->
  cluster`) driven by the `karak` CLI with Rich TUI, HDF5 checkpoints, and
  `--from-stage` resume support.
- Colormap inversion of false-color SEM-EDS element maps (jet or any
  matplotlib colormap, or custom `.npy` LUT) to scalar [0, 1] intensities
  via a cached 256^3 RGB-to-scalar lookup table.
- Background/epoxy masking from a napari valid-region polygon CSV combined
  with all-channel-zero detection.
- Edge-aware denoising (bilateral filter or anisotropic diffusion) on raw
  intensities, followed by per-channel z-score normalization over mineral
  pixels.
- PCA dimensionality reduction and HDBSCAN density-based phase discovery
  with two strategies: `global` (with optional subsample +
  `approximate_predict`) and `tiled` (per-tile HDBSCAN with cosine-similarity
  phase-registry merging).
- Optional two-pass rare-phase reclustering and post-clustering refinement
  (threshold-based olivine extraction, GMM sub-phase splitting).
- kNN distance-weighted reassignment of HDBSCAN noise pixels.
- Per-cluster chemical fingerprinting with cosine-similarity flagging of
  potentially over-split clusters, plus human-in-the-loop mineral naming.
- QC figure generation at every stage (mask overlays, denoise comparisons,
  scree plots, phase maps, fingerprint charts).
- Full provenance: pipeline config YAML, library versions, platform info,
  and per-stage timestamps embedded in every HDF5 output.
- Pydantic-validated YAML configuration covering every pipeline parameter,
  with seeded randomness (`random_state`) for reproducible runs.
- Fast pytest suite (config round-trip, colormap inversion, mask utilities,
  synthetic end-to-end smoke test) and GitHub Actions CI.
