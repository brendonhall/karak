# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `karak run` shows a live dashboard on a terminal: run context, the
  running step's parameters (changed ones first), per-file progress for
  the load step, output summaries, cache status with recipe hashes,
  memory use, and the latest log lines. The final frame stays on screen
  as the run summary. `--plain` (automatic when stdout is not a terminal)
  prints the same information as lines.
- `stepwise` builtin flow, which grows one step at a time; it currently
  runs the load step only.
- Payloads have `summary()`; the cache stores it next to each payload.
- Exit codes for `karak run`: 1 when a step fails, 130 on Ctrl-C.
- Warnings raised while loading maps (for example PIL's
  `DecompressionBombWarning` on 104 Mpx exports) are logged once through
  the `karak` logger, in serial and `--workers` runs alike, so they show
  in the dashboard log panel instead of breaking the live display.

- `colormap: tima:jet`, the 256-entry palette TIMA renders element maps
  with, recovered from the NWA 4587 exports (all 256 entries occur there).
  Level k inverts to exactly k/255.

### Changed

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
