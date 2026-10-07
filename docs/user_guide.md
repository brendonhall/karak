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
| `denoise` | `cube:raw`, `masks`, optional `bse` → `cube:denoised` | Edge-aware smoothing of the raw [0, 1] cube, channel by channel — bilateral filter (default; scikit-image's, including its off-centre spatial kernel, for the published baseline), `bilateral_sym` (symmetric kernel), `joint_bilateral_total` / `joint_bilateral_bse` (range weight from the summed channels / the BSE image on the `bse` port), or anisotropic (Perona-Malik) diffusion. Operating on raw intensities preserves grain boundaries and physical signal. On `--device cuda` the bilateral methods run karak's CuPy kernel. |
| `normalize` | `cube:denoised`, `masks` → `cube:normalized` | Per-channel z-score normalization: mean and standard deviation are computed over **mineral pixels only**, then `z = (x - mean) / std`. Non-mineral pixels are set to 0. On `--device cuda` runs on the GPU. `accumulate` sets the precision of the mean and standard-deviation sums: `float64` (default) is accurate; `float32` reproduces the published NWA 4587 baseline, whose standard deviations are up to 3.6 % off because numpy summed 12.5 M rows in float32. |
| `pca` | `cube:normalized`, `masks` → `features` | PCA fit + projection of mineral pixels. `n_components: 0` auto-selects the first count reaching 95% cumulative variance (minimum 5). On `--device cuda` runs on the GPU; features match the CPU path within 1e-3. |
| `hdbscan_global` | `features` → `labels:raw` | Single HDBSCAN run over all mineral-pixel features. |
| `hdbscan_tiled` | `features`, `cube:denoised` → `labels:raw`, `tiles` | Per-tile HDBSCAN with cosine-similarity phase-registry merging across tiles. `accumulate` sets the precision of the tile-fingerprint sums: `float64` (default) or `float32` (the published baseline). A tile with fewer than `min_clusters_per_tile` clusters is deferred: its pixels stay unassigned for `noise_assign`, and `rare_phase` skips them. |
| `rare_phase` | `labels:raw`, `features`, `cube:denoised`, `tiles` → `labels:raw`, `tiles` | Recluster still-unassigned pixels with more sensitive parameters (Pass 2 of the two-pass workflow). Including this stage in a flow is what enables the workflow. `accumulate` sets the precision of the rare-cluster fingerprint sums, as for `hdbscan_tiled`. On `--device cuda` the pass-2 HDBSCAN runs with cuML (NWA 4587: 92 s instead of 684 s on 16 CPU threads, the same 11 phases); the registry merge stays on the host. Pixels of deferred tiles are skipped (they would otherwise form one block of the pass-2 majority phase). |
| `noise_assign` | `labels:raw`, `features` → `labels:cleaned` | Distance-weighted k-NN reassignment of every remaining unlabeled pixel. On `--device cuda` a CuPy brute-force search with the same vote runs on the GPU (NWA 4587: 24 s instead of 281 s, identical labels); labels can differ from the cpu only where two neighbor distances tie within float32 precision. |
| `name_phases` | `labels:cleaned` → `labels:cleaned` | Attaches mineral names (`names`: `'0: Ilmenite; 1: Silica'`) that travel with the labels to the fingerprints, the QC figures and the export (`clusters/mineral_names`). Fails at run time if a named label is not in the data. |
| `split_threshold` | `labels:cleaned`, `cube:denoised` → `labels:cleaned` | Moves the pixels of one phase that satisfy a rule on denoised channels (`'Fe-K > 0.6 & Ca < 0.10'`) to a new, named label. Appends a record to the split history. |
| `split_gmm` | `labels:cleaned`, `cube:denoised` (+`bse`) → `labels:cleaned` | Gaussian mixture on z-scored features (channels, `BSE`, ratios `A/(A+B)`). `keep_parent` lets the largest component keep the parent label; `order_by` orders the new labels by a feature's component mean. |
| `split_hires` | `labels:cleaned`, `cube:raw` (higher resolution) → `labels:cleaned`, `labels_hires` | One GMM on a channel or ratio of a full-resolution cube inside a set of phases; every full-resolution pixel is classified (`labels_hires`, exported to `clusters/hires/labels`) and the working labels take the majority of their children. The second cube comes from a `load_elements` node with `downsample_factor: 1` and `include_elements`. |
| `cluster_stats` | `labels:cleaned` → `stats` | Cluster counts, sizes, and noise fraction. |
| `fingerprints` | `labels:cleaned`, `cube:denoised` → `fingerprints` | Per-cluster mean/std element intensities from the denoised cube, with cosine-similar pairs flagged. `accumulate` sets the precision of the mean and standard-deviation sums: `float64` (default) is accurate; `float32` reproduces the published baseline, which drifts by a few percent on clusters of millions of pixels. |
| `export_h5` | eleven optional inputs → sink | Writes the provenance HDF5 file; only connected groups are written (see [HDF5 output layout](#hdf5-output-layout)). |
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
and QC figures, then record the names in the `names` param of a
`name_phases` node. The names travel with the labels: `fingerprints`,
`qc_fingerprints`, `qc_named_phase_map` and `export_h5` use them. You can
also record names after the run with
`karak.io.storage.save_mineral_names()`. `preprocessing.denoise.compare_denoisers()` is
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
by `edges` (output port to input port). Five builtins ship with karak:

| Flow | Clustering path |
|------|-----------------|
| `global` | `hdbscan_global → noise_assign` |
| `tiled` | `hdbscan_tiled → noise_assign` |
| `tiled-rare` | `hdbscan_tiled → rare_phase → noise_assign` |
| `stepwise` | `hdbscan_global → noise_assign`, then statistics and fingerprints; `global` without the export and QC figures |
| `paper` | `tiled-rare` + the published names and splits; 1x pyroxene map |

`stepwise` is the dashboard flow: every processing step of `global`, with
the same parameters and wiring, but without the HDF5 export and the QC
figures, so a run writes only the cache, which `karak view` reads. A test
keeps the two flows in step. Because the recipes are the same, `karak run
--builtin global` with the same `--input`, `--out`, `--device` and `--set`
values afterwards takes every processing step from the cache and runs only
the export and the figures (on NWA 4587: 9 steps cached, 8 run, 122 s):

```bash
karak run --builtin global --input DIR --out BASE --device cuda \
    --set 'msk.valid_mask_path={input}/mask/Valid_mask.csv' --set hdb.subsample_n=50000
```

`stepwise` runs the load step (`src`), the mask step (`msk`), the denoise
step (`dn`, bilateral with the `global` values), the normalize step
(`nrm`, z-scores with float64 sums), the PCA step (`pca`, components
kept up to 95% of the variance, at least 5), the HDBSCAN step (`hdb`,
`hdbscan_global` with the `global` values), the noise-reassignment step
(`knn`, 5 neighbours; about 24 s on a GPU at full NWA 4587 scale), and
then the cluster statistics (`stats`) and the chemical fingerprints (`fp`,
from the denoised cube) of the reassigned phases. With the `global` values
(no `subsample_n`) HDBSCAN fits every mineral pixel. At full NWA 4587
scale that needs about 400 GB on the CPU, so the step stops at once with
an error that names `subsample_n`, and on a 24 GB GPU cuML runs out of
memory; see
[HDBSCAN settings for full-scale runs](#hdbscan-settings-for-full-scale-runs)
for the values to set instead. The
builtin leaves `msk.valid_mask_path` at `null` (no polygon); set it in your
own copy (`karak flow init --builtin stepwise -o FILE`), for example to
`"{input}/mask/Valid_mask.csv"`.

`paper` is `tiled-rare` with the settings of the published NWA 4587 run,
plus the six steps that were done by hand after the March 2026
clustering, each as a node with its parameters and a `note`:

| node | type | what it decides |
|---|---|---|
| `src_hires` | `load_elements` | Ca and Mg at 1x (no downsample), the same 100 px trim and jet palette |
| `names` | `name_phases` | the 11 base names, from the fingerprints and the TIMA reference |
| `oliv` | `split_threshold` | olivine out of phase 2: `Fe-K > 0.6 & Ca < 0.10` |
| `weath` | `split_gmm` | the smaller of two components of phase 2 on Ca/(Ca+Mg), BSE and Fe-K (the three features of the published split) is a weathering assemblage |
| `pyx` | `split_hires` | pigeonite and augite from Ca/(Ca+Mg) at 1x inside phase 2; the 1x map goes to `clusters/hires/labels` |
| `phos` | `split_gmm` | merrillite and chlorapatite from Cl, Na, Mg, F of phase 7, ordered by Cl |

The clustering settings (stored in the published HDF5 as
`clusters.attrs["cluster_config"]`) are the matplotlib jet palette
(`cmap:jet`) and a 100 px header trim for the load, the valid mask at
`{input}/mask/Valid_mask.csv`, `min_cluster_size` 100, `min_samples` 25,
1024 px tiles and merge threshold 0.88 for the tiled HDBSCAN, merge
threshold 0.88 for the rare phases, and float32 sums in the normalize,
tile, rare-phase and fingerprint steps. Tests pin the nodes, the wiring
and these settings.

What a CPU run reproduces of the published result:

- load, mask, denoise and normalize outputs are bit-identical;
- the PCA explained variance agrees within 5e-7; the features differ at
  about 1e-5 relative because the OpenBLAS kernel of the CPU that ran the
  paper differs, which moves pass-1 noise by up to 4,111 pixels per tile;
- the tiled pass finds the published 22 tiles with the same cluster count
  in each; its 11 phases each match one published phase at 100 % purity;
- the rare-phase pass matches the published raw labels once the one
  deferred 1024 px tile is left to kNN (the published file had it reset
  by hand): replayed on the published pass-1 noise, 6 of 12,495,787
  pixels differ;
- the splits give the published phases: 15 with pixels, where the published
  file keeps a 16th label for 92 pixels of unresolved pyroxene that the
  flow splits away. The numbering differs in one place (13 pigeonite, 14
  augite here; the published file has them the other way round).
  Abundances against Table 1 of the paper: within 0.51 pp for every phase
  (largest: Weathering Assemblage, -0.50 pp; then Epoxy, +0.28 pp and
  Plagioclase, +0.16 pp). The 1x pyroxene map gives 272 grains against the
  published 128, a median lamella spacing of 45 um against 51 um, and a
  Rayleigh p of 0.80 against 0.008. The published count rests on a stale
  setting: the paper's lamellae script read a downsample factor of 4 against
  a factor-2 file, so its pyroxene mask was the top-left quarter of the
  section. With the correct factor the same script gives 272 grains on the
  published labels, and karak's 1x map agrees with it at 99.25 %.

Run it on the CPU: cuML selects different clusters on these full tiles
(see [Computational requirements](#computational-requirements)). The tiled
HDBSCAN takes about 30 minutes on 16 threads, with the pool limited to 7
workers by memory; the 1x load adds about a minute and 0.8 GB.

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

### HDBSCAN settings for full-scale runs

The builtins keep the `global` HDBSCAN values (`min_cluster_size` 1000,
`min_samples` 1000, `subsample_n` 0), so their recipe hashes and the
published baseline stay the same. At full scale, set these instead:

```bash
--set hdb.subsample_n=50000 --set hdb.min_cluster_size=125 --set hdb.min_samples=125
```

With `subsample_n`, HDBSCAN fits a random subsample and then assigns every
pixel to the clusters it found. A phase that covers a share p of the
mineral pixels has about p × `subsample_n` points in the subsample, and it
can only become a cluster when that number is well above
`min_cluster_size`. With 50000 and 1000, a phase needs more than about 2 %
of the pixels, so small phases disappear into their neighbours. With 50000
and 125, the limit is about 0.25 %.

Measured on NWA 4587 (`flows/nwa4587_stepwise.json`, 12,495,787 mineral
pixels at `downsample_factor` 2; RTX 4090 and 16 CPU threads with
`--workers 0`):

| `subsample_n` | `min_cluster_size` = `min_samples` | phases | `hdb` time |
|---|---|---|---|
| 50000 | 1000 | 5: a 0.4 % Ca-P-Y phase and a 2.1 % Fe-O phase merge into larger ones | 65 s (GPU) |
| 50000 | 125 | 7 | 15 s (GPU), 267 s (CPU) |
| 100000 | 100 | the same 7 | 27 s (GPU) |
| 50000 | 50 | 8: adds a 0.28 % silica-like phase (Si 0.94, O 0.31) | 13 s (GPU) |

The 7 phases at 50000 / 125 match the 7 phases of a fit on every pixel at
`downsample_factor` 8 (each mean element spectrum has a cosine similarity
of at least 0.999 with its match), and the CPU and the GPU give the same
HDBSCAN labels (adjusted Rand index 1.000). At `downsample_factor` 8, three
random seeds at 50000 / 1000 give the same phases. Smaller values find smaller phases but also more noise
pixels (24 % at 125 against 19 % at 1000; the noise-reassignment step gives
each of them a phase). Check any new small phase with the fingerprints
that `karak view` prints before you treat it as a mineral.

On the CPU, the HDBSCAN fit holds about 32 bytes per fitted pixel and
`min_samples` neighbour (24 GB for 782 k pixels at 1000). Before it starts,
the step compares that estimate with the available memory and stops with
an error that names `subsample_n` when the fit would need more than 80 %
of it. With the `global` values at full NWA 4587 scale the estimate is
400 GB, so the step stops at once instead of swapping. With `--workers N`
the tiled flows fit N tiles at once, so they run only as many workers as
the largest tile fits need together (a warning names the number); on NWA
4587 with the `global` values (8.4 GB per full 512 px tile) that is 2 of
16 workers, with the full-scale settings all 16.

#### Which flow and settings for which goal

Measured on NWA 4587 at full scale on the CPU (16 threads), against the
named phases of the published run:

| goal | flow and settings | phases | `hdb` time |
|---|---|---|---|
| main phases, fast | `global` (or `stepwise`) with `subsample_n` 50000, `min_cluster_size` = `min_samples` = 125 | 7: ilmenite, spinel, plagioclase, epoxy, the pyroxene group, merrillite/chlorapatite, ferroan olivine | 267 s (15 s on a GPU) |
| every small phase | `--builtin paper` (tiled, `min_cluster_size` 100, `min_samples` 25, 1024 px tiles, no subsample) | 11 after `hdb` (the later nodes of `paper` split these to 15 with pixels): adds silica (0.10 %), calcite (0.15 %), Fe oxyhydroxide (0.03 %), a Zn phase (0.01 %) and xenotime (0.004 %), each 100 % pure; olivine stays in the pyroxene group | 1,775 s; CPU only |
| in between | `tiled` or `tiled-rare` with 50000 / 125 | 9: silica and calcite mixed with weathering material; Fe oxyhydroxide, the Zn phase and xenotime lost | 365 s |

A per-tile subsample speeds up the `paper` settings but does not keep their
small-phase recovery: with `subsample_n` 200000 the step takes 672 s
instead of 1,775 s but loses or mixes all five small phases, and with
500000 it takes 1,590 s and keeps only xenotime. Only the
subsampled fits separate the ferroan olivine (about 2.5 %) from the
pyroxene group; the `paper` flow separates it afterwards in its `oliv`
node, with the threshold of the published run.

---

## HDF5 output layout

All results are written to a single HDF5 file (`hdf5_output`). The
`export_h5` param `compression` picks the filter for the large datasets:
`gzip` (the default; level 4, readable by any HDF5 tool), `lzf` (faster,
readable by h5py and PyTables only) or `none`. Compressed datasets use
chunks of 256 × 256 pixels with every element in one chunk, and the
denoised and normalized cubes also use HDF5's shuffle filter, which makes
continuous floats compress smaller and faster (it makes the raw maps,
which hold few distinct values, larger, so they do without it). On NWA
4587 at full scale the export takes 67 s with gzip (2.06 GB), 28 s with
lzf (2.51 GB) and 6 s without compression (6.34 GB); the data are the same
in each.

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
│   │             attrs: mineral_names (JSON), cluster_N_name (both written
│   │             when the flow names the phases)
│   ├── subclustering/  split history (written when the flow splits phases);
│   │             attrs: history (JSON list), split_NN (JSON record each)
│   ├── hires/    labels (H1, W1) int16, full-resolution sub-phase map
│   │             (written when the flow has split_hires); attrs: ratio,
│   │             names (JSON), downsample_factor, header_trim_px,
│   │             left_trim_px
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
karak run ... --no-cache                                   # force a clean run
```

A change to one node reruns that node and what follows it. In the `paper`
flow, this reruns only `weath` and its downstream nodes (after a prior
`paper` run to the same `--out`):

```bash
karak run --builtin paper --input data/ --out output/s1 --set weath.random_state=7
```

Cache files are written on a background thread, so the next step starts
while the previous step's outputs are still being written. A run waits for
the writer before it returns, also after a failure or Ctrl-C, so completed
outputs always land. Files use the HDF5 `lzf` filter by default
(`--cache-compression gzip` for smaller files, `none` for the fastest
writes and reads). Readers accept any of the three. Each finished write
appears in the log lines as `cache: dn.cube written in 10.8 s`.

Between steps, outputs stay in RAM up to a budget, half of the available
memory by default (`--ram-budget GB` to change it). An output that would
exceed the budget is spilled: karak does not hold it, and each consumer
reads it back from the cache. The log says so (`store: dn.cube (1.98 GB)
spilled to cache, budget 11.0 GB`). Outputs waiting for their cache write
have a second bound of the same size: when the queued writes would exceed
it, the next step waits until the disk catches up. Host memory for stage
outputs therefore stays below twice the budget (the default is half of the
available memory). With `--no-cache` nothing is written or spilled.

With `--device cuda`, a step's outputs stay on the GPU for the next GPU
step. The executor moves each input to the consuming step's device (its
`device` parameter, or the CPU when it has none), so a CPU step always
sees host arrays. In a chain such as denoise, normalize, pca, hdbscan,
the cube moves to the device once and each step passes device outputs to
the next. Device outputs are held up to a
budget, 80 % of the free device memory at start (`--gpu-budget GB` to
change it). Beyond it an output moves to host RAM (logged as `store:
dn.cube (1.98 GB) moved to host, gpu budget 19.6 GB`), the freed device
memory goes back to the driver before the next step, and the RAM rules
apply. The budget governs outputs held between steps only: if moving a
step's inputs to the GPU, or the step itself, runs out of device memory,
the step fails with an `error:` line that names it. The cache always
receives a host copy. The run record notes where
the run held each output (`"device": "cuda"`, or `"cpu"` after a move to
host RAM), and the dashboard shows the
device memory next to the host figure.

If a cache write fails (a full disk), the run stops before the next step
and prints `error: cache writer: ...`. A run also deletes the partial
`.tmp` files that killed runs left in the cache directory.

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
          [--cache-compression lzf|gzip|none] [--ram-budget GB] [--gpu-budget GB]
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
reports one tick per file, the denoise step one per element), output
summaries such as
`ElementCube 6525×3990×19 float32 1.98 GB space=raw`, and the last five
log lines. A cached step shows the first 8 characters of its recipe hash.
When the run ends, the last frame stays on screen as the summary.

With `--plain`, or when stdout is not a terminal (a pipe, `nohup`, a CI
log), the same information prints as plain lines.

`karak run` exits with 1 when a step fails and with 130 on Ctrl-C;
completed steps stay cached.

To inspect the outputs of the load through fingerprint steps, open them in
napari (install the extra once with `uv sync --extra view`):

```
uv run --extra view karak view BASE [--show Fe-K,Si] [--mask mask/Valid_mask.csv]
```

`BASE` is the run's `--out` value, its work or cache directory, or a single
cached `.h5` file copied from another machine. Given an `--out` value, the
command opens the outputs listed in the latest run record
(`{out}/runs/latest/run.json`): the load step's element cube, the BSE
image from the same step, the mask step's masks, the denoise step's cube,
the normalize step's cube, the PCA step's features, the HDBSCAN step's
labels, and the noise-reassignment step's labels, and it prints the
statistics and fingerprints computed from those labels. Without a record
(or if its cache files are gone) it opens the newest cached load cube, and
the newest cached masks and denoised cube computed from it, the normalized
cube computed from that, the PCA features computed from the normalized
cube, the raw labels computed from those features, and the cleaned labels
computed from the raw ones, instead (each cached output records the recipes it consumed;
outputs from another run or cube are never overlaid; the masks shown
with a denoised cube are the ones it consumed, and a cached file without
these recipes is only opened through its run record). It shows one gray
layer per element (only `--show` elements visible), the denoised elements
as `dn: <element>` layers and the z-scores as `nrm: <element>` layers
(both visible for the same `--show` elements; z-score layers take their
contrast limits from the 1st and 99th percentiles of the mineral pixels),
one `pca: PC<k>` layer per kept principal component (hidden; the
component's scores at the mineral pixels and 0 elsewhere, with contrast
limits from the 1st and 99th percentiles of the scores), the HDBSCAN
phases as the `hdb: phases` labels layer (visible; phase k drawn as k + 1,
noise and non-mineral pixels transparent), the noise pixels as `hdb: noise`
(hidden), the membership probabilities as `hdb: probability` (hidden), and
the reassigned phases as `knn: phases` (hidden; the same colours, so
toggling it against `hdb: phases` shows where the noise pixels went),
then the mineral mask as a labels layer (visible) and, when the flow set
`msk.valid_mask_path`, the valid mask as a second labels layer (hidden).
Toggle an element and its `dn:` layer to compare raw and denoised. With
the denoised and normalized cubes the viewer holds three cubes in memory
(about 6 GB for NWA 4587), plus one image per kept component (9
components, about 0.9 GB for NWA 4587). The command prints the explained
variance of each kept component, the HDBSCAN phase count, noise share
and pixels per phase, the pixels per phase after reassignment with
each phase's gain, each phase's pixels, share and mean probability
(`stats:`), and each phase's three strongest mean element intensities plus
the cosine-similar phase pairs (`fp:`). Every layer is placed in full-resolution
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
- **Declared judgments.** Every split or naming decision is a node with
  a `note` param; `clusters/subclustering` in the HDF5 lists them in order
  with their pixel counts and component means.
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
- `--workers N` (0 = all cores) runs the load, denoise and tiled HDBSCAN
  steps in N processes, and on the cpu `hdbscan_global` and the pass-2
  HDBSCAN of `rare_phase` use N jobs for the core distances and N
  processes for `approximate_predict` after a `subsample_n` fit. Results are identical for any N. On NWA 4587 at
  `downsample_factor` 8 (782 k pixels, `min_samples` 1000) with 16 workers,
  the prediction takes 35 s instead of 360 s and a full fit 117 s instead
  of 238 s. The prediction runs in chunks of at most 2 GB together; a full
  fit (no `subsample_n`) still holds about 25 GB at that size.
- A run with `--set src.downsample_factor=4 --set
  src.include_elements=Fe-K,Ca,Mg,Si` checks an installation in minutes on
  a laptop.
- No GPU is required. With the `cuda` extra and `--device cuda`, denoise,
  normalize, PCA, HDBSCAN (including the pass-2 HDBSCAN of `rare_phase`)
  and noise reassignment run on the GPU.
- cuML HDBSCAN needs device memory in proportion to the fitted pixel count
  times `min_samples` (it builds a `min_samples`-neighbour graph). The
  `global` flow at full NWA 4587 scale (12.5 M pixels, `min_samples` 1000)
  does not fit on a 24 GB GPU without `subsample_n`. With `subsample_n`,
  cuda fits the same random subsample as the cpu and then assigns every
  pixel with `approximate_predict`, in batches sized from the free device
  memory: with the [full-scale settings](#hdbscan-settings-for-full-scale-runs)
  `hdb` takes about 15 s on NWA 4587 on an RTX 4090. With `min_samples`
  1000, 50000 takes about a minute and 200000 runs out of memory in the
  cuML fit. For `hdbscan_global` with `subsample_n`, labels agree with the
  cpu but are not identical (adjusted Rand index 1.000 at 50000 / 125 on
  NWA 4587).
- On full tiles, cuML can select different clusters than the cpu. With the
  paper's tiled settings (`min_cluster_size` 100, `min_samples` 25, 1024 px
  tiles) on NWA 4587, the cpu reproduces the published per-tile cluster
  counts, but cuda finds 12 phases instead of 11 and 4.2 M noise pixels
  instead of 2.8 M (adjusted Rand index 0.57 against the cpu); in one
  733 k-pixel tile cuML finds 2 clusters where the cpu finds 8, whatever its
  options or input precision. `hdbscan_tiled` therefore warns when it runs
  on cuda. Use `--set hdb.device=cpu` for tiled results that match the cpu
  or the published baseline.
- The tiled flows on cuda fit cuML once per tile, so the same limit applies
  per tile. A full 512 px tile (up to 262 k pixels) runs out of memory with
  `min_samples` 1000 even on an empty 24 GB GPU. Set `hdb.subsample_n`
  (50000 runs `hdb` in about 105 s on NWA 4587) or a smaller
  `hdb.tile_size` (one full 256 px tile, 65 k pixels, fits in 1 s). To run HDBSCAN on the
  CPU after GPU earlier steps, use `--device cuda --set hdb.device=cpu`.
  `--device` sets every node that has a `device` param, and an explicit
  `--set NODE.device=...` then overrides it for that node.
