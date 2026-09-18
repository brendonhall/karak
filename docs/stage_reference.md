# Stage reference

Every karak processing step is a stage: one class with named input/output
ports and typed, bounded parameters. This file is generated from the live
stage registry (`karak schema` prints the same contract as JSON).

Do not edit by hand. Regenerate with:

```bash
uv run python -m karak.stages.reference
```

Port type tags: cubes carry a space (`raw`, `denoised`, `normalized`) and
label arrays a state (`raw` = may contain -1, `cleaned` = fully labeled).
The flow validator rejects connections whose tags disagree. A dash means
the port accepts its payload type without a tag constraint. Stages with no
outputs are sinks: they run for their side effect (a file or a figure) and
are never cached.

## `cluster_stats` — Cluster statistics

Cluster counts, sizes, and noise fraction for the labels.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels` | `cleaned` | yes | final labels with membership probabilities |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `stats` | - | cluster counts, sizes, and noise fraction |

**Parameters**: none

## `denoise` — Denoise

Edge-aware denoising (bilateral or Perona-Malik anisotropic diffusion) applied per channel on raw intensities.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube` | `raw` | yes | (H, W, C) raw element cube |
| `masks` | - | yes | mineral mask restricts smoothing to sample pixels |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `cube` | `denoised` | (H, W, C) denoised cube, same channels and geometry |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `method` | enum | `bilateral` | bilateral \| anisotropic_diffusion | Method |
| `sigma_color` | float | `None` | 0.0.. | Bilateral color sigma (None = auto from data range) |
| `sigma_spatial` | float | `1.0` | 0.0.. | Spatial sigma |
| `niter` | int | `10` | 1.. | Iterations |
| `kappa` | float | `50.0` | 0.0.. | Conductance coefficient for diffusion |
| `gamma` | float | `0.1` | 0.0..0.25 | Diffusion speed (0-0.25 stable) |
| `option` | int | `2` | 1 \| 2 | Perona-Malik option |
| `device` | str | `cpu` | cpu \| cuda | cpu or cuda (GPU bilateral via cuCIM; needs karak[cuda]). Results match cpu within float tolerance. |

## `export_h5` — Export HDF5

Write connected results to the provenance HDF5 file using the legacy group layout. Every input is optional; only connected groups are written.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube_raw` | `raw` | no | written to the raw/ group, one dataset per element |
| `bse` | - | no | written to bse/image |
| `masks` | - | no | written to masks/mineral and masks/valid |
| `cube_denoised` | `denoised` | no | written to denoised/cube |
| `cube_normalized` | `normalized` | no | written to normalized/{cube,means,stds} |
| `features` | - | no | supplies pca_variance_ratio and the component count |
| `labels_raw` | `raw` | no | written to clusters/raw_labels |
| `labels` | `cleaned` | no | written to clusters/cleaned_labels (+ probabilities) |
| `stats` | - | no | cluster statistics stored as clusters/ attributes |
| `tiles` | - | no | written to clusters/tiled/ (tile metadata + registry) |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `path` | str | `{out}.h5` | - | Output path |
| `flow_json` | str | `{flow}` | - | JSON of the executing flow, embedded for provenance |

## `fingerprints` — Chemical fingerprints

Per-cluster mean/std element intensities from the denoised cube, with cosine-similar cluster pairs flagged for review.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels` | `cleaned` | yes | final phase labels |
| `cube` | `denoised` | yes | denoised intensities the fingerprints are computed from |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `fingerprints` | - | per-cluster mean/std spectra + flagged similar pairs |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `similarity_threshold` | float | `0.95` | 0.0..1.0 | Cosine similarity above which cluster pairs are flagged |

## `hdbscan_global` — HDBSCAN (global)

Single HDBSCAN run over all mineral-pixel features.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `features` | - | yes | PCA features for all mineral pixels |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `labels` | `raw` | per-pixel phase labels; -1 = HDBSCAN noise |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `min_cluster_size` | int | `1000` | 1.., px | Min cluster size |
| `min_samples` | int | `0` | 0.. | HDBSCAN min_samples; 0 = defaults to min_cluster_size |
| `subsample_n` | int | `0` | 0.. | Max pixels for fitting (rest via approximate_predict); 0 = all |
| `random_state` | int | `42` | - | Random seed |
| `device` | str | `cpu` | cpu \| cuda | cpu (hdbscan package, exact baseline) or cuda (cuML; needs karak[cuda]). cuda results differ from cpu and ignore subsample_n. |

## `hdbscan_tiled` — HDBSCAN (tiled)

Per-tile HDBSCAN with cosine-similarity phase-registry merging. Unassigned pixels are left at -1 for the noise_assign stage.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `features` | - | yes | PCA features for all mineral pixels |
| `cube` | `denoised` | yes | denoised cube; used to fingerprint tile clusters for registry matching |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `labels` | `raw` | registry-unified labels; -1 = unassigned/deferred |
| `tiles` | - | per-tile diagnostics + the phase registry |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `min_cluster_size` | int | `1000` | 1.., px | Min cluster size |
| `min_samples` | int | `0` | 0.. | HDBSCAN min_samples; 0 = defaults to min_cluster_size |
| `subsample_n` | int | `0` | 0.. | Max pixels for fitting (rest via approximate_predict); 0 = all |
| `random_state` | int | `42` | - | Random seed |
| `device` | str | `cpu` | cpu \| cuda | cpu (hdbscan package, exact baseline) or cuda (cuML; needs karak[cuda]). cuda results differ from cpu and ignore subsample_n. |
| `tile_size` | int | `512` | 1.., px | Tile size |
| `merge_threshold` | float | `0.92` | 0.0..1.0 | Cosine similarity for matching tile clusters to the registry |
| `min_tile_pixels` | int | `0` | 0.. | Minimum mineral pixels per tile; 0 = 2 * min_cluster_size |
| `min_clusters_per_tile` | int | `3` | 0.. | Tiles with fewer clusters defer to the k-NN pass |

## `load_elements` — Load element maps

Load false-color element map images, invert the colormap to scalar [0, 1] intensities, downsample/trim, and stack into a cube.

**Inputs**: none (source stage)

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `cube` | `raw` | (H, W, C) raw element cube |
| `bse` | - | (H, W) BSE grayscale image |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `input_dir` | str | `{input}` | - | Directory containing element map files |
| `file_glob` | str | `*.png` | - | Glob pattern (relative to input_dir) for element map files |
| `filename_pattern` | str | `None` | - | Pattern with {element} placeholder; None = legacy TIMA heuristic |
| `bse_filename` | str | `None` | - | Exact BSE file name when it does not match the glob |
| `colormap` | str | `cmap:jet` | - | 'cmap:NAME' or 'lut:PATH' inversion spec |
| `exclude_elements` | str | `Fe-L` | - | Comma-separated element names to skip |
| `include_elements` | str | `None` | - | Comma-separated allowlist; None = load all |
| `bse_channel` | str | `SEM` | - | Element name of the BSE/SEM channel |
| `header_trim_px` | int | `100` | 0.., px | Header trim |
| `bottom_trim_px` | int | `0` | 0.., px | Bottom trim |
| `left_trim_px` | int | `0` | 0.., px | Left trim |
| `right_trim_px` | int | `0` | 0.., px | Right trim |
| `downsample_factor` | int | `2` | 1.. | Downsample factor |

## `mask` — Mineral mask

Boolean mineral-pixel mask: all-zero pixels are background; an optional napari polygon CSV restricts the valid sample region.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube` | `raw` | yes | (H, W, C) raw element cube; all-zero pixels = background |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `masks` | - | mineral + valid masks with statistics |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `min_object_size` | int | `100` | 0.., px | Remove connected components smaller than this |
| `valid_mask_path` | str | `None` | - | napari shapes CSV of the sample boundary polygon (coordinates in original, pre-downsample image space) |

## `noise_assign` — Noise reassignment

Assign every remaining -1 pixel to its nearest phase by distance-weighted k-NN voting in feature space.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels` | `raw` | yes | labels that may still contain -1 pixels |
| `features` | - | yes | PCA features; the k-NN voting space |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `labels` | `cleaned` | fully labeled pixels; probabilities pass through |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `k` | int | `5` | 1.. | Neighbors |

## `normalize` — Normalize

Per-channel z-score normalization over mineral pixels; non-mineral pixels are set to 0.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube` | `denoised` | yes | (H, W, C) denoised cube |
| `masks` | - | yes | statistics are computed over mineral pixels only |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `cube` | `normalized` | z-scored cube; per-channel means/stds ride on the payload |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `method` | enum | `zscore` | zscore | Method |

## `pca` — PCA

Fit PCA on mineral pixels of the normalized cube and project them into reduced feature space. n_components=0 auto-selects the first component count reaching the cumulative-variance threshold.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube` | `normalized` | yes | z-scored element cube |
| `masks` | - | yes | defines which pixels are fitted and projected |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `features` | - | (N_mineral, n_kept) PCA features + pixel coordinates + explained variance ratios |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `n_components` | int | `0` | 0.. | Number of components to keep; 0 = auto from variance |
| `variance_threshold` | float | `0.95` | 0.0..1.0 | Cumulative explained variance target for auto selection |
| `min_components` | int | `5` | 1.. | Floor for auto selection |
| `subsample_fraction` | float | `0.0` | 0.0..1.0 | Fraction of mineral pixels used for fitting; 0 = all |
| `random_state` | int | `42` | - | Random seed |

## `qc_cluster_summary` — QC: cluster summary

Cluster size and probability summary chart.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `stats` | - | yes | cluster sizes and probabilities to chart |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |

## `qc_denoise` — QC: denoise

Before/after denoising comparison panels.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube_raw` | `raw` | yes | before panel |
| `cube_denoised` | `denoised` | yes | after panel |
| `bse` | - | yes | context panel |
| `masks` | - | yes | restricts the compared pixels |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |
| `method` | str | `bilateral` | - | Denoise method name shown on the figure |

## `qc_fingerprints` — QC: fingerprints

Per-cluster chemical fingerprint chart.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `fingerprints` | - | yes | per-cluster chemical signatures to chart |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |
| `mineral_names` | str | `None` | - | JSON mapping of cluster id to name, e.g. {"0": "olivine"} |

## `qc_mask` — QC: mask

Mask coverage overlay with optional TIMA reference panel.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `bse` | - | yes | background image for the overlay |
| `masks` | - | yes | mineral + valid masks to visualize |
| `cube_raw` | `raw` | no | Supplies downsample/trim geometry for TIMA alignment |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |
| `tima_path` | str | `None` | - | Optional reference TIMA phase map image for comparison |

## `qc_named_phase_map` — QC: named phase map

Final phase map with researcher-assigned mineral names.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels` | `cleaned` | yes | final phase labels |
| `bse` | - | yes | grayscale underlay |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |
| `mineral_names` | str | `{}` | - | JSON mapping of cluster id to name, e.g. {"0": "olivine"} |

## `qc_normalize` — QC: normalize

Z-score histograms and channel correlation matrix.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `cube` | `normalized` | yes | z-scored cube for histograms + correlation matrix |
| `masks` | - | yes | restricts statistics to mineral pixels |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |

## `qc_phase_map` — QC: phase map

Raw vs cleaned phase map over the BSE image.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels_raw` | `raw` | yes | left panel |
| `labels` | `cleaned` | yes | right panel |
| `bse` | - | yes | grayscale underlay |
| `stats` | - | yes | cluster counts for the legend |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |

## `qc_scree` — QC: scree plot

PCA explained-variance scree plot with selection cutoff.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `features` | - | yes | supplies explained variance ratios and the cutoff |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |

## `qc_tiled` — QC: tiled clustering

Tile grid overlay and phase discovery chart. Recomputes the tile grid from the features payload.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `bse` | - | yes | underlay for the tile grid overlay |
| `tiles` | - | yes | tile diagnostics + registry to chart |
| `features` | - | yes | pixel coordinates to recompute the grid |

**Outputs**: none (sink)

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `figure_dir` | str | `{out}/figures` | - | Directory the diagnostic figures are written to |
| `min_tile_pixels` | int | `2000` | 1.. | Must match the hdbscan_tiled setting for an accurate overlay |

## `rare_phase` — Rare phases

Recluster still-unassigned pixels with more sensitive HDBSCAN parameters; novel clusters join the phase registry. Presence of this stage in a flow is what enables the two-pass workflow.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels` | `raw` | yes | Pass-1 labels; -1 pixels are the recluster candidates |
| `features` | - | yes | PCA features for all mineral pixels |
| `cube` | `denoised` | yes | denoised cube; fingerprints rare clusters for matching |
| `tiles` | - | yes | phase registry from the tiled pass |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `labels` | `raw` | labels with rare phases assigned; residual -1 remains |
| `tiles` | - | registry extended with the new rare phases |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `min_cluster_size` | int | `50` | 1.., px | Min cluster size |
| `min_samples` | int | `0` | 0.. | 0 = defaults to min_cluster_size |
| `subsample_n` | int | `500000` | 0.. | Max unassigned pixels to fit on; 0 = all |
| `merge_threshold` | float | `0.0` | 0.0..1.0 | Cosine similarity vs existing registry; 0 = reuse the tiled merge threshold |
| `noise_reassign_k` | int | `5` | 1.. | kNN k |
| `random_state` | int | `42` | - | Random seed |

## `refine` — Phase refinement

Split a composite phase: threshold-based olivine extraction, then a GMM split of the remaining target-phase pixels.

**Inputs**

| port | type tag | required | notes |
|------|----------|----------|-------|
| `labels` | `cleaned` | yes | cleaned labels containing the target composite phase |
| `cube` | `denoised` | yes | denoised intensities for thresholds and GMM features |
| `bse` | - | yes | BSE image; optional GMM feature channel |

**Outputs**

| port | type tag | notes |
|------|----------|-------|
| `labels` | `cleaned` | labels with the target phase split into sub-phases |

**Parameters**

| name | type | default | bounds / choices | help |
|------|------|---------|------------------|------|
| `target_phase` | int | `2` | - | Cluster label of the phase to refine |
| `olivine_enabled` | bool | `False` | - | Olivine extraction |
| `olivine_fe_threshold` | float | `0.6` | 0.0..1.0 | Olivine Fe threshold |
| `olivine_ca_threshold` | float | `0.1` | 0.0..1.0 | Olivine Ca threshold |
| `gmm_enabled` | bool | `False` | - | GMM split |
| `gmm_n_components` | int | `2` | 2.. | GMM components |
| `gmm_features` | str | `Ca,Mg,Fe-K,BSE` | - | Comma-separated channel names; 'BSE' adds backscatter |
| `gmm_bse_weight` | float | `1.0` | 0.0.. | BSE weight |
| `gmm_subsample_n` | int | `500000` | 0.. | Max pixels to fit GMM on; 0 = all |
| `random_state` | int | `42` | - | Random seed |
