<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/karak-logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/karak-logo-light.svg">
  <img alt="Karak" src="docs/images/karak-logo-light.svg" width="500">
</picture>

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

**Karak** is an automated mineralogy pipeline that transforms SEM-EDS elemental map PNGs into mineral phase maps with chemical fingerprints. Named after the dwarven word for *stronghold*, it digs into the hidden structure of rocks, meteorites, and other geological samples.

<p align="center">
  <img src="docs/images/phase_map_example.png" alt="Mineral phase map of meteorite NWA 4587" width="350">
  <br>
  <em>Automated mineral phase map of meteorite NWA 4587</em>
</p>

## Paper

Karak accompanies the manuscript:

> Izawa, M. R. M., Hall, B. J., Cao, F., Luo, T., Yokoyama, S. T., & Zhao, Y. S.
> **Unsupervised mineral phase mapping from SEM-EDS element maps: a
> density-based clustering pipeline with multi-resolution refinement.**
> *Computers & Geosciences* (submitted 2026).

If you use Karak in your research, please cite the paper and the software —
citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Features

- **End-to-end pipeline** — from raw jet-colormapped PNGs to labeled mineral phase maps in a single command
- **Modular stages** — each processing step is a self-describing stage with typed, bounded parameters and named input/output ports
- **JSON flow graphs** — pipelines are DAGs of stages defined in JSON; three builtin flows cover the standard workflows
- **Per-node caching** — content-addressed caching of every stage output; change one parameter and only downstream stages re-run
- **HDBSCAN clustering** — density-based clustering discovers mineral phases without requiring a predefined number of clusters
- **Tiled progressive strategy** — process large images tile-by-tile with automatic phase registry unification across tiles
- **Two-pass workflow** — optional second pass to detect rare mineral phases with a finer clustering resolution
- **Post-clustering refinement** — GMM-based splitting of composite phases (e.g., pyroxene into pigeonite + augite)
- **Full provenance** — the flow definition, library versions, and timestamps embedded in every output file
- **QC diagnostics** — figure sinks generate detailed diagnostics at each stage for visual validation
- **Legacy YAML mode** — existing `config.yaml` files keep working; they are converted to flows internally

## Installation

Requires Python 3.12+. Clone the repository and install with
[uv](https://docs.astral.sh/uv/) (recommended — reproduces the exact locked
environment used for the paper):

```bash
git clone https://github.com/brendonhall/karak.git
cd karak
uv sync            # creates .venv from uv.lock
uv run karak --help
```

Or install with pip into an existing environment:

```bash
git clone https://github.com/brendonhall/karak.git
cd karak
pip install -e .
```

## Quick Start

### 1. Organize your data

Place your SEM-EDS elemental map PNGs in a directory. Each PNG should be a jet-colormapped image named after the element it represents (e.g., `Si.png`, `Fe-K.png`, `Mg.png`). Include a `SEM.png` for the backscatter electron image.

```
data/
├── SEM.png
├── Si.png
├── Fe-K.png
├── Mg.png
├── Ca.png
├── Al.png
├── Na.png
└── config.yaml
```

### 2. Create a configuration file

```yaml
# config.yaml — minimal example
input_dir: "."
hdf5_output: "eds_pipeline.h5"
figure_dir: "figures"

exclude_elements: ["Fe-L"]
bse_channel: "SEM"

downsample:
  header_trim_px: 100
  downsample_factor: 2

mask:
  min_object_size: 100

denoise:
  method: bilateral
  sigma_spatial: 1.0

cluster:
  strategy: global
  pca:
    n_components: null       # inspect scree plot, then set
  hdbscan:
    min_cluster_size: 1000
    subsample_n: 500000
    noise_reassign_k: 5
```

### 3. Run the pipeline

```bash
cd data
karak                    # reads ./config.yaml
```

Or skip the YAML file entirely and run a builtin flow:

```bash
karak run --builtin global --input data/ --out output/sample
```

Either way, Karak runs the full pipeline and writes results to an HDF5 file alongside QC figures.

> **Note:** The first run with a given colormap builds a 256³ RGB-to-scalar
> lookup table (30–60 s). It is cached on disk and reused by all later runs.
> For a fast installation check, run `karak --test-mode` (4 elements at 4x
> downsample).

## Usage

Flow mode (recommended):

```bash
karak run --builtin global --input data/ --out output/sample     # standard pipeline
karak run --builtin tiled --input data/ --out output/sample      # tiled clustering
karak run --builtin tiled-rare --input data/ --out output/sample # tiled + rare phases
karak run my_flow.json --input data/ --out output/sample         # custom flow
karak run --builtin global --set hdb.min_cluster_size=500 ...    # override any parameter
karak validate my_flow.json        # structural checks without running
karak schema                       # print every stage's parameter schema as JSON
```

Legacy YAML mode (converted to a flow internally, results identical):

```bash
karak -c path/to/config.yaml       # run from a YAML config
karak --test-mode                  # fast validation (4 elements, 4x downsample)
karak --no-qc                      # skip QC figure generation
karak --clean                      # delete previous HDF5 and cache, start fresh
karak --emit-flow -c config.yaml   # print the equivalent flow JSON, then migrate
karak -v                           # enable debug logging
```

Re-running a flow is cheap: every stage output is cached under
`<out dir>/work/cache/`, keyed by the stage's parameters, its upstream
results, and the input files' signatures. Change one parameter and only the
affected stages re-run. This replaces the old `--from-stage` resume flag.

## Architecture

Karak is built in three layers. Higher layers wrap lower ones and never
reimplement the math.

```
┌─ flow/     graph model · validation · cache · executor · CLI   (orchestration)
│            runs JSON DAGs of stages headlessly, caches every node output
├─ stages/   self-describing steps: typed params + named ports    (composition)
│            one thin class per operation, over the numeric core
└─ io/ preprocessing/ clustering/ identification/ qc/             (numeric core)
             pure functions on numpy arrays
```

Stages pass immutable payloads (element cubes, masks, PCA features, labels)
between named ports. Each cube carries a space tag (`raw`, `denoised`,
`normalized`) and each label array a state tag (`raw`, `cleaned`), so the
validator rejects nonsensical connections before anything runs. `karak
schema` prints the full stage palette — the contract a graphical editor
would consume.

## Pipeline

The builtin `global` flow runs the standard sequence:

```
    PNG element maps
          │
    ┌─────▼─────┐
    │   Load     │  Read PNGs, invert jet colormap, build compositional cube
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │   Mask     │  Separate mineral pixels from background / epoxy
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Denoise   │  Edge-aware smoothing (bilateral or anisotropic diffusion)
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │ Normalize  │  Per-channel z-score normalization
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │  Cluster   │  PCA → HDBSCAN → k-NN noise reassignment
    └─────┬─────┘
          │
    Phase map + chemical fingerprints
```

In flow terms these are the stages `load_elements → mask → denoise →
normalize → pca → hdbscan_global → noise_assign → cluster_stats →
fingerprints → export_h5`, plus QC figure sinks. The `tiled` flow swaps in
`hdbscan_tiled`; `tiled-rare` adds a `rare_phase` stage; a `refine` stage
(olivine extraction + GMM split) can be added to any flow. Optional
post-run stages `qc_named_phase_map` and the notebook helpers
`save_mineral_names`/`load_mineral_names` attach researcher-assigned
mineral names.

### Clustering Strategies

| Strategy | Best for | Description |
|----------|----------|-------------|
| `global` | Small–medium images | Single HDBSCAN run on entire image |
| `tiled` | Large images | Per-tile HDBSCAN with cosine-similarity phase registry unification |

The tiled strategy divides the image into spatial tiles, clusters each independently, then merges phases across tiles by matching their chemical fingerprints. This keeps memory usage bounded regardless of image size.

## Configuration

Flows are JSON: `nodes` (a stage `type` plus a `params` dict) connected by
`edges` (output port to input port). Copy a builtin from
[`src/karak/flow/flows/`](src/karak/flow/flows/) as a starting point, or
print one from an existing YAML config with `karak --emit-flow -c
config.yaml`. String parameters accept the run-scoped tokens `{input}`,
`{out}`, and `{work}`, so one flow file works across datasets.

```json
{
  "id": "dn", "type": "denoise",
  "params": {"method": "bilateral", "sigma_spatial": 1.0}
}
```

The legacy YAML config remains fully supported. Below are some key
parameters beyond the minimal example above.

### Tiled clustering

```yaml
cluster:
  strategy: tiled
  tiled:
    tile_size: 512
    merge_threshold: 0.92       # cosine similarity for phase matching
    min_clusters_per_tile: 3    # tiles with fewer clusters deferred to k-NN
```

### Two-pass rare phase detection

```yaml
cluster:
  rare_phase:
    enabled: true
    min_cluster_size: 50        # much smaller than primary pass
    subsample_n: 500000
```

### Post-clustering refinement

```yaml
cluster:
  refinement:
    enabled: true
    target_phase: 2             # cluster label to refine
    olivine:
      enabled: true
      fe_threshold: 0.6
      ca_threshold: 0.10
    gmm_split:
      enabled: true
      n_components: 2           # e.g., pigeonite + augite
      features: ["Ca", "Mg", "Fe-K", "BSE"]
```

## Documentation

The **[User Guide](docs/user_guide.md)** covers input data expectations
(element maps, BSE image, valid-region polygon mask), the five pipeline
stages, the complete `pipeline_config.yaml` schema (every field with its
default and meaning), the HDF5 output layout, checkpoint/resume, and
reproducibility notes.

## Hardware Requirements

Karak is CPU-only (no GPU required). Development and the analyses in the
paper were run on an **AMD Ryzen AI 5 340 with 32 GB RAM** (Linux).

- Full-resolution runs (~7400 x 5400 px, 21 element channels) use
  **~10–12 GB RAM**.
- Memory can be bounded for larger images via `cluster.hdbscan.subsample_n`
  or the `tiled` clustering strategy (constant per-tile memory).
- `karak --test-mode` validates an installation in minutes on a laptop.

## Testing

A fast pytest suite (config round-trip, colormap-inversion round-trip on a
synthetic ramp, mask utilities, per-stage parity against the numeric core,
flow validation/caching/executor behavior, and end-to-end flow runs on a
synthetic two-phase scene) runs in well under a minute:

```bash
uv sync
uv run pytest
```

The same suite runs in CI on every push and pull request
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## How It Works

**Jet colormap inversion** — SEM-EDS software commonly exports elemental maps as jet-colormapped PNGs. Karak inverts these back to scalar intensity values using a precomputed 256<sup>3</sup> RGB-to-scalar lookup table derived from matplotlib's jet colormap. The LUT is cached to disk and reused across runs.

**HDBSCAN clustering** — Unlike k-means, [HDBSCAN](https://hdbscan.readthedocs.io/) discovers the number of clusters automatically from data density. Pixels that don't belong to any dense region are labeled as noise and later reassigned to their nearest cluster via k-nearest-neighbor voting.

**Provenance tracking** — Every HDF5 output file embeds the full pipeline YAML config, library versions (NumPy, scikit-learn, HDBSCAN, etc.), Python version, and platform info as root-level attributes. This means any output file is self-documenting and fully reproducible.

## Authors

- **Brendon Hall** — [@brendonhall](https://github.com/brendonhall) · [ORCID](https://orcid.org/0000-0002-2244-4994)
- **Matthew Izawa** — [@matthewizawa](https://github.com/matthewizawa) · [ORCID](https://orcid.org/0000-0001-5456-2912)

## Contributing

Contributions are welcome! Please open an issue to discuss proposed changes before submitting a pull request.

```bash
git clone https://github.com/brendonhall/karak.git
cd karak
pip install -e .
```

## License

[MIT](LICENSE)
