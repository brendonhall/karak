# The reproduction repository: `nwa4587-karak-paper` regenerates the paper's main results from one karak run

Date: 2026-10-09 (sections 1 to 4 approved in conversation 2026-10-13 to
2026-10-16). Status: approved in conversation; implementation plan to
follow.

Sub-project 2 of the publication plan. Sub-project 1 (the complete `paper`
flow, `docs/superpowers/specs/2026-10-06-paper-flow-complete-design.md`)
is merged. Sub-projects 3 to 6 (paper corrections, karak release, data
deposit, speed) get their own specs.

## Goal

A public repository, `brendonhall/nwa4587-karak-paper`, turns the output
of `karak run --builtin paper` into the figures, tables and numbers of
the paper, with one command per result and one `--check` against the
committed expected numbers. A reader with the deposited HDF5 regenerates
every main result without the private paper repository.

## Context

The paper repository `NWA4587_LPSC26` stays private. Its figure and
analysis scripts were written against the March 2026 HDF5
(`data/eds_pipeline.h5`) and a `karak.config` module that no longer
exists. An inventory on 2026-10-13 found:

- The compiled paper numbers its figures and tables differently from the
  script file names. Fingerprints is Fig 2 (file `fig5`), the xenotime
  callout is Fig 6 (file `fig7`), the hires pyroxene refinement is Fig 7
  (file `fig6`), the pyroxene TIMA close-up is Fig 5. Table 1 is the
  parameter table, Table 2 the abundances, Table 3 the TIMA
  correspondence, Table 4 the concordance metrics. This spec uses the
  paper numbers.
- Table 2 has no script; its values are typed into `main.tex`.
- Every TIMA figure decodes the karak 1x map from a rendered PNG
  (`poster/figures/phase_map_1x_clean.png`) by exact RGB. Epoxy and
  unresolved pyroxene are absent from the decode table and become -1.
- The palette and display names are duplicated in eight files. The
  canonical palette is `poster/make_phase_map_1x.py` lines 39 to 58,
  which Fig 1 imports.
- Several numbers quoted in the paper exist only on stdout: the pyroxene
  GMM means 0.54 and 0.30 and the 48.1 / 51.9 % split, the colormap
  round-trip errors 0.14 % and 1.6 %, the TIMA confusion matrix, the
  Fig 5 close-up window and the xenotime region of interest.
- `make_fig_lamellae.py` hard-codes a downsample factor of 4 against a 2x
  file. The published lamellae numbers (128 grains, 59 analyzed, median
  51 µm, Rayleigh p 0.008) come from that mismatch. The correct analysis
  on the regenerated 1x map gives 272 grains, 154 analyzed, median 45 µm,
  p 0.80.

The regenerated run (`karak run --builtin paper`, CPU, 2026-10-07, karak
commit 7ccd7a7) is the reference. Its HDF5 holds `clusters/cleaned_labels`
with 15 phases that carry pixels (labels 0 to 16 minus 2 and 7),
`clusters/mineral_names`, `clusters/subclustering` (the `history` JSON and
one `split_NN` attribute per split, with component means and pixel
counts), `clusters/hires/labels` (int16, 12950 by 7980, labels 13
pigeonite and 14 augite, -1 elsewhere, attributes `ratio`,
`downsample_factor`, `header_trim_px`, `left_trim_px`, `names`), `bse`
with `downsample_factor` and shapes, `denoised/cube` with
`element_order`, `masks/mineral` and `masks/valid`, and the flow JSON as
the root attribute `pipeline_config`. It has no fingerprint dataset: the
`fp` node is not wired to `exp`. Label numbers differ from the published
file (13 and 14 swap), so the repository resolves phases by name.

## Decisions already made

- New public repository `brendonhall/nwa4587-karak-paper`. Code under
  MIT; generated figures, tables and numbers under CC BY 4.0. The paper
  repository's scripts are ported, not copied.
- v1 regenerates the main results only: Table 1 to Table 4, Fig 1 to
  Fig 8, the pyroxene close-up and the colormap round-trip numbers.
  Sensitivity runs and the NWA 4587-3 transfer are later opt-in targets.
- One small package `nwa4587_paper` with a staged CLI `paper`, one
  constants module, `numbers.json` plus `numbers.tex`, synthetic tests,
  a CI smoke run, and `--check` against committed expected numbers.
- karak is pinned to git commit 7ccd7a7 (the merge of the `paper` flow)
  until the PyPI release of sub-project 4. Later commits on `main` are
  documentation only.
- The TIMA raster is optional; it is under a TESCAN data agreement.
- The numbers the repository quotes and checks are the regenerated ones,
  not the March ones. Sub-project 3 revises the paper text to match.
- Karak does not change for this sub-project. One finding is logged as a
  separate karak issue: in a cache scan `karak view` shows the cleaned
  labels that follow the raw labels directly (the `knn` output), not the
  last labels in the chain.

## Section 1: layout and data access

```
pyproject.toml            # package nwa4587_paper, CLI entry point `paper`
src/nwa4587_paper/
  cli.py                  # the staged CLI
  constants.py            # section 3
  io.py                   # the Run reader
  numbers.py              # registry, merge, tex rendering, --check
  tables.py fingerprints.py tima.py pyroxene.py xenotime.py
  lamellae.py colormap.py workflow.py     # one module per stage
expected/numbers.json     # the regenerated numbers --check compares against
expected/h5_layout.json   # dataset and attribute names of the reference HDF5
tests/                    # synthetic fixtures, no real data
data/                     # git-ignored: eds/ PNGs and mask, tima/ raster
results/                  # git-ignored: paperflow.h5, figures/, tables/,
                          #   numbers/, numbers.json, numbers.tex
.github/workflows/ci.yml  # tests plus a smoke run on the synthetic HDF5
LICENSE                   # MIT
LICENSE-FIGURES           # CC BY 4.0, covers results/
README.md CITATION.cff
```

Data access has two entry points:

- `paper data --from PATH | zenodo:DOI` copies or downloads the inputs
  into `data/`: the 21 elemental PNGs and `mask/Valid_mask.csv` into
  `data/eds/`, and the TIMA phase PNG into `data/tima/` when present. In
  v1 the Zenodo option prints the deposit instructions and exits, until
  sub-project 5 gives a DOI.
- `paper karak` runs
  `karak run --builtin paper --input data/eds --out results/paperflow`
  (about 45 minutes on a CPU, `--workers 0`). Readers who do not run karak
  place the deposited `paperflow.h5` in `results/` and start at the next
  stage.

Every stage checks its inputs first and exits with one line that names
the missing file and the command that produces it. The TIMA stage exits
with status 0 when `data/tima/` is absent, so `paper figures` completes
without the raster.

## Section 2: stages and the numbers bridge

### The `Run` reader (`io.py`)

One class opens the karak HDF5 once and gives every stage the same views:

- `labels`: the dense 2x label image, -1 outside the mineral mask, built
  from `clusters/cleaned_labels` and `clusters/mineral_indices`.
- `names`: label to name, from `clusters/mineral_names`.
- `phase(key)`: the label whose name starts with `key` (the name before
  the first parenthesis, for example `Augite`). Never a label number.
  A missing key is an error that names the key and lists the names.
- `labels_before(node_id)`: the label image before the named split node.
  It walks the `history` records from the end back to that node and
  remaps each record's new labels to its parent. Fig 1 panel (d) and
  Fig 7 panel (a) need the map before `pyx`; the published file stored it
  as `cleaned_labels_pre_pyroxene_split`, the regenerated file does not.
- `history`: the parsed `history` JSON, with the `pyx` record's
  `component_means` and `n_pixels_hires`.
- `hires_labels`: `clusters/hires/labels` with its attributes.
- `labels_1x`: the 2x label image repeated `ratio` by `ratio`, cropped to
  the hires shape, with the hires labels written over it where they are
  not -1. Epoxy and -1 at 2x stay as they are.
- `bse`, `channel(name)` from `denoised/cube`, `mask`.
- `geometry`: downsample factor, header trim, 2x and 1x shapes, from the
  attributes; pixel size from `constants.PIXEL_SIZE_UM_1X` times the
  factor.
- `flow`: the flow JSON from the `pipeline_config` root attribute.

### Stages

Each stage is one module with one function `run(run, out, cfg) -> dict`
that writes its files under `results/` and returns its numbers.

| command | inputs | outputs | numbers returned |
|---|---|---|---|
| `tables` | labels, names, flow | `tables/table1_params.tex`, `table2_phases.tex` and `.csv` | pixel count and area % per phase (of all mineral pixels, epoxy and weathering included), total mineral pixels, phase count |
| `fingerprints` | denoised cube, labels | `figures/fig2_fingerprints.pdf` and `.png` | element order, per-phase mean and std per element |
| `tima` | `labels_1x`, TIMA raster, offset | `figures/fig3_phase_comparison`, `fig4_agreement_map`, `fig5_pyroxene_tima`; `tables/table3_correspondence.tex`, `table4_concordance.tex` and `.csv`; `tables/tima_confusion_split.csv`, `_combined.csv` | overall accuracy split and combined, Spearman r and p, per-phase precision, recall, F1 and pixel count, included pixel count, rare-phase overlaps, the close-up window |
| `pyroxene` | `hires_labels`, the `pyx` history record, `labels_before("pyx")`, BSE, Ca and Mg at 1x from `data/eds` | `figures/fig7_pyroxene_refinement` | 2x pyroxene pixel count, 1x augite and pigeonite counts and %, the two GMM means |
| `xenotime` | labels, BSE, Y and P | `figures/fig6_xenotime_callout` | component count, largest component size, bounding box, region of interest |
| `lamellae` | `hires_labels`, geometry | `figures/fig8_lamellae`, `tables/lamellae_grain_table.csv` | grains total and analyzed, mean orientation, Rayleigh R and p, median and quartile spacing, spacing range, Spearman rho and p, mean coherence, Nyquist limit |
| `colormap` | none | `numbers/colormap.json` only | jet and viridis mean absolute, maximum and RMS errors, percentiles |
| `workflow` | Ca PNG, mask, raw and denoised Ca, `labels_before("pyx")`, labels | `figures/fig1_workflow` | none |

`paper figures` runs every stage in the table. `paper all` runs `data`,
`karak`, `figures` and `numbers`.

Table 1 is generated from the flow nodes embedded in the HDF5 (trim,
factor, denoise, PCA, tile size, merge threshold, HDBSCAN, rare phase,
kNN, split features), so a changed flow changes the table. Table 2 lists
every phase that carries pixels, sorted by area; the published 92-pixel
"Pyroxene (unresolved)" row does not exist in the regenerated run, and the
README says why.

The TIMA stage ports the old algorithm as it is, except for the source of
the karak map:

1. Decode the TIMA PNG: trim the bottom 1000 rows, take the nearest of
   the 14 table colours by Euclidean RGB distance, set -1 when the
   distance exceeds 50.0 or the pixel is achromatic (channel range under
   30 and maximum over 20).
2. Align by the fixed offset: TIMA window (1919, 1051) to
   (1919 + 7980, 1051 + 12685), karak window (0, 0) to (7980, 12685).
   The karak map loses the bottom rows that the TIMA legend trim removes.
3. Count the confusion matrix over pixels whose labels are in neither
   exclusion set, with the split mapping and with the combined mapping
   (karak augite into pigeonite's row; TIMA high-Ca and "Pyroxene Bi" into
   low-Ca).
4. Compute accuracy, per-phase precision, recall and F1 with the shared
   label rule of the old script, and the Spearman abundance correlation
   over pixels where both labels are non-negative.
5. Rare phases: for each karak phase under the threshold, the TIMA phase
   that holds the largest share of its pixels, no exclusions.

The lamellae stage ports the published segmentation and measurements
without the GMM: 3 by 3 closing twice, 8-connected components, grains of
at least 500 px, analysis of grains of at least 2000 px, structure tensor
orientation at sigma 1.5, coherence threshold 0.2, 20 profiles per grain,
spacing from the profile autocorrelation, the Rayleigh test and the
Spearman size-spacing test. The augite and pigeonite labels come from
`hires_labels`; the pixel size is 3.0 µm at 1x.

### The numbers bridge (`numbers.py`)

- Every stage writes its returned dict to `results/numbers/<stage>.json`.
- `paper numbers` merges them into `results/numbers.json` (one object per
  stage) and renders `results/numbers.tex`, one `\newcommand` per leaf,
  named by stage and key in camel case with the `nwa` prefix, for example
  `\nwaTablesAugitePct`. Values carry the format of the registry.
- A registry lists every key with its type, display format and tolerance
  (absolute for percentages and angles, relative for counts over 1000
  and for errors, exact for small integers and names). A stage that
  returns a key outside the registry fails its test.
- `--check` on any stage or on `numbers` compares the fresh values with
  `expected/numbers.json` and reports every miss with key, expected,
  actual and tolerance. Exit status 1 on any miss.

### Rulings

1. The Fig 5 close-up window is not recorded anywhere in the paper
   repository. The stage uses a fixed window in `constants.py`, chosen
   once by the densest-pyroxene-window search and then frozen.
2. The karak 1x map comes from `clusters/hires/labels` over the upscaled
   2x labels, not from a rendered PNG. Epoxy and undefined pixels become
   -1 before the TIMA comparison, as the old decode produced.
3. The xenotime figure uses the canonical palette, not tab20.
4. Fig 1 panel (d) counts the phases of `labels_before("pyx")`; the
   pre-split apatite colour `#0072B2` is kept in the palette for that
   panel only.

## Section 3: the constants module

One file, `constants.py`, holds every value that the old scripts
duplicated or printed. No other module carries a literal colour, name,
offset, window or threshold. Tests import it, so a typo fails a test and
not a figure.

All phase tables are keyed by name key, never by label number, because
the published and regenerated files number the pyroxenes differently.

| group | contents |
|---|---|
| `PALETTE` | the poster 1x palette: Ilmenite `#8B4513`, Silica `#F0E442`, Plagioclase `#56B4E9`, Epoxy `#999999`, Spinel `#D55E00`, Calcite `#CC79A7`, Apatite `#0072B2` (pre-split panels only), FeOOH `#BBBBBB`, Zn `#882255`, Xenotime `#00FF00`, Olivine `#44AA99`, Weathering `#332288`, Augite `#E6AB02`, Pigeonite `#2B83BA`, Merrillite `#1B9E77`, Chlorapatite `#E7298A`, unassigned `#666666`; overlay alpha 0.75 |
| `DISPLAY_NAMES` | the short names of Fig 1 and Fig 2 (`Fe–Ti–Cr spinel`, `Augite (high-Ca px)`, `Pigeonite (low-Ca px)`, `Weathering assemblage`, and so on) and the four shorter Fig 4 labels (Spinel, Olivine, Silica, Fe-OOH) |
| `TIMA_PHASES` | the 14 TIMA colours with index and name: `(170,0,127)` Plagioclase, `(0,255,255)` Pyroxene low-Ca, `(103,70,156)` Pyroxene high-Ca, `(128,170,10)` Ilmenite Cr, `(0,85,0)` Olivine+Ilmenite, `(170,85,255)` Pyroxene Bi, `(170,170,0)` Apatite, `(255,170,255)` Calcite, `(85,85,255)` Quartz, `(85,0,255)` Unknown1, `(100,100,150)` Mixture, `(255,0,0)` Troilite, `(190,255,242)` Unknown2, `(0,85,255)` Hematite/Magnetite; their display names for Fig 3 and Fig 4 |
| `TIMA_MAPPING_SPLIT` | Plagioclase [0], Augite [2], Pigeonite [1], Olivine [4], Ilmenite [3, 4], Spinel [3], Merrillite [6], Chlorapatite [6], Calcite [7], Silica [8], FeOOH [13] |
| `TIMA_MAPPING_COMBINED` | as split, with Pigeonite [1] and no Augite row |
| `EXCLUDED_KARAK`, `EXCLUDED_TIMA` | split: karak {unassigned, Zn, Xenotime, Weathering}, TIMA {-1, 5, 9, 10, 11, 12}; combined: the same karak set, TIMA {-1, 9, 10, 11, 12} |
| `TIMA_OFFSET` | `DX = 1919`, `DY = 1051`; the verified pair karak (2160, 272) to TIMA (4079, 1323); legend trim 1000 rows; colour distance limit 50.0; achromatic rule (range under 30, maximum over 20) |
| `MATCHED_COLORS`, `LEGEND_ORDER` | the Fig 3 and Fig 4 colour pairs and legend orders for split and combined, the background grey (0.92), the agreement greens, reds and greys |
| `PIXEL_SIZE_UM_1X` | 3.0 (from the sample metadata; the HDF5 does not carry it) |
| windows | Fig 7 panel (a) crop at 1x rows 6061 to 6861, columns 3570 to 4370; the Fig 5 close-up window (size 520 at 1x, position frozen after one search); the xenotime pad of 80 px at 2x |
| `LAMELLAE` | grain minimum 500 px, analysis minimum 2000 px, closing iterations 2, structure tensor sigma 1.5, coherence 0.2, 20 profiles |
| `COLORMAP` | 4096 samples, a 256 cubed lookup table, 100 000 test colours, seed 0, colormaps jet and viridis |
| other | rare phase threshold 500 000 px; fingerprint exclusions (Epoxy, Weathering); the Table 3 notes text; the Table 1 row labels |

What stays out: the three obsolete poster palettes, the tab20 colours of
the old xenotime figure, the legacy 2x mapping and its control points,
and every `karak.config` value.

## Section 4: tests, CI, acceptance, licensing, delivery

### Synthetic fixture

One test helper writes a tiny HDF5 in the karak layout: a 64 by 48 map at
2x, a 128 by 96 hires map, six named phases including Augite, Pigeonite,
Epoxy and Xenotime, four history records in the `oliv`, `weath`, `pyx`
and `phos` form, BSE attributes, a denoised cube with Ca, Mg, Y and P
channels, masks, and the paper flow JSON as the `pipeline_config`
attribute. A layout test compares the fixture's dataset and attribute
names with `expected/h5_layout.json`, a listing taken from the reference
file, so drift in karak's export shows up here first.

### Unit tests (synthetic, fast)

- `Run`: phase lookup by key, an unknown key, `labels_before` undoes one
  and two splits, `labels_1x` overlays the hires labels, geometry from
  the attributes.
- Every stage runs on the fixture, writes its files, and returns exactly
  the keys the registry lists for it. A second run returns equal numbers.
- TIMA decode on a synthetic raster: exact colours, colours inside and
  outside the distance limit, grey pixels, the legend trim, and a shifted
  copy that the offset realigns to a known accuracy and confusion matrix.
  The combined mapping merges the pyroxene rows.
- Lamellae on a synthetic grain with stripes at a known angle and spacing
  recovers both within tolerance; grains under the size limits are
  dropped.
- Colormap round trip at a small sample size gives a mean absolute error
  under 1 %.
- Numbers bridge: merge, `numbers.tex` rendering, and `--check` with one
  value inside and one outside its tolerance, and with a missing key.
- Constants: every key is a prefix of a name in the names fixture; every
  TIMA mapping index exists in `TIMA_PHASES`.

### CI

GitHub Actions on Ubuntu with Python 3.12: `uv sync`, pytest, then a smoke
job that builds the fixture, runs `paper figures` and `paper numbers` on
it and checks that every expected file exists. The TIMA stage reports the
missing raster and exits 0. karak is a git dependency in `pyproject.toml`
at commit 7ccd7a7. No real data touches CI.

### Real-data acceptance (local)

`paper figures` and `paper numbers --check` on
`output/nwa4587/item2/paperflow.h5` with the raw PNGs and the TIMA
raster. `expected/numbers.json` holds the regenerated values:

- Table 2: the 15 abundances of the 2026-10-07 run (Plagioclase 28.923 %,
  Pigeonite 19.320 %, Augite 17.760 %, Weathering 14.140 %, Epoxy
  8.725 %, and the rest), 12,495,787 mineral pixels.
- Pyroxene: 4,633,456 pixels at 2x; 8,904,186 augite (48.0 %) and
  9,629,508 pigeonite (52.0 %) at 1x; GMM means 0.534 and 0.306.
- Lamellae: 272 grains, 154 analyzed, median spacing 45.0 µm, Rayleigh
  p 0.80.
- Xenotime: 213 pixels, main grain 200 pixels at 2x row 3418, column
  3554.
- TIMA: the metrics come from `hires_labels` for the first time, so the
  first acceptance run records them, and the PR body compares them with
  the published 84.1 % (split) and 99.1 % (combined).

Every figure is compared side by side with the published PDF in the PR.
The published `eds_pipeline.h5` and the paper repository are not
modified.

### Licensing

`LICENSE` is MIT for the code. `LICENSE-FIGURES` is CC BY 4.0 and covers
everything under `results/`. `CITATION.cff` names the paper. The README
states that the data are not in the repository, where they come from, and
that the TIMA raster is under a TESCAN agreement.

### Delivery: one PR each in the new repository, in this order

1. Skeleton, `constants.py`, `io.py`, the fixture, the layout test, CI.
2. `tables`, `fingerprints`, the numbers bridge with `--check`.
3. `tima`: Figs 3, 4, 5 and Tables 3, 4.
4. `pyroxene`, `xenotime`, `workflow`.
5. `lamellae`, `colormap`.
6. Real-data acceptance: `expected/numbers.json`, README, the comparison
   report.

The GitHub repository is created at step 1, with the user's consent.

## Out of scope

Sensitivity runs and the NWA 4587-3 transfer (later opt-in stages); paper
text (sub-project 3); the karak release and PyPI pin (4); the Zenodo
deposit and the `zenodo:` download (5); any karak change, including the
`karak view` label pick and an exported fingerprint dataset.
