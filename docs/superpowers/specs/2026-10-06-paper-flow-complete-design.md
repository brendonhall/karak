# The complete paper flow: every hand step of the NWA 4587 analysis becomes a declared node

Date: 2026-10-06. Status: approved in conversation; implementation plan
to follow.

Sub-project 1 of the publication plan (see "Context"). Sub-projects 2 to
6 (reproduction repo, paper corrections, karak release, data deposit,
speed) get their own specs.

## Goal

`karak run --builtin paper` produces the published 16-phase NWA 4587 map
in one run, with no hand edit afterwards. Every decision that was made by
hand in March 2026 is a node parameter in the flow file, with its reason
beside it. The flow also writes the full-resolution pyroxene map that the
lamellae analysis needs.

## Context: what the March 2026 run did by hand

The published `eds_pipeline.h5` (clusters stage completed 2026-03-16) is
the output of the tiled two-pass clustering plus five manual steps:

1. `tools/fix_tile.py` in the paper repo reset one 1024 px tile (rows
   3072 to 4095, columns 2048 to 3071; 1,043,317 pixels) to -1 and
   re-ran kNN. Pass 1 had found 2 clusters in that tile, below
   `min_clusters_per_tile` 3, so the whole tile went to pass 2, which
   put 99 % of it in phase 2 as a square block. Karak's `rare_phase`
   on the paper's own pass-1 noise, plus that reset, reproduces the
   published raw labels to within 6 of 12,495,787 pixels.
2. Olivine extraction from phase 2 (Fe-K > 0.6 and Ca < 0.10) and a
   GMM split of the rest of phase 2 (features Ca, Mg, Fe-K, BSE,
   Ca/(Ca+Mg)); the smaller component became label 12, later named
   "Weathering Assemblage" by hand. The code ran before it was committed.
3. `tools/merge_hires_to_phasemap.py`: Ca and Mg from the 1x PNGs,
   sampled at each 2x pyroxene pixel, a 1-D GMM on Ca/(Ca+Mg), labels
   13 (augite) and 14 (pigeonite). A second script fitted its own GMM
   for the 1x figure and the section 3.4 numbers, and the lamellae
   analysis fitted a third.
4. `tools/split_apatite.py`: a GMM on Cl, Na, Mg, F of phase 7, labels
   15 (merrillite) and 16 (chlorapatite), ordered by mean Cl.
5. Names typed into a notebook dictionary, then edited in a web app.

Karak's pass 1 matches the paper to within 4,111 noise pixels per tile;
the remaining difference is the OpenBLAS kernel of the CPU that ran the
paper (PCA features differ at 1e-5 relative), not code or library
versions. The result is therefore regenerated, not matched bit for bit.

## Decisions already made

- Reproducible means: the revised paper's numbers come from a karak
  rerun on this machine. Small shifts from the March numbers are stated.
- Defects become general rules that every flow gets; scientific
  judgments become declared parameters with a note.
- Karak holds every step that maps a sample. Paper-specific analysis
  and figures live in a separate public reproduction repo (sub-project 2).
- The 1x sampling stays: it is the "multi-resolution refinement" of the
  paper's title, and the lamellae analysis depends on it.

## Section 1: the deferred-tile rule

Today a tile with fewer than `min_clusters_per_tile` clusters becomes all
noise and pass 2 clusters its pixels with the rest of the noise.

New rule: `hdbscan_tiled` records which tiles were deferred;
`rare_phase` excludes their pixels from the pass-2 fit, the prediction
and the registry merge. They stay -1 and `noise_assign` fills them from
their feature-space neighbours, as the published result did.

- `TileResult` gains `deferred: bool`; `TiledArtifacts` gains
  `deferred_tiles: tuple[int, ...]`. Both are written to HDF5; cache
  files without them load as "no deferred tiles".
- `rare_phase` already takes the `tiles` port; no wiring changes. It
  logs the number of skipped pixels.
- `hdbscan_tiled.recipe_revision` and `rare_phase.recipe_revision`
  return a tag so old cache entries are not reused. `tiled-rare` and
  `paper` recipe hashes change; the golden table is regenerated.
- On NWA 4587 this also resets the three small deferred tiles (19,833,
  1,257 and 30,499 pixels) that the paper did not reset by hand. Pass 2
  had put 98 % of them in phase 2. Expected effect: under 0.5 pp in any
  phase.

## Section 2: the split and naming stages

Common to all split stages: input `labels` (cleaned), output `labels`
(cleaned). New labels are `max(label) + 1` onwards, so the numbers follow
the node order in the flow (NWA 4587: 11 olivine, 12 weathering, 13 and
14 pyroxene, 15 and 16 phosphates, as published). Every split stage has
a `note` string param, the reason for the split; the flow file and the
HDF5 carry it.

`Labels` gains `names: dict[int, str]` (empty by default) and
`history: tuple[dict, ...]`, one record per split (stage type, parent
label, new labels, pixel counts, a method summary, the note). Split
stages add to both. Cache files without them load with empty values.

### `split_threshold`

Inputs `labels`, `cube` (denoised). Params:

| name | type | default | meaning |
|---|---|---|---|
| `target_phase` | int | 0 | label to split |
| `rule` | str | `""` | comparisons joined by `&`, e.g. `Fe-K > 0.6 & Ca < 0.10`; operators `<`, `<=`, `>`, `>=` on channel names of the cube |
| `new_name` | str | `""` | name of the new label |
| `note` | str | `""` | reason |

Pixels of the target phase that satisfy every comparison get the new
label. Wraps `refinement._extract_olivine`, generalised to any channels.

### `split_gmm`

Inputs `labels`, `cube` (denoised), `bse` (optional). Params:

| name | type | default | meaning |
|---|---|---|---|
| `target_phase` | int | 0 | label to split |
| `features` | str | `""` | comma list: channel names, `BSE`, or a ratio `A/(A+B)` |
| `n_components` | int | 2 | GMM components |
| `bse_weight` | float | 1.0 | multiplier on the z-scored BSE column |
| `subsample_n` | int | 500000 | pixels fitted; 0 = all |
| `random_state` | int | 42 | seed |
| `keep_parent` | bool | true | the largest component keeps the parent label |
| `order_by` | str | `""` | feature whose component means (ascending) order the new labels; empty = by component size, descending |
| `new_names` | str | `""` | comma list, one name per new label |
| `note` | str | `""` | reason |

Features are z-scored within the phase, as `_gmm_split` does today.
`n_init` stays 5, covariance `full`. With `keep_parent` false every
component gets a new label and the parent empties.

### `split_hires`

Inputs `labels`, `cube_hires` (raw space; from a second `load_elements`
node at `downsample_factor` 1 with `include_elements` set to the channels
needed). Params:

| name | type | default | meaning |
|---|---|---|---|
| `target_phases` | str | `""` | comma list of labels that form the region |
| `feature` | str | `""` | one channel or a ratio `A/(A+B)` of `cube_hires` channels |
| `n_components` | int | 2 | GMM components |
| `subsample_n` | int | 500000 | pixels fitted; 0 = all |
| `random_state` | int | 42 | seed |
| `new_names` | str | `""` | comma list, ordered by ascending component mean of the feature |
| `note` | str | `""` | reason |

Procedure:

1. Resolution ratio `ds` = hires height / working height; it must be an
   integer and the widths must agree to the same ratio (else
   `StageError`).
2. Region = the working-resolution mask of the target phases, upscaled
   by repeating each pixel `ds` x `ds` and clipped to the hires shape.
3. Feature at every hires pixel of the region; a ratio with a zero
   denominator is undefined.
4. One GMM fit on a `subsample_n` draw of the defined values (seed
   `random_state`), then predict every defined pixel.
5. `labels_hires` output: `HiresLabels`, an int16 image at the hires
   shape, new labels inside the region, -1 outside and where the feature
   is undefined. Carries `ds`, the names, and the source node's trim and
   factor.
6. `labels` output: each working pixel takes the majority label of its
   defined children; a tie goes to the child at (r * ds, c * ds); a pixel
   with no defined child keeps its parent label (the paper's 92 pixels).

Memory: the two 1x channels of NWA 4587 are about 0.8 GB; the executor's
spill rule covers them.

### `name_phases`

Input and output `labels`. Params `names` (str; entries `label: name`
separated by `;`) and `note`. Sets names of the base phases. It sits
after `noise_assign`, so later splits log parent names. The label-12
judgment lives in the `weath` node's `new_names` and `note`, not in a
rename.

### Base addition: `Stage.check_params`

`check_params(cls, params) -> list[str]`, a classmethod that both the
flow validator and `Stage.run()` call after coercion. The default returns
an empty list. The split and naming stages parse `rule`, `features`,
`feature`, `target_phases`, `new_names` and `names` here and report each
error by param name, so `karak validate` catches them.

### Removal

`refine` is removed. `clustering/refinement.py` and the
`RefinementConfig` bundles in `core_params.py` stay, because
`split_threshold` and `split_gmm` call the core functions. Its parity
test (`test_cluster_stage_parity.py`) moves to the new stages; the
registry list in `test_registry.py`, the stage table and the `--set
refine.target_phase` example in the user guide, and the README's stage
paragraph change with it. No shipped flow uses `refine`.

## Section 3: the `paper` flow, export and QC

### Nodes

| node | type | params that matter |
|---|---|---|
| `src` | `load_elements` | unchanged: 2x, trim 100 px, `cmap:jet`, exclude Fe-L |
| `src_hires` | `load_elements` | same input dir, colormap and trim; `downsample_factor` 1; `include_elements` `Ca,Mg`; feeds only `pyx` |
| `msk`, `dn`, `nrm`, `pca`, `hdb`, `rare`, `knn` | as today | `hdb` and `rare` carry the deferred-tile rule |
| `names` | `name_phases` | the 11 published base names; label 8 is "Fe Oxyhydroxide (FeOOH)" (the final name, not the notebook's "Uncertain") |
| `oliv` | `split_threshold` | phase 2; `Fe-K > 0.6 & Ca < 0.10`; "Ferroan Olivine ((Fe,Mg)₂SiO₄)" |
| `weath` | `split_gmm` | phase 2; `Ca,Mg,Fe-K,BSE,Ca/(Ca+Mg)`; 2 components; `keep_parent` true; "Weathering Assemblage"; note: the smaller component carries S, Cl and altered mafic chemistry |
| `pyx` | `split_hires` | phases `2`; `Ca/(Ca+Mg)`; 2 components; 500 k subsample; "Pigeonite (low-Ca pyroxene)", "Augite (high-Ca pyroxene)" |
| `phos` | `split_gmm` | phase 7; `Cl,Na,Mg,F`; `keep_parent` false; `order_by` `Cl`; "Merrillite (Ca₉NaMg(PO₄)₇)", "Chlorapatite (Ca₅(PO₄)₃(Cl,F,OH))" |
| `stats`, `fp`, `exp`, QC sinks | as today | `exp` gains the `labels_hires` input |

The published label numbering results: 0 to 10 base, 11 olivine, 12
weathering, 13 pigeonite, 14 augite, 15 merrillite, 16 chlorapatite.
(The paper file has 13 = augite and 14 = pigeonite; the flow orders by
ascending Ca, so the two swap. The reproduction repo maps by name, not
number.)

`tiled-rare`, `tiled`, `global` and `stepwise` keep their nodes; only
the deferred-tile rule changes their results.

### Export

`export_h5`:

- writes `clusters/mineral_names` (JSON) and the `cluster_N_name`
  attributes from `labels.names`, so `storage.load_mineral_names` keeps
  working;
- writes `clusters/subclustering` attributes from `labels.history`, the
  layout the paper's analysis scripts read;
- keeps the current `cluster_config` keys and adds `refinement_nodes`,
  the ordered list of split node ids, types and params (two nodes now
  share a type, so the first-node-of-type lookup is not enough);
- writes a connected `labels_hires` to `clusters/hires/labels` (int16,
  gzip) with `ds`, names and source geometry as attributes.

### QC

`fingerprints` copies `labels.names` into its data. `qc_fingerprints` and
`qc_phase_map` use those names when their `mineral_names` param is null;
the param stays as an override.

### Caching and regenerated files

`hdbscan_tiled` and `rare_phase` get a `recipe_revision` tag; the new
nodes have new recipes; everything downstream of `rare` changes hash.
Regenerate: the five shipped flows (`karak flow complete`),
`docs/flow.schema.json`, `docs/stage_reference.md`, the golden recipe
table. Docs: the user guide's `paper` paragraph, the README reproduction
sentence, CHANGELOG.

## Section 4: tests, acceptance and delivery

### Tests (synthetic, fast)

- Deferred-tile rule: a two-tile fixture, one deferred; `rare_phase`
  leaves its pixels at -1; `noise_assign` fills them; `TiledArtifacts`
  round trip with and without the new fields.
- `split_threshold` and `split_gmm`: parity with `_extract_olivine` and
  `_gmm_split` on the existing fixtures; `keep_parent` false empties the
  parent; `order_by` fixes label order on two separated components;
  `check_params` rejects a bad rule, an unknown channel, and a names
  string with a missing label.
- `split_hires`: a 4 x 4 working image with a 2x hires cube; one asserted
  pixel each for the hires map, the majority vote, the tie rule and the
  undefined-feature fallback; a non-integer ratio is a `StageError`.
- `name_phases`; `Labels.names` and `history` round trip; `export_h5`
  writes `mineral_names`, `subclustering`, `refinement_nodes` and
  `clusters/hires/labels`; `load_mineral_names` reads them back.
- Builtins: `paper` pins the 15 nodes and wiring; the other four flows
  pin unchanged nodes; the golden recipe table; schema and
  stage-reference drift tests.
- Validation: `check_params` errors appear in `karak validate` output.

### Real-data acceptance (NWA 4587, CPU, `--workers 0`)

- 16 named phases, labels 0 to 16, label 2 nearly empty.
- Each phase maps to one published phase by name. Abundances are
  reported against Table 1 of the paper; any shift over 1 pp is
  explained in the PR. Expected shifts: under 0.5 pp.
- `clusters/hires/labels` loads in a copy of `lamellae_analysis.py` with
  its own GMM fit removed; the grain count is within a few of the paper's
  128. This is the handshake with sub-project 2.
- The published `eds_pipeline.h5` is not modified.

### Delivery: one PR each, in this order

1. Deferred-tile rule and the recipe tags.
2. `Labels.names` and `history`, `name_phases`, `check_params`, export of
   names and history.
3. `split_threshold` and `split_gmm`; `refine` removed.
4. `split_hires`, `HiresLabels`, export of the hires map.
5. The `paper` flow, docs, the real-data acceptance report.

## Out of scope

The reproduction repo, figure scripts, TIMA comparison and lamellae
code (sub-project 2); paper text (3); release and deposit (4, 5); GPU
paths for the new stages (6).
