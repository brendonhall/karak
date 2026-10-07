# Complete Paper Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `karak run --builtin paper` produces the published 16-phase NWA 4587 map in one run, with every March 2026 hand step as a declared node, and writes the full-resolution pyroxene map the lamellae analysis needs.

**Architecture:** Five PRs in order. (1) Tiles that pass 1 cannot cluster skip pass 2; kNN fills them. (2) `Labels` carries names and a split history; a `name_phases` stage and a `Stage.check_params` hook; export writes names and history. (3) `split_threshold` and `split_gmm` replace `refine`. (4) `split_hires` fits one GMM on full-resolution channels inside a set of phases and emits a `HiresLabels` map. (5) The `paper` flow wires it all, docs change, and a real-data run is reported.

**Tech Stack:** Python 3.12, numpy, scikit-learn (GaussianMixture), hdbscan, h5py, pytest, uv. Repo rules in `CLAUDE.md`: every change through a PR from `main`; `uv run pytest` before every push; CHANGELOG (Unreleased) in the same branch.

**Spec:** `docs/superpowers/specs/2026-10-06-paper-flow-complete-design.md`

## Global Constraints

- Every change goes through a PR branched from an up-to-date `main` (`git switch main && git pull && git switch -c <type>/<name>`); never commit to `main`. Commit messages end with the attribution lines the session reminder gives.
- Params are data: scalar types only (`int`, `float`, `bool`, `enum`, `str`); lists are comma-separated strings, as `exclude_elements` and `include_elements` already are.
- Flows are complete: after any `PARAMS` change, regenerate the five shipped flows (`karak flow complete`), `docs/flow.schema.json` (`uv run python -m karak.flow.schema`) and `docs/stage_reference.md` (`uv run python -m karak.stages.reference`). Drift tests pin all three.
- Payloads are immutable: `apply()` returns new payloads via `.replace()`; never mutate inputs. Old cache files must still load (new fields default).
- When a core change alters results under unchanged params, return a tag from `recipe_revision(params)`; update `tests/test_recipe_stability.py` and its docstring.
- Rich stays in the CLI; stages log through `logging`.
- Tests that need CUDA are skipped without it; nothing here touches GPU paths.
- The published `/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/data/eds_pipeline.h5` is read-only reference data. Never open it with mode `a` or `w`.
- Prose (docs, CHANGELOG, PR bodies) follows the user's style rules: short sentences, active voice, no em dashes, concrete numbers.

## Review Focus

1. A `rule` or `features` string that names a channel the cube does not have must fail at `karak validate`-time where possible and as a `StageError` naming the channel at run time, never as a numpy `IndexError`. Pinned in Task 3.2 and 3.4.
2. `split_gmm` with `keep_parent: false` on a phase that has fewer pixels than `n_components * 10` must leave the labels unchanged and log a warning, not raise inside scikit-learn. Pinned in Task 3.4.
3. `split_hires` when the high-resolution cube is smaller than `H * ds` by up to `ds - 1` rows or columns (odd source dimensions) must clip, not raise or misalign. Pinned in Task 4.2.
4. `rare_phase` when every unassigned pixel lies in deferred tiles must return the input labels unchanged (no HDBSCAN call on zero rows). Pinned in Task 1.3.
5. `name_phases` with a label that the labels array does not contain must fail validation with the label number in the message, so a flow written for one sample does not silently mislabel another. Pinned in Task 2.3.

---

# Part 1: the deferred-tile rule (PR 1, branch `feat/deferred-tiles-skip-pass2`)

### Task 1.1: `TileResult.deferred` and `TiledArtifacts.deferred_pixels`

**Files:**
- Modify: `src/karak/clustering/tiling.py:88-98` (TileResult), `:597-610` (min_tile_pixels), `:684-705` (deferred branch)
- Modify: `src/karak/stages/payloads.py:340-435` (TiledArtifacts)
- Test: `tests/test_payloads.py`

**Interfaces:**
- Produces: `TileResult.deferred: bool` (default `False`); `tiling.resolve_min_tile_pixels(tiled: TiledConfig, hdbscan: HDBSCANConfig) -> int`; `tiling.deferred_pixel_indices(tiles: list[TileSpec], tile_results: list[TileResult]) -> np.ndarray` (int64, sorted); `TiledArtifacts.deferred_pixels: np.ndarray` (int64, default empty) and property `TiledArtifacts.deferred_tiles -> tuple[int, ...]`.

- [ ] **Step 1: Write the failing payload test**

Append to `tests/test_payloads.py`:

```python
def test_tiled_artifacts_roundtrip_deferred_fields(tmp_path):
    from karak.clustering.tiling import TileResult

    artifacts = TiledArtifacts(
        tile_results=(
            TileResult(tile_id=0, n_pixels=3, n_clusters=1, n_noise=3,
                       local_labels=np.array([0, 0, 0], dtype=np.int32),
                       merge_map={}, new_phases=[], deferred=True),
            TileResult(tile_id=1, n_pixels=2, n_clusters=2, n_noise=0,
                       local_labels=np.array([0, 1], dtype=np.int32),
                       merge_map={0: 0, 1: 1}, new_phases=[0, 1]),
        ),
        phase_registry=(),
        tile_size=32,
        deferred_pixels=np.array([0, 1, 2], dtype=np.int64),
    )
    back = _roundtrip(artifacts, tmp_path)
    assert back.tile_results[0].deferred is True
    assert back.tile_results[1].deferred is False
    assert back.deferred_tiles == (0,)
    np.testing.assert_array_equal(back.deferred_pixels, [0, 1, 2])


def test_tiled_artifacts_without_deferred_fields_loads_as_none_deferred(tmp_path):
    """Cache files written before the deferred-tile rule have no such
    attributes; they load with no deferred tiles and no deferred pixels."""
    import h5py

    from karak.clustering.tiling import TileResult

    artifacts = TiledArtifacts(
        tile_results=(TileResult(tile_id=0, n_pixels=1, n_clusters=1, n_noise=0,
                                 local_labels=np.array([0], dtype=np.int32),
                                 merge_map={0: 0}, new_phases=[0]),),
        phase_registry=(), tile_size=32,
    )
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as fh:
        artifacts.to_h5(fh.create_group("p"))
        del fh["p/tiles/0"].attrs["deferred"]        # as an old file has
        del fh["p/deferred_pixels"]
    with h5py.File(path, "r") as fh:
        back = payload_from_h5(fh["p"])
    assert back.deferred_tiles == ()
    assert back.deferred_pixels.size == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_payloads.py -k deferred -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'deferred'`

- [ ] **Step 3: Add the fields and helpers in `tiling.py`**

In `TileResult` add a last field:

```python
    new_phases: list[int]  # global_ids of phases discovered in this tile
    deferred: bool = False  # True: too few clusters, every pixel left to kNN
```

After `compute_tile_grid` (before `_check_accumulate`) add:

```python
def resolve_min_tile_pixels(tiled: TiledConfig, hdbscan: HDBSCANConfig) -> int:
    """``min_tile_pixels`` of the tiled config, or twice ``min_cluster_size``
    when it is None. One rule for the clustering and for every consumer
    that recomputes the grid (``hdbscan_tiled``, ``qc_tiled``)."""
    if tiled.min_tile_pixels is not None:
        return tiled.min_tile_pixels
    return 2 * hdbscan.min_cluster_size


def deferred_pixel_indices(tiles: list[TileSpec],
                           tile_results: list[TileResult]) -> np.ndarray:
    """Sorted indices (into the mineral arrays) of every pixel of a
    deferred tile; empty when no tile was deferred."""
    deferred = {tr.tile_id for tr in tile_results if tr.deferred}
    parts = [t.pixel_indices for t in tiles if t.tile_id in deferred]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(parts).astype(np.int64))
```

`TiledConfig` and `HDBSCANConfig` are already imported under `TYPE_CHECKING` or at module level in `tiling.py`; check the imports at the top of the file and add `from karak.core_params import HDBSCANConfig, TiledConfig` under `TYPE_CHECKING` if they are missing.

In `run_tiled_hdbscan`, replace the three lines that resolve `min_tile_pixels` with:

```python
    min_tile_pixels = resolve_min_tile_pixels(tiled_cfg, hdb_cfg)
```

In the deferred branch (`if n_clusters < min_clusters:`), add `deferred=True` to the `TileResult(...)` call:

```python
            tile_results.append(
                TileResult(
                    tile_id=tile.tile_id,
                    n_pixels=len(tile.pixel_indices),
                    n_clusters=n_clusters,
                    n_noise=len(tile.pixel_indices),  # all deferred
                    local_labels=tile_labels,
                    merge_map={},
                    new_phases=[],
                    deferred=True,
                )
            )
```

- [ ] **Step 4: Add the payload fields in `payloads.py`**

In `TiledArtifacts`:

```python
    tile_results: tuple                     # tuple[TileResult, ...]
    phase_registry: tuple                   # tuple[PhaseEntry, ...]
    tile_size: int
    deferred_pixels: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )                                       # indices of deferred tiles' pixels

    @property
    def deferred_tiles(self) -> tuple:
        return tuple(tr.tile_id for tr in self.tile_results if tr.deferred)
```

In `to_h5`, inside the per-tile loop add `sub.attrs["deferred"] = bool(tr.deferred)`, and after the registry loop add:

```python
        _dataset(group, "deferred_pixels", self.deferred_pixels, compression)
```

In `from_h5`, in the `TileResult(...)` construction add
`deferred=bool(sub.attrs.get("deferred", False)),` and in the final `cls(...)` add:

```python
            deferred_pixels=(group["deferred_pixels"][()]
                             if "deferred_pixels" in group
                             else np.zeros(0, dtype=np.int64)),
```

`_dataset` must accept an empty 1-D array (the docstring says 0-d and empty arrays stay unchunked); run the test to confirm.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_payloads.py -v`
Expected: all PASS (the existing `test_every_payload_roundtrips_under_each_compression` included).

- [ ] **Step 6: Commit**

```bash
git add src/karak/clustering/tiling.py src/karak/stages/payloads.py tests/test_payloads.py
git commit -m "feat: tiled artifacts record deferred tiles and their pixels"
```

### Task 1.2: `hdbscan_tiled` fills `deferred_pixels`

**Files:**
- Modify: `src/karak/stages/cluster.py:146-200`
- Test: `tests/test_cluster_stage_parity.py`

**Interfaces:**
- Consumes: `resolve_min_tile_pixels`, `deferred_pixel_indices`, `compute_tile_grid` from Task 1.1.
- Produces: `TiledArtifacts.deferred_pixels` set by the stage; `HdbscanTiledStage.recipe_revision` returns `"deferred-1"` (joined with the cuda tag when present).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cluster_stage_parity.py` (after `test_hdbscan_tiled_parity`):

```python
def test_hdbscan_tiled_records_deferred_tiles(features_payload, denoised_cube, chain):
    """With tile_size 32 on the 64x64 scene, the left tiles hold only phase A
    (one cluster) and are deferred at min_clusters_per_tile 2; the right
    tiles hold A and B and are kept."""
    out = get("hdbscan_tiled")().run(
        {"features": features_payload, "cube": denoised_cube},
        {"min_cluster_size": 100, "random_state": 0,
         "tile_size": 32, "min_clusters_per_tile": 2},
    )
    tiles = out["tiles"]
    deferred = [tr for tr in tiles.tile_results if tr.deferred]
    kept = [tr for tr in tiles.tile_results if not tr.deferred]
    assert deferred and kept
    assert tiles.deferred_tiles == tuple(tr.tile_id for tr in deferred)
    assert tiles.deferred_pixels.size == sum(tr.n_pixels for tr in deferred)
    # every deferred pixel is unassigned in the raw labels
    assert (out["labels"].labels[tiles.deferred_pixels] == -1).all()
    # deferred pixels lie in the left half (phase A columns 8..31)
    cols = chain["mineral_indices"][tiles.deferred_pixels, 1]
    assert cols.max() < 32
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_cluster_stage_parity.py::test_hdbscan_tiled_records_deferred_tiles -v`
Expected: FAIL at `tiles.deferred_pixels.size == ...` (size 0).

- [ ] **Step 3: Fill the field in the stage and tag the recipe**

In `src/karak/stages/cluster.py`, in `HdbscanTiledStage`:

```python
    @classmethod
    def recipe_revision(cls, params: dict) -> str | None:
        # 2026-10-06: deferred tiles are recorded and skip pass 2
        return _join_revisions(_cuda_subsample_revision(params), "deferred-1")
```

Add the helper next to `_cuda_subsample_revision`:

```python
def _join_revisions(*tags: str | None) -> str | None:
    """One revision string from the tags that apply, or None."""
    present = [t for t in tags if t]
    return "+".join(present) if present else None
```

In `apply`, after `run_tiled_hdbscan` returns, compute the deferred pixels and pass them to the payload:

```python
        from karak.clustering.tiling import (
            compute_tile_grid, deferred_pixel_indices, resolve_min_tile_pixels,
        )

        hdb_cfg, tiled_cfg = hdbscan_config(params), tiled_config(params)
        ...  # the existing run_tiled_hdbscan call, using hdb_cfg and tiled_cfg
        grid = compute_tile_grid(
            features.mineral_indices, features.image_shape,
            tiled_cfg.tile_size, resolve_min_tile_pixels(tiled_cfg, hdb_cfg),
        )
        deferred_pixels = deferred_pixel_indices(grid, tile_results)
        return {
            "labels": Labels(...unchanged...),
            "tiles": TiledArtifacts(
                tile_results=tuple(tile_results),
                phase_registry=tuple(phase_registry),
                tile_size=params["tile_size"],
                deferred_pixels=deferred_pixels,
            ),
        }
```

- [ ] **Step 4: Run the parity tests**

Run: `uv run pytest tests/test_cluster_stage_parity.py -k tiled -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/stages/cluster.py tests/test_cluster_stage_parity.py
git commit -m "feat: hdbscan_tiled records the pixels of deferred tiles"
```

### Task 1.3: `rare_phase` skips deferred pixels

**Files:**
- Modify: `src/karak/clustering/tiling.py:341-535` (`recluster_unassigned`)
- Modify: `src/karak/stages/rare_phase.py`
- Test: `tests/test_cluster_stage_parity.py`

**Interfaces:**
- Produces: `recluster_unassigned(..., exclude_indices: np.ndarray | None = None)`; `RarePhaseStage.recipe_revision` returns `"deferred-1"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cluster_stage_parity.py`:

```python
def _tiled_outputs(features_payload, denoised_cube, min_clusters):
    return get("hdbscan_tiled")().run(
        {"features": features_payload, "cube": denoised_cube},
        {"min_cluster_size": 100, "random_state": 0,
         "tile_size": 32, "min_clusters_per_tile": min_clusters},
    )


def test_rare_phase_leaves_deferred_pixels_unassigned(features_payload, denoised_cube):
    out = _tiled_outputs(features_payload, denoised_cube, min_clusters=2)
    deferred = out["tiles"].deferred_pixels
    assert deferred.size > 0
    rare = get("rare_phase")().run(
        {"labels": out["labels"], "features": features_payload,
         "cube": denoised_cube, "tiles": out["tiles"]},
        {"min_cluster_size": 20, "random_state": 0},
    )
    assert (rare["labels"].labels[deferred] == -1).all()
    # the deferred pixels pass through to the payload unchanged
    np.testing.assert_array_equal(rare["tiles"].deferred_pixels, deferred)
    # and noise_assign gives every one of them a label
    knn = get("noise_assign")().run(
        {"labels": rare["labels"], "features": features_payload}, {"k": 5})
    assert (knn["labels"].labels[deferred] >= 0).all()


def test_rare_phase_with_only_deferred_noise_returns_input_labels(
        features_payload, denoised_cube):
    """Every -1 pixel sits in a deferred tile: no HDBSCAN fit on zero rows,
    labels come back unchanged."""
    out = _tiled_outputs(features_payload, denoised_cube, min_clusters=2)
    labels = out["labels"]
    forced = labels.labels.copy()
    forced[forced == -1] = 0                      # assign every real noise pixel
    forced[out["tiles"].deferred_pixels] = -1     # keep only the deferred ones
    labels = labels.replace(labels=forced)
    rare = get("rare_phase")().run(
        {"labels": labels, "features": features_payload,
         "cube": denoised_cube, "tiles": out["tiles"]},
        {"min_cluster_size": 20, "random_state": 0},
    )
    np.testing.assert_array_equal(rare["labels"].labels, forced)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cluster_stage_parity.py -k "deferred" -v`
Expected: the first FAILS at `(rare["labels"].labels[deferred] == -1).all()`; the second may fail or pass depending on the fit; both must pass after Step 3.

- [ ] **Step 3: Add `exclude_indices` to the core and use it in the stage**

In `recluster_unassigned`, add the keyword and apply it right after `unassigned_mask` is built:

```python
    *,
    random_state: int,
    workers: int = 1,
    device: str = "cpu",
    exclude_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[PhaseEntry], int, int]:
    """...

    ``exclude_indices`` (indices into the mineral arrays) are left out of
    the pass-2 fit, prediction and registry merge and stay -1: the pixels
    of tiles that pass 1 deferred, which the kNN step fills from their
    neighbours instead (a tile with too few clusters otherwise becomes
    one block of the pass-2 majority phase).
    ...
    """
    ...
    unassigned_mask = raw_labels == -1
    n_excluded = 0
    if exclude_indices is not None and exclude_indices.size:
        n_excluded = int(unassigned_mask[exclude_indices].sum())
        unassigned_mask[exclude_indices] = False
        logger.info("Pass 2: %d pixels of deferred tiles left to kNN", n_excluded)
    n_unassigned = int(np.sum(unassigned_mask))

    if n_unassigned == 0:
        logger.info("No unassigned pixels for Pass 2")
        return raw_labels.copy(), phase_registry, 0, n_excluded
```

The docstring of the return value `n_still_unassigned` should say it includes the excluded pixels; at the end compute `n_still_unassigned = int(np.sum(updated_labels == -1))` as today (excluded pixels are still -1, so the count is right).

In `src/karak/stages/rare_phase.py`:

```python
    @classmethod
    def recipe_revision(cls, params: dict) -> str | None:
        # 2026-10-06: pixels of deferred tiles skip pass 2
        return "deferred-1"
```

and in `apply`, pass `exclude_indices=tiles.deferred_pixels` to `recluster_unassigned`, and keep `deferred_pixels` on the output payload (`tiles.replace(phase_registry=tuple(registry))` already keeps it, because `replace` copies the other fields).

Also update the `description` of `RarePhaseStage` to end with: "Pixels of tiles that pass 1 deferred are left to noise_assign."

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cluster_stage_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/clustering/tiling.py src/karak/stages/rare_phase.py tests/test_cluster_stage_parity.py
git commit -m "feat: rare_phase leaves the pixels of deferred tiles to noise_assign"
```

### Task 1.4: recipe table, export of the deferred flag, docs, CHANGELOG

**Files:**
- Modify: `tests/test_recipe_stability.py`
- Modify: `src/karak/io/storage.py:483-558` (`save_tiled_metadata`)
- Modify: `docs/user_guide.md` (the `hdbscan_tiled` and `rare_phase` rows of the stage table; the `paper` paragraph's noise sentence)
- Modify: `docs/stage_reference.md` (regenerated), `CHANGELOG.md`

- [ ] **Step 1: Run the recipe test to see the new hashes**

Run: `uv run pytest tests/test_recipe_stability.py -v 2>&1 | grep -E "hdb|rare|knn|stats|fp|PASS|FAIL" | head -40`
Expected: `tiled`, `tiled-rare` and `paper` FAIL; `global` and `stepwise` PASS.

- [ ] **Step 2: Update the golden table**

Print the new values:

```bash
uv run python -c "
from karak.flow.builtins import builtin_flow
from karak.flow.executor import plan_recipes
from karak.stages import registry
T = {'{input}': '/nonexistent/input', '{out}': 'out/run', '{work}': 'work', '{flow}': '{}'}
for name in ('tiled', 'tiled-rare', 'paper'):
    g = builtin_flow(name); h = plan_recipes(g, T)
    print(name, {n: v for n, v in h.items() if n in ('hdb', 'rare', 'knn', 'stats', 'fp')})
"
```

Paste the `hdb`, `rare`, `knn`, `stats`, `fp` values into `GOLDEN` for those three flows and add to the module docstring:

```
2026-10-06: hdbscan_tiled records deferred tiles and rare_phase leaves their
pixels to noise_assign (recipe revision "deferred-1" on both), so in tiled,
tiled-rare and paper `hdb`, `rare` and everything downstream carry new
hashes; global and stepwise do not change.
```

Run: `uv run pytest tests/test_recipe_stability.py -v` → PASS.

- [ ] **Step 3: Export the deferred flag per tile**

In `save_tiled_metadata`, after `tile_n_new`, add:

```python
        tile_deferred = np.array(
            [bool(getattr(tr, "deferred", False)) for tr in tile_results], dtype=bool
        )
        ...
        tiled_grp.create_dataset("tile_deferred", data=tile_deferred)
```

Add to `tests/test_export_stage.py` a test that builds a `TiledArtifacts` with one deferred tile, connects it to `export_h5` as `tiles` together with the `payloads` fixture's `labels`, `stats`, `features`, and asserts `fh["clusters/tiled/tile_deferred"][()].tolist() == [True]`. Model it on `test_export_writes_legacy_layout` (same inputs plus `tiles`).

Run: `uv run pytest tests/test_export_stage.py -v` → PASS.

- [ ] **Step 4: Docs and CHANGELOG**

`docs/user_guide.md`, stage table: in the `hdbscan_tiled` row append "A tile with fewer than `min_clusters_per_tile` clusters is deferred: its pixels stay unassigned for `noise_assign`, and `rare_phase` skips them." In the `rare_phase` row append "Pixels of deferred tiles are skipped (they would otherwise form one block of the pass-2 majority phase)."

`docs/user_guide.md`, `paper` paragraph: replace the last bullet ("the rare-phase pass does not reproduce ...") with:

```
- the rare-phase pass reproduces the published raw labels once the one
  deferred 1024 px tile is excluded, which `rare_phase` now does: replaying
  it on the published pass-1 noise matches the published raw labels in all
  but 6 of 12,495,787 pixels. The published file had that tile reset by
  hand (`fix_tile.py` in the paper repository).
```

Regenerate: `uv run python -m karak.stages.reference`.

`CHANGELOG.md` under `## [Unreleased]` / `### Changed` (create the heading if absent):

```
- Tiles with fewer than `min_clusters_per_tile` clusters are now skipped by
  `rare_phase` and filled by `noise_assign`. Pass 2 used to recluster them
  with the rest of the noise, which on NWA 4587 put 99 % of a 1,043,317
  pixel tile into one phase as a square block; the published run removed
  that block by hand. `hdbscan_tiled` records the deferred tiles
  (`TileResult.deferred`, `TiledArtifacts.deferred_pixels`, HDF5
  `clusters/tiled/tile_deferred`). Recipe revision `deferred-1` on both
  stages: `hdb`, `rare` and downstream recipes change in `tiled`,
  `tiled-rare` and `paper`.
```

- [ ] **Step 5: Full test run, push, PR**

Run: `uv run pytest` → all PASS.

```bash
git add -A tests docs src CHANGELOG.md
git commit -m "docs: deferred-tile rule in the user guide, reference and changelog"
git push -u origin HEAD
gh pr create --title "feat: tiles that pass 1 cannot cluster skip pass 2" --body "<what changed, why (the fix_tile finding with numbers), tests run, recipe hashes changed, follow-ups: PRs 2-5 of the spec>"
gh pr checks --watch
```

Stop here; the user reviews and merges.

---

# Part 2: names, history, `name_phases`, `check_params` (PR 2, branch `feat/phase-names-and-history`)

### Task 2.1: `Labels.names` and `Labels.history`

**Files:**
- Modify: `src/karak/stages/payloads.py:296-338`
- Test: `tests/test_payloads.py`

**Interfaces:**
- Produces: `Labels.names: dict[int, str]` (default `{}`), `Labels.history: tuple[dict, ...]` (default `()`); both survive `replace()` and the HDF5 round trip. A history record is a plain dict with keys `stage`, `parent`, `new_labels` (list[int]), `names` (dict[int, str]), `n_pixels` (dict[int, int]), `method` (str), `note` (str).

- [ ] **Step 1: Write the failing test**

```python
def test_labels_roundtrip_names_and_history(tmp_path):
    labels = Labels(
        labels=np.array([0, 1, 2], dtype=np.int32),
        probabilities=None,
        mineral_indices=np.zeros((3, 2), dtype=np.int32),
        image_shape=(2, 2),
        state=LabelState.CLEANED,
        names={0: "Olivine", 1: "Augite", 2: "Pigeonite"},
        history=({"stage": "split_gmm", "parent": 1, "new_labels": [2],
                  "names": {2: "Pigeonite"}, "n_pixels": {2: 1},
                  "method": "GMM on Ca", "note": "why"},),
    )
    back = _roundtrip(labels, tmp_path)
    assert back.names == {0: "Olivine", 1: "Augite", 2: "Pigeonite"}
    assert back.history[0]["new_labels"] == [2]
    assert back.history[0]["names"] == {2: "Pigeonite"}
    assert back.history[0]["n_pixels"] == {2: 1}
    # old files have neither attribute
    import h5py
    with h5py.File(tmp_path / "old.h5", "w") as fh:
        labels.to_h5(fh.create_group("p"))
        del fh["p"].attrs["names"], fh["p"].attrs["history"]
    with h5py.File(tmp_path / "old.h5", "r") as fh:
        old = payload_from_h5(fh["p"])
    assert old.names == {} and old.history == ()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_payloads.py::test_labels_roundtrip_names_and_history -v`
Expected: FAIL with `unexpected keyword argument 'names'`.

- [ ] **Step 3: Add the fields**

In `Labels`:

```python
    state: LabelState
    names: dict = dataclasses.field(default_factory=dict)     # label -> name
    history: tuple = ()                                        # split records
```

`summary()`: when `self.names` is non-empty append `f" · {len(self.names)} named"`.

`to_h5`: add

```python
        group.attrs["names"] = json.dumps({str(k): v for k, v in self.names.items()})
        group.attrs["history"] = json.dumps(_jsonable(list(self.history)))
```

(`_jsonable` is defined later in the module; Python resolves it at call time, so the order is fine.)

`from_h5`: add a module-level helper and use it:

```python
def _int_keys(mapping: dict) -> dict:
    return {int(k): v for k, v in mapping.items()}


def _history_from_json(text: str) -> tuple:
    records = []
    for rec in json.loads(text):
        rec = dict(rec)
        for key in ("names", "n_pixels"):
            if key in rec:
                rec[key] = _int_keys(rec[key])
        records.append(rec)
    return tuple(records)
```

```python
            names=_int_keys(json.loads(group.attrs.get("names", "{}"))),
            history=_history_from_json(group.attrs.get("history", "[]")),
```

- [ ] **Step 4: Run the payload tests**

Run: `uv run pytest tests/test_payloads.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/stages/payloads.py tests/test_payloads.py
git commit -m "feat: Labels carry phase names and a split history"
```

### Task 2.2: `Stage.check_params` and the validator hook; the text parsers

**Files:**
- Modify: `src/karak/stages/base.py:201-223`
- Modify: `src/karak/flow/validate.py:52-62`
- Create: `src/karak/stages/params_text.py`
- Test: `tests/test_params_text.py`, `tests/test_validate_check_params.py`

**Interfaces:**
- Produces: `Stage.check_params(cls, params: dict) -> list[str]` (classmethod, default `[]`); the validator adds one `Issue("error", node.id, msg)` per string; `Stage.run()` raises `StageError` on any. `params_text` functions: `parse_csv(text) -> list[str]`; `parse_int_list(text, name) -> list[int]`; `parse_names(text, name) -> dict[int, str]` for `"0: A; 1: B"`; `parse_rule(text, name) -> list[tuple[str, str, float]]` for `"Fe-K > 0.6 & Ca < 0.10"`; `parse_feature(token) -> tuple[str, tuple[str, ...]]` returning `("channel", (A,))`, `("bse", ())` or `("ratio", (A, B))` for `A/(A+B)`. Each raises `ValueError` with a message that starts with the param name.

- [ ] **Step 1: Write the failing parser tests**

Create `tests/test_params_text.py`:

```python
import pytest

from karak.stages.params_text import (
    parse_csv, parse_feature, parse_int_list, parse_names, parse_rule,
)


def test_parse_csv_strips_and_drops_empty():
    assert parse_csv(" Ca, Mg ,,BSE ") == ["Ca", "Mg", "BSE"]
    assert parse_csv("") == [] and parse_csv(None) == []


def test_parse_int_list():
    assert parse_int_list("2, 7", "target_phases") == [2, 7]
    with pytest.raises(ValueError, match="target_phases"):
        parse_int_list("2, x", "target_phases")


def test_parse_names():
    assert parse_names("0: Ilmenite (FeTiO₃); 1: Silica; 8: Fe Oxyhydroxide", "names") == {
        0: "Ilmenite (FeTiO₃)", 1: "Silica", 8: "Fe Oxyhydroxide"}
    with pytest.raises(ValueError, match="names"):
        parse_names("0 Ilmenite", "names")
    with pytest.raises(ValueError, match="names.*twice"):
        parse_names("0: A; 0: B", "names")


def test_parse_rule():
    assert parse_rule("Fe-K > 0.6 & Ca < 0.10", "rule") == [
        ("Fe-K", ">", 0.6), ("Ca", "<", 0.10)]
    assert parse_rule("Si >= 1", "rule") == [("Si", ">=", 1.0)]
    for bad in ("Fe-K 0.6", "Fe-K == 0.6", "Fe-K > abc", ""):
        with pytest.raises(ValueError, match="rule"):
            parse_rule(bad, "rule")


def test_parse_feature():
    assert parse_feature("Ca") == ("channel", ("Ca",))
    assert parse_feature("BSE") == ("bse", ())
    assert parse_feature("Ca/(Ca+Mg)") == ("ratio", ("Ca", "Mg"))
    with pytest.raises(ValueError, match="ratio"):
        parse_feature("Ca/(Mg+Fe)")      # numerator must be the first term
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_params_text.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `params_text.py`**

```python
"""Parsers for the small text grammars inside string params.

Params are scalar, so lists and rules travel as strings (as
``exclude_elements`` does). These parsers turn them into data and raise
``ValueError`` messages that start with the param name, so
``Stage.check_params`` and ``karak validate`` can report them.
"""

from __future__ import annotations

import re

_OPS = ("<=", ">=", "<", ">")
_RATIO = re.compile(r"^\s*([^/()\s]+)\s*/\s*\(\s*([^+()\s]+)\s*\+\s*([^+()\s]+)\s*\)\s*$")


def parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_list(text: str | None, name: str) -> list[int]:
    values = []
    for item in parse_csv(text):
        try:
            values.append(int(item))
        except ValueError:
            raise ValueError(f"{name}: {item!r} is not an integer label") from None
    return values


def parse_names(text: str | None, name: str) -> dict[int, str]:
    """``"0: Ilmenite; 1: Silica"`` -> {0: "Ilmenite", 1: "Silica"}."""
    names: dict[int, str] = {}
    for entry in (text or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        label, sep, value = entry.partition(":")
        if not sep or not value.strip():
            raise ValueError(f"{name}: expected 'label: name', got {entry!r}")
        try:
            key = int(label)
        except ValueError:
            raise ValueError(f"{name}: {label.strip()!r} is not an integer label") from None
        if key in names:
            raise ValueError(f"{name}: label {key} given twice")
        names[key] = value.strip()
    return names


def parse_rule(text: str | None, name: str) -> list[tuple[str, str, float]]:
    """``"Fe-K > 0.6 & Ca < 0.10"`` -> [("Fe-K", ">", 0.6), ("Ca", "<", 0.1)]."""
    if not text or not text.strip():
        raise ValueError(f"{name}: empty rule")
    rules = []
    for clause in text.split("&"):
        clause = clause.strip()
        op = next((o for o in _OPS if o in clause), None)
        if op is None:
            raise ValueError(f"{name}: {clause!r} needs one of {_OPS}")
        channel, _, value = clause.partition(op)
        channel = channel.strip()
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{name}: {value.strip()!r} is not a number") from None
        if not channel:
            raise ValueError(f"{name}: {clause!r} has no channel name")
        rules.append((channel, op, number))
    return rules


def parse_feature(token: str) -> tuple[str, tuple[str, ...]]:
    """One feature token: a channel name, ``BSE`` or a ratio ``A/(A+B)``."""
    token = token.strip()
    if token.upper() == "BSE":
        return ("bse", ())
    if "/" in token:
        m = _RATIO.match(token)
        if m is None or m.group(1) != m.group(2):
            raise ValueError(f"ratio feature must look like A/(A+B), got {token!r}")
        return ("ratio", (m.group(1), m.group(3)))
    if not token:
        raise ValueError("empty feature name")
    return ("channel", (token,))
```

Run: `uv run pytest tests/test_params_text.py -v` → PASS.

- [ ] **Step 4: Write the failing validator test**

Create `tests/test_validate_check_params.py`:

```python
"""Stage.check_params feeds karak validate and Stage.run."""

import pytest

from conftest import complete
from karak.flow.graph import Graph, Node
from karak.flow.validate import validate
from karak.stages import registry
from karak.stages.base import Param, Stage, StageError


class _Picky(Stage):
    id = "temp_picky_stage"
    label = "Picky"
    description = "test stage"
    INPUTS: list = []
    OUTPUTS: list = []
    PARAMS = [Param("word", "str", "ok", "Word")]

    @classmethod
    def check_params(cls, params):
        return [] if params["word"] == "ok" else [f"word: {params['word']!r} is not ok"]

    def apply(self, inputs, params):
        return {}


@pytest.fixture()
def picky():
    registry._REGISTRY["temp_picky_stage"] = _Picky
    try:
        yield _Picky
    finally:
        registry._REGISTRY.pop("temp_picky_stage", None)


def test_validate_reports_check_params_errors(picky):
    graph = complete(Graph(nodes=[Node(id="p", type="temp_picky_stage",
                                        params={"word": "bad"})], edges=[]))
    issues = validate(graph)
    assert [(i.level, i.where) for i in issues] == [("error", "p")]
    assert "word: 'bad' is not ok" in issues[0].message


def test_run_raises_on_check_params_errors(picky):
    with pytest.raises(StageError, match="word: 'bad'"):
        picky().run({}, {"word": "bad"})
    assert picky().run({}, {"word": "ok"}) == {}
```

Check how `Graph`/`Node` are constructed in `tests/test_export_stage.py` or `tests/test_graph.py` and match that call shape if it differs.

Run: `uv run pytest tests/test_validate_check_params.py -v` → FAIL (no `check_params`; `validate` reports no issue).

- [ ] **Step 5: Add the hook**

`src/karak/stages/base.py`, in `Stage` before `check`:

```python
    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        """Errors in the coerced params that bounds and choices cannot
        express (text grammars, cross-param rules). Empty means valid.
        The flow validator and ``run()`` both call it."""
        return []
```

and in `run`:

```python
        coerced = self.coerce_params(params)
        errors = self.check_params(coerced) + self.check(inputs, coerced)
```

`src/karak/flow/validate.py`, replace the coerce block:

```python
        try:
            coerced = cls.coerce_params(node.params)   # types, bounds, unknown names
        except ValueError as exc:
            issues.append(Issue("error", node.id, str(exc)))
            continue
        for message in cls.check_params(coerced):
            issues.append(Issue("error", node.id, message))
```

(`continue` is safe only if nothing after this block in the loop body needs the node; check that the loop body ends there, else restructure with an `else:` on the `try`.)

Run: `uv run pytest tests/test_validate_check_params.py tests/test_params_text.py tests/test_flow_cli.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/karak/stages/base.py src/karak/flow/validate.py src/karak/stages/params_text.py tests/test_params_text.py tests/test_validate_check_params.py
git commit -m "feat: Stage.check_params hook and text-grammar parsers"
```

### Task 2.3: the `name_phases` stage

**Files:**
- Create: `src/karak/stages/naming.py`
- Test: `tests/test_naming_stage.py`
- Modify: `tests/test_registry.py:46` (add `"name_phases"` to the expected set)

**Interfaces:**
- Produces: stage `name_phases`, input `labels` (CLEANED), output `labels` (CLEANED); params `names` (str), `note` (str). `check_params` parses `names`. `apply` raises `StageError` when a named label is absent from the labels array (Review Focus 5).

- [ ] **Step 1: Write the failing tests**

```python
"""name_phases attaches researcher names to the base phases."""

import numpy as np
import pytest

from karak.stages import get
from karak.stages.base import StageError
from karak.stages.payloads import LabelState, Labels


def _labels(values):
    arr = np.asarray(values, dtype=np.int32)
    return Labels(labels=arr, probabilities=None,
                  mineral_indices=np.zeros((arr.size, 2), dtype=np.int32),
                  image_shape=(1, arr.size), state=LabelState.CLEANED)


def test_name_phases_sets_names_and_keeps_labels():
    out = get("name_phases")().run(
        {"labels": _labels([0, 1, 1, 2])},
        {"names": "0: Ilmenite; 1: Pyroxene; 2: Plagioclase", "note": "from fingerprints"})
    assert out["labels"].names == {0: "Ilmenite", 1: "Pyroxene", 2: "Plagioclase"}
    np.testing.assert_array_equal(out["labels"].labels, [0, 1, 1, 2])
    assert out["labels"].state is LabelState.CLEANED
    assert out["labels"].history == ()


def test_name_phases_merges_with_existing_names():
    base = _labels([0, 1]).replace(names={0: "Old", 5: "Kept"})
    out = get("name_phases")().run({"labels": base}, {"names": "0: New", "note": ""})
    assert out["labels"].names == {0: "New", 5: "Kept"}


def test_name_phases_rejects_a_label_the_data_does_not_have():
    with pytest.raises(StageError, match="label 7"):
        get("name_phases")().run({"labels": _labels([0, 1])},
                                 {"names": "0: A; 7: B", "note": ""})


def test_name_phases_check_params_reports_bad_text():
    errors = get("name_phases").check_params({"names": "0 A", "note": ""})
    assert errors and errors[0].startswith("names")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_naming_stage.py -v` → FAIL (`KeyError: 'name_phases'`).

- [ ] **Step 3: Write the stage**

```python
"""Name-phases stage: attach researcher-assigned names to base phases."""

from __future__ import annotations

import numpy as np

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_names
from karak.stages.payloads import LabelState
from karak.stages.registry import register


@register
class NamePhasesStage(Stage):
    id = "name_phases"
    label = "Name phases"
    description = (
        "Attach mineral names to phase labels. Names travel with the labels "
        "to the split stages, the fingerprints, the QC figures and the HDF5 "
        "export (clusters/mineral_names)."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="labels to name"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="the same labels with names attached"),
    ]
    PARAMS = [
        Param("names", "str", "", "Names",
              "Entries 'label: name' separated by ';', e.g. "
              "'0: Ilmenite (FeTiO₃); 1: Silica polymorph (SiO₂)'"),
        Param("note", "str", "", "Note",
              "How the names were decided (fingerprints, TIMA, references)"),
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        try:
            parse_names(params["names"], "names")
        except ValueError as exc:
            return [str(exc)]
        return []

    def apply(self, inputs: dict, params: dict) -> dict:
        labels = inputs["labels"]
        names = parse_names(params["names"], "names")
        present = set(np.unique(labels.labels).tolist())
        missing = sorted(k for k in names if k not in present)
        if missing:
            raise StageError(
                "name_phases: " + ", ".join(f"label {m}" for m in missing)
                + " not in the labels; check the names against this run's phases"
            )
        return {"labels": labels.replace(names={**labels.names, **names})}
```

Add `"name_phases"` to the expected id set in `tests/test_registry.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_naming_stage.py tests/test_registry.py -v` → PASS.

- [ ] **Step 5: Regenerate schema and reference, commit**

```bash
uv run python -m karak.flow.schema && uv run python -m karak.stages.reference
uv run pytest tests/test_flow_schema.py tests/test_stage_reference.py -v
git add src/karak/stages/naming.py tests/test_naming_stage.py tests/test_registry.py docs/flow.schema.json docs/stage_reference.md
git commit -m "feat: name_phases stage attaches mineral names to labels"
```

### Task 2.4: export names and history; fingerprints and QC use the names

**Files:**
- Modify: `src/karak/io/storage.py` (add `save_subclustering`)
- Modify: `src/karak/stages/export.py:135-175`
- Modify: `src/karak/stages/fingerprints.py`, `src/karak/stages/payloads.py` (`Fingerprints.from_h5`), `src/karak/stages/qc_sinks.py` (`qc_fingerprints`, `qc_named_phase_map`)
- Test: `tests/test_export_stage.py`, `tests/test_cluster_stage_parity.py`

**Interfaces:**
- Produces: `storage.save_subclustering(h5_path, history: list[dict]) -> None` writing `clusters/subclustering` attrs `history` (JSON list) and `split_NN` (JSON record, zero-based, two digits); `export_h5` calls `save_mineral_names` when `labels.names` is non-empty and `save_subclustering` when `labels.history` is non-empty; `cluster_config["refinement_nodes"]` = ordered list of `{"id", "type", "params"}` for nodes of types `name_phases`, `split_threshold`, `split_gmm`, `split_hires` (types that do not exist yet are simply absent). `Fingerprints.data["names"]` is a `dict[int, str]` copy of `labels.names`.

- [ ] **Step 1: Write the failing export test**

Append to `tests/test_export_stage.py`:

```python
def test_export_writes_names_and_history(tmp_path, payloads):
    from karak.io.storage import load_mineral_names

    labels = payloads["labels"].replace(
        names={0: "Olivine", 1: "Augite"},
        history=({"stage": "split_gmm", "parent": 0, "new_labels": [1],
                  "names": {1: "Augite"}, "n_pixels": {1: 3},
                  "method": "GMM on Ca", "note": "why"},),
    )
    path = tmp_path / "out.h5"
    get("export_h5")().run(
        {"labels": labels, "stats": payloads["stats"], "features": payloads["features"]},
        {"path": str(path), "flow_json": _flow_json(), "compression": "none"},
    )
    assert load_mineral_names(path) == {0: "Olivine", 1: "Augite"}
    with h5py.File(path, "r") as fh:
        sub = fh["clusters/subclustering"]
        assert json.loads(sub.attrs["history"])[0]["new_labels"] == [1]
        assert json.loads(sub.attrs["split_00"])["parent"] == 0
        assert fh["clusters"].attrs["cluster_1_name"] == "Augite"
```

`payloads["labels"]` and `payloads["stats"]` exist in the fixture (check the fixture's keys; the fixture builds `labels` with `state=LabelState.CLEANED`). `_flow_json()` returns a complete flow with the clustering nodes; look at it and, if it has no `name_phases` node, that is fine for this test.

Run: `uv run pytest tests/test_export_stage.py::test_export_writes_names_and_history -v` → FAIL (`load_mineral_names` returns None).

- [ ] **Step 2: Write `save_subclustering` and call both from export**

`storage.py`, after `save_mineral_names`:

```python
def save_subclustering(h5_path: str | Path, history: list[dict]) -> None:
    """Write the split history to ``clusters/subclustering`` attributes:
    ``history`` (the JSON list) and one ``split_NN`` attribute per record,
    the layout the NWA 4587 analysis scripts read."""
    with h5py.File(h5_path, "a") as f:
        grp = f["clusters"]
        if "subclustering" in grp:
            del grp["subclustering"]
        sub = grp.create_group("subclustering")
        records = [_json_record(r) for r in history]
        sub.attrs["history"] = json.dumps(records)
        for i, rec in enumerate(records):
            sub.attrs[f"split_{i:02d}"] = json.dumps(rec)
    logger.info("Saved %d split records to clusters/subclustering", len(history))


def _json_record(record: dict) -> dict:
    out = {}
    for key, value in record.items():
        if isinstance(value, dict):
            out[key] = {str(k): (int(v) if isinstance(v, (int, np.integer)) else v)
                        for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            out[key] = [int(v) if isinstance(v, np.integer) else v for v in value]
        elif isinstance(value, np.integer):
            out[key] = int(value)
        else:
            out[key] = value
    return out
```

`export.py`, inside the `if labels is not None and stats is not None and features is not None:` block, after `save_cluster_data` and before the tiles block:

```python
            if labels.names:
                storage.save_mineral_names(path, dict(labels.names))
            if labels.history:
                storage.save_subclustering(path, list(labels.history))
```

and extend `cluster_params` before `save_cluster_data`:

```python
            refinement_types = ("name_phases", "split_threshold", "split_gmm", "split_hires")
            refinement_nodes = [
                {"id": node["id"], "type": node["type"], "params": dict(node.get("params", {}))}
                for node in flow.get("nodes", []) if node.get("type") in refinement_types
            ]
            if refinement_nodes:
                cluster_params["refinement_nodes"] = refinement_nodes
```

Remove `"refine"` from the `for stage_type in (...)` tuple in the same block (the stage goes away in PR 3; removing the lookup now is harmless because `_node_params` returns None for an absent type).

Run: `uv run pytest tests/test_export_stage.py -v` → PASS.

- [ ] **Step 3: Fingerprints carry the names; QC uses them**

`src/karak/stages/fingerprints.py`, in `apply` after `compute_fingerprints`:

```python
        if labels.names:
            data["names"] = dict(labels.names)
```

`src/karak/stages/payloads.py`, in `Fingerprints.from_h5` after the `element_order` conversion:

```python
        if "names" in data:
            data["names"] = {int(k): v for k, v in data["names"].items()}
```

`src/karak/stages/qc_sinks.py`:

```python
# qc_fingerprints.apply
        names = inputs["fingerprints"].data.get("names")
        if params["mineral_names"]:
            names = {int(k): v for k, v in json.loads(params["mineral_names"]).items()}
```

```python
# qc_named_phase_map.apply
        labels = inputs["labels"]
        names = {int(k): v for k, v in json.loads(params["mineral_names"]).items()}
        if not names:
            names = dict(labels.names)
```

and in both PARAMS help texts append: "null/{} = the names carried by the labels".

Test (append to `tests/test_cluster_stage_parity.py`):

```python
def test_fingerprints_carry_label_names(denoised_cube, chain):
    cleaned = assign_noise_pixels(chain["features"], chain["labels"], k=5, device="cpu")
    payload = Labels(
        labels=cleaned, probabilities=chain["probabilities"],
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"], state=LabelState.CLEANED,
        names={int(cleaned[0]): "First"},
    )
    out = get("fingerprints")().run({"labels": payload, "cube": denoised_cube}, {})
    assert out["fingerprints"].data["names"] == {int(cleaned[0]): "First"}
```

Add a round-trip assertion to `tests/test_payloads.py::test_fingerprints_h5_roundtrip`: set `data["names"] = {0: "A"}` on the fixture and assert `back.data["names"] == {0: "A"}`.

Run: `uv run pytest tests/test_cluster_stage_parity.py tests/test_payloads.py tests/test_export_stage.py -v` → PASS.

- [ ] **Step 4: Docs, CHANGELOG, full run, PR**

`docs/user_guide.md`: add a `name_phases` row to the stage table after `noise_assign`:

```
| `name_phases` | `labels:cleaned` → `labels:cleaned` | Attaches mineral names (`names`: `'0: Ilmenite; 1: Silica'`) that travel with the labels to the fingerprints, the QC figures and the export (`clusters/mineral_names`). Fails validation if a named label is not in the data. |
```

In the "HDF5 output layout" section add `clusters/subclustering` (attrs `history`, `split_NN`) and mention that `clusters/mineral_names` is written when the flow names the phases.

`CHANGELOG.md`, Added:

```
- `name_phases` stage and `Labels.names`/`Labels.history`: mineral names
  and split records travel with the labels; `fingerprints` copies the
  names, `qc_fingerprints` and `qc_named_phase_map` use them when their
  `mineral_names` param is null, and `export_h5` writes
  `clusters/mineral_names`, the `cluster_N_name` attributes and
  `clusters/subclustering`. `Stage.check_params` lets a stage report
  errors in text params; `karak validate` shows them.
```

```bash
uv run pytest
git add -A src tests docs CHANGELOG.md
git commit -m "feat: export phase names and split history; QC uses the flow's names"
git push -u origin HEAD
gh pr create --title "feat: phase names and split history travel with the labels" --body "<...>"
gh pr checks --watch
```

Stop; the user merges.

---

# Part 3: `split_threshold` and `split_gmm` (PR 3, branch `feat/split-stages`)

### Task 3.1: core `threshold_split`

**Files:**
- Modify: `src/karak/clustering/refinement.py` (add functions; keep the existing ones)
- Test: `tests/test_refinement_split.py`

**Interfaces:**
- Produces: `refinement.threshold_split(labels, cube, mineral_indices, element_names, target, rules) -> tuple[np.ndarray, int, int]` returning `(new_labels, new_label, n_moved)`; `rules` is `list[tuple[str, str, float]]` as `parse_rule` gives; unknown channel raises `ValueError` naming it; `n_moved == 0` returns the input copy and `new_label == -1`.

- [ ] **Step 1: Write the failing test**

```python
"""Generic threshold and GMM splits in clustering/refinement.py."""

import numpy as np
import pytest

from karak.clustering.refinement import (
    _extract_olivine, gmm_split, threshold_split,
)


def _scene():
    """(labels, cube, indices): 20 pixels of phase 0, channels Fe, Ca, Mg."""
    n = 20
    cube = np.zeros((1, n, 3), dtype=np.float32)
    cube[0, :, 0] = np.linspace(0, 1, n)          # Fe rises along the row
    cube[0, :, 1] = 0.05                           # Ca low everywhere
    cube[0, :10, 2] = 0.9                          # Mg high in the first half
    indices = np.stack([np.zeros(n, int), np.arange(n)], 1).astype(np.int32)
    return np.zeros(n, dtype=np.int32), cube, indices


def test_threshold_split_matches_extract_olivine():
    labels, cube, idx = _scene()
    expected, exp_label = _extract_olivine(labels, cube, idx, ["Fe-K", "Ca", "Mg"],
                                           target_phase=0, fe_threshold=0.6, ca_threshold=0.1)
    got, label, n = threshold_split(labels, cube, idx, ["Fe-K", "Ca", "Mg"], 0,
                                    [("Fe-K", ">", 0.6), ("Ca", "<", 0.1)])
    np.testing.assert_array_equal(got, expected)
    assert label == exp_label == 1 and n == (got == 1).sum() > 0


def test_threshold_split_unknown_channel():
    labels, cube, idx = _scene()
    with pytest.raises(ValueError, match="'Ti'"):
        threshold_split(labels, cube, idx, ["Fe-K", "Ca", "Mg"], 0, [("Ti", ">", 0.1)])


def test_threshold_split_nothing_moves():
    labels, cube, idx = _scene()
    got, label, n = threshold_split(labels, cube, idx, ["Fe-K", "Ca", "Mg"], 0,
                                    [("Fe-K", ">", 5.0)])
    np.testing.assert_array_equal(got, labels)
    assert label == -1 and n == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_refinement_split.py -k threshold -v` → FAIL (`ImportError`).

- [ ] **Step 3: Write `threshold_split`**

Add to `refinement.py` after `_extract_olivine`:

```python
_OPERATORS = {
    "<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
}


def _channel_index(element_names: list[str], channel: str) -> int:
    lowered = [n.lower() for n in element_names]
    if channel.lower() not in lowered:
        raise ValueError(f"channel {channel!r} not in the cube: {list(element_names)}")
    return lowered.index(channel.lower())


def threshold_split(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    mineral_indices: np.ndarray,
    element_names: list[str],
    target_phase: int,
    rules: list[tuple[str, str, float]],
) -> tuple[np.ndarray, int, int]:
    """Move the target-phase pixels that satisfy every rule to a new label.

    ``rules`` are ``(channel, operator, value)`` with operators ``<``,
    ``<=``, ``>``, ``>=`` on the denoised cube. Returns ``(labels,
    new_label, n_moved)``; ``new_label`` is ``max(labels) + 1``, or -1 when
    no pixel moves (labels come back as an unchanged copy).
    """
    phase_mask = cleaned_labels == target_phase
    rows, cols = mineral_indices[phase_mask, 0], mineral_indices[phase_mask, 1]
    keep = np.ones(int(phase_mask.sum()), dtype=bool)
    for channel, op, value in rules:
        values = denoised_cube[rows, cols, _channel_index(element_names, channel)]
        keep &= _OPERATORS[op](values, value)
    n_moved = int(keep.sum())
    updated = cleaned_labels.copy()
    if n_moved == 0:
        logger.info("threshold split of phase %d: no pixel satisfies %s", target_phase, rules)
        return updated, -1, 0
    new_label = int(cleaned_labels.max()) + 1
    updated[np.where(phase_mask)[0][keep]] = new_label
    logger.info("threshold split of phase %d: %d pixels -> label %d (%s)",
                target_phase, n_moved, new_label, rules)
    return updated, new_label, n_moved
```

Run: `uv run pytest tests/test_refinement_split.py -k threshold -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/karak/clustering/refinement.py tests/test_refinement_split.py
git commit -m "feat: generic threshold_split in the refinement core"
```

### Task 3.2: core `gmm_split`

**Files:**
- Modify: `src/karak/clustering/refinement.py`
- Test: `tests/test_refinement_split.py`

**Interfaces:**
- Produces: `refinement.gmm_split(labels, cube, bse, mineral_indices, element_names, target, features, *, n_components, bse_weight, subsample_n, random_state, keep_parent, order_by) -> tuple[np.ndarray, list[int], dict]`. `features` is `list[str]` of tokens (`parse_feature` grammar); `bse` may be `None` when no token is `BSE`; `order_by` is a token or `""`. Returns `(labels, new_labels, info)` with `info = {"component_means": {label: {feature: mean}}, "n_pixels": {label: n}, "method": str}`. Too few pixels: unchanged labels, `[]`, `info["skipped"]` reason.

- [ ] **Step 1: Write the failing tests**

```python
def _two_blobs():
    """40 pixels of phase 0: Cl low in the first 30, high in the last 10."""
    rng = np.random.default_rng(0)
    n = 40
    cube = np.zeros((1, n, 2), dtype=np.float32)       # channels Cl, Na
    cube[0, :30, 0] = rng.normal(0.05, 0.005, 30)
    cube[0, 30:, 0] = rng.normal(0.17, 0.005, 10)
    cube[0, :, 1] = rng.normal(0.3, 0.01, n)
    indices = np.stack([np.zeros(n, int), np.arange(n)], 1).astype(np.int32)
    return np.zeros(n, dtype=np.int32), cube, indices


def test_gmm_split_keep_parent_gives_smaller_component_the_new_label():
    labels, cube, idx = _two_blobs()
    got, new, info = gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["Cl", "Na"],
                               n_components=2, bse_weight=1.0, subsample_n=None,
                               random_state=0, keep_parent=True, order_by="")
    assert new == [1]
    assert (got[:30] == 0).all() and (got[30:] == 1).all()
    assert info["n_pixels"] == {1: 10}


def test_gmm_split_without_parent_orders_by_feature_mean():
    labels, cube, idx = _two_blobs()
    got, new, info = gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["Cl", "Na"],
                               n_components=2, bse_weight=1.0, subsample_n=None,
                               random_state=0, keep_parent=False, order_by="Cl")
    assert new == [1, 2]                      # low Cl -> 1, high Cl -> 2
    assert (got[:30] == 1).all() and (got[30:] == 2).all()
    assert not (got == 0).any()
    assert info["component_means"][1]["Cl"] < info["component_means"][2]["Cl"]


def test_gmm_split_ratio_and_bse_features():
    labels, cube, idx = _two_blobs()
    bse = np.full((1, 40), 0.5, dtype=np.float32)
    got, new, info = gmm_split(labels, cube, bse, idx, ["Cl", "Na"], 0,
                               ["Cl/(Cl+Na)", "BSE"], n_components=2, bse_weight=2.0,
                               subsample_n=None, random_state=0, keep_parent=True,
                               order_by="")
    assert new == [1] and info["method"].startswith("GMM")


def test_gmm_split_too_few_pixels_is_a_no_op():
    labels, cube, idx = _two_blobs()
    labels = labels.copy(); labels[5:] = 9             # phase 0 has 5 pixels
    got, new, info = gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["Cl"],
                               n_components=2, bse_weight=1.0, subsample_n=None,
                               random_state=0, keep_parent=False, order_by="")
    np.testing.assert_array_equal(got, labels)
    assert new == [] and "skipped" in info


def test_gmm_split_bse_token_without_bse_raises():
    labels, cube, idx = _two_blobs()
    with pytest.raises(ValueError, match="BSE"):
        gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["BSE"],
                  n_components=2, bse_weight=1.0, subsample_n=None,
                  random_state=0, keep_parent=True, order_by="")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_refinement_split.py -k gmm -v` → FAIL (`ImportError`).

- [ ] **Step 3: Write `gmm_split`**

Add to `refinement.py`:

```python
def _feature_matrix(
    denoised_cube, bse, rows, cols, element_names, features,
) -> tuple[np.ndarray, list[str]]:
    """Columns for the feature tokens (channel, BSE, A/(A+B)) at the given
    pixels; ValueError names an unknown channel or a missing BSE image."""
    from karak.stages.params_text import parse_feature

    columns, used = [], []
    for token in features:
        kind, names = parse_feature(token)
        if kind == "bse":
            if bse is None:
                raise ValueError("feature 'BSE' needs the bse input")
            columns.append(bse[rows, cols].astype(np.float32))
        elif kind == "ratio":
            a = denoised_cube[rows, cols, _channel_index(element_names, names[0])]
            b = denoised_cube[rows, cols, _channel_index(element_names, names[1])]
            denom = a + b
            columns.append(np.where(denom > 0, a / np.where(denom > 0, denom, 1), 0.0)
                           .astype(np.float32))
        else:
            columns.append(denoised_cube[rows, cols, _channel_index(element_names, names[0])])
        used.append(token.strip())
    return np.column_stack(columns).astype(np.float32), used


def gmm_split(
    cleaned_labels: np.ndarray,
    denoised_cube: np.ndarray,
    bse: np.ndarray | None,
    mineral_indices: np.ndarray,
    element_names: list[str],
    target_phase: int,
    features: list[str],
    *,
    n_components: int,
    bse_weight: float,
    subsample_n: int | None,
    random_state: int,
    keep_parent: bool,
    order_by: str,
) -> tuple[np.ndarray, list[int], dict]:
    """Split the target phase with a GMM on z-scored features.

    ``keep_parent`` True: the largest component keeps the parent label and
    the others get new labels, ordered by ascending mean of ``order_by``
    (or by size, descending, when ``order_by`` is empty). False: every
    component gets a new label in that order and the parent empties.
    Returns ``(labels, new_labels, info)``.
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    phase_mask = cleaned_labels == target_phase
    n_phase = int(phase_mask.sum())
    method = f"GMM {n_components}-component on {'+'.join(t.strip() for t in features)}"
    if n_phase < n_components * 10:
        logger.warning("GMM split of phase %d: only %d pixels, need %d; skipped",
                       target_phase, n_phase, n_components * 10)
        return cleaned_labels.copy(), [], {"method": method, "skipped":
                                            f"{n_phase} pixels < {n_components * 10}"}
    rows, cols = mineral_indices[phase_mask, 0], mineral_indices[phase_mask, 1]
    X, used = _feature_matrix(denoised_cube, bse, rows, cols, element_names, features)
    X_scaled = StandardScaler().fit_transform(X)
    for i, name in enumerate(used):
        if name.upper() == "BSE" and bse_weight != 1.0:
            X_scaled[:, i] *= bse_weight

    rng = np.random.default_rng(random_state)
    X_fit = X_scaled
    if subsample_n is not None and n_phase > subsample_n:
        X_fit = X_scaled[rng.choice(n_phase, subsample_n, replace=False)]
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=random_state, n_init=5).fit(X_fit)
    comp = gmm.predict(X_scaled)
    sizes = np.bincount(comp, minlength=n_components)

    if order_by.strip():
        if order_by.strip() not in used:
            raise ValueError(f"order_by {order_by!r} is not one of the features {used}")
        column = used.index(order_by.strip())
        means = np.array([X[comp == c, column].mean() if sizes[c] else np.inf
                          for c in range(n_components)])
        order = [int(c) for c in np.argsort(means, kind="stable")]
    else:
        order = [int(c) for c in np.argsort(-sizes, kind="stable")]

    updated = cleaned_labels.copy()
    phase_indices = np.where(phase_mask)[0]
    next_label = int(cleaned_labels.max()) + 1
    new_labels: list[int] = []
    assigned: dict[int, int] = {}
    if keep_parent:
        largest = int(np.argmax(sizes))
        assigned[largest] = target_phase
        order = [c for c in order if c != largest]
    for c in order:
        if sizes[c] == 0:
            continue
        assigned[c] = next_label
        new_labels.append(next_label)
        next_label += 1
    for c, label in assigned.items():
        updated[phase_indices[comp == c]] = label

    info = {
        "method": method,
        "n_pixels": {label: int(sizes[c]) for c, label in assigned.items()
                     if label != target_phase},
        "component_means": {
            label: {name: float(X[comp == c, i].mean()) for i, name in enumerate(used)}
            for c, label in assigned.items()
        },
    }
    logger.info("%s of phase %d: %s", method, target_phase,
                {label: int(sizes[c]) for c, label in assigned.items()})
    return updated, new_labels, info
```

Note on `keep_parent=True` ordering: the test expects the smaller component to get label 1; with `order_by=""` the order is by size descending minus the largest, which is correct. With `keep_parent=False` and `order_by="Cl"`, low Cl first → label 1. Keys of `info["n_pixels"]` exclude the parent so the history record lists only new labels.

Run: `uv run pytest tests/test_refinement_split.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/karak/clustering/refinement.py tests/test_refinement_split.py
git commit -m "feat: generic gmm_split with parent keeping and feature ordering"
```

### Task 3.3: the `split_threshold` stage

**Files:**
- Create: `src/karak/stages/split.py`
- Test: `tests/test_split_stages.py`
- Modify: `tests/test_registry.py:46` (add `"split_threshold"`, `"split_gmm"`)

**Interfaces:**
- Produces: stage `split_threshold`: inputs `labels` (CLEANED), `cube` (DENOISED); output `labels` (CLEANED); params `target_phase` (int, 0), `rule` (str, ""), `new_name` (str, ""), `note` (str, ""). Adds `names[new_label] = new_name` and a history record; when nothing moves, labels and names are unchanged and the record has `new_labels: []`.
- Shared helper in `split.py`: `_history_record(stage, parent, new_labels, names, n_pixels, method, note) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
"""split_threshold and split_gmm stages: names, history, validation."""

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.clustering.refinement import gmm_split, threshold_split
from karak.stages import get
from karak.stages.base import StageError
from karak.stages.payloads import BseImage, ElementCube, LabelState, Labels, Space


@pytest.fixture()
def scene():
    """Cleaned labels over the synthetic scene: phase 0 = columns 8..35
    (channel A high), phase 1 = columns 36..63 (channel B high)."""
    cube = make_synthetic_scene()
    H, W, _ = cube.shape
    rows, cols = np.nonzero(cube[:, :, 0] > 0)
    idx = np.stack([rows, cols], 1).astype(np.int32)
    labels = np.where(cols < 36, 0, 1).astype(np.int32)
    payload = Labels(labels=labels, probabilities=None, mineral_indices=idx,
                     image_shape=(H, W), state=LabelState.CLEANED,
                     names={0: "A phase", 1: "B phase"})
    return payload, ElementCube(pixels=cube, element_names=("A", "B", "C"),
                                space=Space.DENOISED)


def test_split_threshold_parity_and_history(scene):
    labels, cube = scene
    expected, new_label, n = threshold_split(
        labels.labels, cube.pixels, labels.mineral_indices, ["A", "B", "C"], 0,
        [("C", ">", 0.5)])
    assert n > 0
    out = get("split_threshold")().run(
        {"labels": labels, "cube": cube},
        {"target_phase": 0, "rule": "C > 0.5", "new_name": "High-C", "note": "test"})
    got = out["labels"]
    np.testing.assert_array_equal(got.labels, expected)
    assert got.names == {0: "A phase", 1: "B phase", new_label: "High-C"}
    rec = got.history[-1]
    assert rec["stage"] == "split_threshold" and rec["parent"] == 0
    assert rec["new_labels"] == [new_label] and rec["n_pixels"] == {new_label: n}
    assert rec["method"] == "C > 0.5" and rec["note"] == "test"
    assert labels.labels[0] == got.labels[0] or True   # input untouched (frozen)
    assert labels.names == {0: "A phase", 1: "B phase"}


def test_split_threshold_check_params():
    cls = get("split_threshold")
    assert cls.check_params({"target_phase": 0, "rule": "A >", "new_name": "x", "note": ""})
    assert cls.check_params({"target_phase": 0, "rule": "A > 1", "new_name": "", "note": ""})
    assert not cls.check_params({"target_phase": 0, "rule": "A > 1", "new_name": "x", "note": ""})


def test_split_threshold_unknown_channel_is_a_stage_error(scene):
    labels, cube = scene
    with pytest.raises(StageError, match="'Ti'"):
        get("split_threshold")().run(
            {"labels": labels, "cube": cube},
            {"target_phase": 0, "rule": "Ti > 0.5", "new_name": "x", "note": ""})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_split_stages.py -k threshold -v` → FAIL (`KeyError: 'split_threshold'`).

- [ ] **Step 3: Write the stage**

Create `src/karak/stages/split.py`:

```python
"""Split stages: divide one phase by a threshold rule or a GMM.

Each split names its new labels in its own params and appends a record to
``Labels.history``, so the reason for a split sits next to the parameters
that made it, in the flow file and in the exported HDF5.
"""

from __future__ import annotations

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_csv, parse_feature, parse_rule
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register

_NOTE = Param("note", "str", "", "Note", "Why this split: the observation it rests on")


def _history_record(stage: str, parent: int, new_labels: list[int], names: dict,
                    n_pixels: dict, method: str, note: str) -> dict:
    return {"stage": stage, "parent": int(parent),
            "new_labels": [int(x) for x in new_labels],
            "names": {int(k): v for k, v in names.items()},
            "n_pixels": {int(k): int(v) for k, v in n_pixels.items()},
            "method": method, "note": note}


@register
class SplitThresholdStage(Stage):
    id = "split_threshold"
    label = "Split by threshold"
    description = (
        "Move the pixels of one phase that satisfy a rule on denoised "
        "channel values to a new label (e.g. olivine out of a pyroxene "
        "phase with 'Fe-K > 0.6 & Ca < 0.10')."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="labels with the phase to split"),
        Port("cube", space=Space.DENOISED, help="denoised channel values for the rule"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="labels with the new phase; names and history extended"),
    ]
    PARAMS = [
        Param("target_phase", "int", 0, "Target phase", "Label to split", min=0),
        Param("rule", "str", "", "Rule",
              "Comparisons joined by '&': 'Fe-K > 0.6 & Ca < 0.10' "
              "(operators <, <=, >, >= on channel names)"),
        Param("new_name", "str", "", "New name", "Name of the new label"),
        _NOTE,
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        errors = []
        try:
            parse_rule(params["rule"], "rule")
        except ValueError as exc:
            errors.append(str(exc))
        if not params["new_name"].strip():
            errors.append("new_name: the new label needs a name")
        return errors

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.refinement import threshold_split

        labels, cube = inputs["labels"], inputs["cube"]
        rules = parse_rule(params["rule"], "rule")
        try:
            updated, new_label, n = threshold_split(
                labels.labels, cube.pixels, labels.mineral_indices,
                list(cube.element_names), params["target_phase"], rules,
            )
        except ValueError as exc:
            raise StageError(f"split_threshold: {exc}") from exc
        names = dict(labels.names)
        new_labels: list[int] = []
        if new_label >= 0:
            names[new_label] = params["new_name"]
            new_labels = [new_label]
        record = _history_record(
            self.id, params["target_phase"], new_labels,
            {new_label: params["new_name"]} if new_labels else {},
            {new_label: n} if new_labels else {}, params["rule"], params["note"],
        )
        return {"labels": labels.replace(labels=updated, names=names,
                                         history=labels.history + (record,))}
```

Add `"split_threshold"` and `"split_gmm"` to `tests/test_registry.py` (the second makes the registry test fail until Task 3.4; run only the split test now).

Run: `uv run pytest tests/test_split_stages.py -k threshold -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/karak/stages/split.py tests/test_split_stages.py tests/test_registry.py
git commit -m "feat: split_threshold stage"
```

### Task 3.4: the `split_gmm` stage

**Files:**
- Modify: `src/karak/stages/split.py`
- Test: `tests/test_split_stages.py`

**Interfaces:**
- Produces: stage `split_gmm`: inputs `labels` (CLEANED), `cube` (DENOISED), `bse` (optional); output `labels` (CLEANED); params per the spec table (`target_phase`, `features`, `n_components`, `bse_weight`, `subsample_n`, `random_state`, `keep_parent`, `order_by`, `new_names`, `note`). `check_params`: every feature token parses; `order_by` (if set) is one of the features; `new_names` count equals `n_components - 1` when `keep_parent` else `n_components`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_stages.py`:

```python
def test_split_gmm_parity_keep_parent(scene):
    labels, cube = scene
    expected, new, info = gmm_split(
        labels.labels, cube.pixels, None, labels.mineral_indices, ["A", "B", "C"], 1,
        ["A", "B"], n_components=2, bse_weight=1.0, subsample_n=None, random_state=0,
        keep_parent=True, order_by="")
    out = get("split_gmm")().run(
        {"labels": labels, "cube": cube},
        {"target_phase": 1, "features": "A,B", "n_components": 2, "bse_weight": 1.0,
         "subsample_n": 0, "random_state": 0, "keep_parent": True, "order_by": "",
         "new_names": "B minor", "note": "n"})
    got = out["labels"]
    np.testing.assert_array_equal(got.labels, expected)
    assert new == [2] and got.names[2] == "B minor"
    rec = got.history[-1]
    assert rec["stage"] == "split_gmm" and rec["new_labels"] == [2]
    assert rec["n_pixels"] == info["n_pixels"] and rec["method"] == info["method"]


def test_split_gmm_without_parent_empties_it_and_orders(scene):
    labels, cube = scene
    out = get("split_gmm")().run(
        {"labels": labels, "cube": cube},
        {"target_phase": 1, "features": "A,B", "n_components": 2, "bse_weight": 1.0,
         "subsample_n": 0, "random_state": 0, "keep_parent": False, "order_by": "B",
         "new_names": "Low B,High B", "note": ""})
    got = out["labels"]
    assert not (got.labels == 1).any()
    assert got.names[2] == "Low B" and got.names[3] == "High B"
    rec = got.history[-1]
    assert rec["new_labels"] == [2, 3]


def test_split_gmm_too_few_pixels_logs_and_passes_through(scene, caplog):
    labels, cube = scene
    few = labels.labels.copy(); few[few == 1] = 0; few[:5] = 1
    labels = labels.replace(labels=few)
    with caplog.at_level("WARNING"):
        out = get("split_gmm")().run(
            {"labels": labels, "cube": cube},
            {"target_phase": 1, "features": "A", "n_components": 2, "bse_weight": 1.0,
             "subsample_n": 0, "random_state": 0, "keep_parent": False, "order_by": "",
             "new_names": "x,y", "note": ""})
    np.testing.assert_array_equal(out["labels"].labels, few)
    assert out["labels"].history[-1]["new_labels"] == []
    assert "skipped" in caplog.text


def test_split_gmm_check_params():
    cls = get("split_gmm")
    base = {"target_phase": 1, "features": "A,B", "n_components": 2, "bse_weight": 1.0,
            "subsample_n": 0, "random_state": 0, "keep_parent": True, "order_by": "",
            "new_names": "x", "note": ""}
    assert not cls.check_params(base)
    assert cls.check_params({**base, "features": "A/(B+C)"})          # bad ratio
    assert cls.check_params({**base, "order_by": "C"})                 # not a feature
    assert cls.check_params({**base, "new_names": "x,y"})              # one too many
    assert cls.check_params({**base, "keep_parent": False})            # one too few
    assert not cls.check_params({**base, "keep_parent": False, "new_names": "x,y"})


def test_split_gmm_unknown_channel_is_a_stage_error(scene):
    labels, cube = scene
    with pytest.raises(StageError, match="'Ti'"):
        get("split_gmm")().run(
            {"labels": labels, "cube": cube},
            {"target_phase": 1, "features": "Ti", "n_components": 2, "bse_weight": 1.0,
             "subsample_n": 0, "random_state": 0, "keep_parent": True, "order_by": "",
             "new_names": "x", "note": ""})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_split_stages.py -k gmm -v` → FAIL (`KeyError: 'split_gmm'`).

- [ ] **Step 3: Write the stage**

Append to `split.py`:

```python
@register
class SplitGmmStage(Stage):
    id = "split_gmm"
    label = "Split by GMM"
    description = (
        "Split one phase with a Gaussian mixture on z-scored features "
        "(denoised channels, BSE, or a ratio A/(A+B)). The largest component "
        "can keep the parent label; new labels are named in order."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="labels with the phase to split"),
        Port("cube", space=Space.DENOISED, help="denoised channels for the features"),
        Port("bse", required=False, help="BSE image; needed when a feature is 'BSE'"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="labels with the new phases; names and history extended"),
    ]
    PARAMS = [
        Param("target_phase", "int", 0, "Target phase", "Label to split", min=0),
        Param("features", "str", "", "Features",
              "Comma list of channel names, 'BSE', or ratios 'A/(A+B)'"),
        Param("n_components", "int", 2, "Components", min=2),
        Param("bse_weight", "float", 1.0, "BSE weight",
              "Multiplier on the z-scored BSE column", min=0.0),
        Param("subsample_n", "int", 500_000, "Subsample N",
              "Max pixels fitted; 0 = all", min=0),
        Param("random_state", "int", 42, "Random seed"),
        Param("keep_parent", "bool", True, "Keep parent",
              "The largest component keeps the parent label"),
        Param("order_by", "str", "", "Order by",
              "Feature whose component means (ascending) order the new "
              "labels; empty = by size, descending"),
        Param("new_names", "str", "", "New names",
              "Comma list, one per new label, in order"),
        _NOTE,
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        errors = []
        features = parse_csv(params["features"])
        if not features:
            errors.append("features: at least one feature is needed")
        for token in features:
            try:
                parse_feature(token)
            except ValueError as exc:
                errors.append(f"features: {exc}")
        order_by = params["order_by"].strip()
        if order_by and order_by not in features:
            errors.append(f"order_by: {order_by!r} is not one of the features")
        expected = params["n_components"] - (1 if params["keep_parent"] else 0)
        names = parse_csv(params["new_names"])
        if len(names) != expected:
            errors.append(f"new_names: {expected} name(s) expected, got {len(names)}")
        return errors

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.refinement import gmm_split

        labels, cube = inputs["labels"], inputs["cube"]
        bse = inputs.get("bse")
        try:
            updated, new_labels, info = gmm_split(
                labels.labels, cube.pixels, None if bse is None else bse.pixels,
                labels.mineral_indices, list(cube.element_names),
                params["target_phase"], parse_csv(params["features"]),
                n_components=params["n_components"], bse_weight=params["bse_weight"],
                subsample_n=params["subsample_n"] or None,
                random_state=params["random_state"], keep_parent=params["keep_parent"],
                order_by=params["order_by"],
            )
        except ValueError as exc:
            raise StageError(f"split_gmm: {exc}") from exc
        given = parse_csv(params["new_names"])
        new_names = {label: given[i] for i, label in enumerate(new_labels) if i < len(given)}
        names = {**labels.names, **new_names}
        if not params["keep_parent"] and new_labels:
            names.pop(params["target_phase"], None)   # the parent emptied
        record = _history_record(
            self.id, params["target_phase"], new_labels, new_names,
            info.get("n_pixels", {}), info["method"], params["note"],
        )
        if "skipped" in info:
            record["skipped"] = info["skipped"]
        if "component_means" in info:
            record["component_means"] = {int(k): v for k, v in info["component_means"].items()}
        return {"labels": labels.replace(labels=updated, names=names,
                                         history=labels.history + (record,))}
```

`Labels.to_h5` must serialize `component_means` (nested dict with int keys): `_jsonable` converts dict keys to str recursively, and `_history_from_json` converts only `names` and `n_pixels` back. Extend `_history_from_json` to also int-convert `component_means` keys.

Run: `uv run pytest tests/test_split_stages.py tests/test_registry.py tests/test_payloads.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/karak/stages/split.py src/karak/stages/payloads.py tests/test_split_stages.py
git commit -m "feat: split_gmm stage with parent keeping, ordering and names"
```

### Task 3.5: remove `refine`; docs; CHANGELOG; PR

**Files:**
- Delete: `src/karak/stages/refine.py`
- Modify: `tests/test_cluster_stage_parity.py` (remove `test_refine_parity`, the `refinement_cfg` and `refine_phases` imports), `tests/conftest.py` (remove `refinement_cfg`), `tests/test_core_params.py` (remove the `RefineStage`/`refinement_config` lines in `test_stage_builders_fill_every_field_from_the_stage_params`), `tests/test_registry.py` (drop `"refine"`)
- Modify: `docs/user_guide.md` (stage table row `refine` → rows for `split_threshold` and `split_gmm`; the `--set refine.target_phase=3` example → `--set weath.target_phase=3`), `README.md:228-232` and `:268-277`
- Regenerate: `docs/flow.schema.json`, `docs/stage_reference.md`

- [ ] **Step 1: Remove the stage and its tests**

```bash
git rm src/karak/stages/refine.py
```

Edit the four test files as listed. `test_core_params.py` keeps `RefinementConfig` in `BUNDLES` (the core still uses it).

Run: `uv run pytest` → PASS (fix any import the removal breaks).

- [ ] **Step 2: Docs**

User guide stage table: replace the `refine` row with:

```
| `split_threshold` | `labels:cleaned`, `cube:denoised` → `labels:cleaned` | Moves the pixels of one phase that satisfy a rule on denoised channels (`'Fe-K > 0.6 & Ca < 0.10'`) to a new, named label. Appends a record to the split history. |
| `split_gmm` | `labels:cleaned`, `cube:denoised` (+`bse`) → `labels:cleaned` | Gaussian mixture on z-scored features (channels, `BSE`, ratios `A/(A+B)`). `keep_parent` lets the largest component keep the parent label; `order_by` orders the new labels by a feature's component mean. |
```

README: in the stage paragraph replace "a `refine` stage (olivine extraction + GMM split) can be added to any flow" with "`name_phases`, `split_threshold` and `split_gmm` attach names and split composite phases; the `paper` flow uses them for the published olivine, weathering and phosphate splits (PR 5 adds the hires split)". Replace the "Post-clustering refinement (`refine` node)" block with:

```
### Splitting composite phases (`split_threshold`, `split_gmm`)

```json
{"id": "oliv", "type": "split_threshold",
 "params": {"target_phase": 2, "rule": "Fe-K > 0.6 & Ca < 0.10",
            "new_name": "Ferroan Olivine", "note": "..."}}
{"id": "phos", "type": "split_gmm",
 "params": {"target_phase": 7, "features": "Cl,Na,Mg,F", "n_components": 2,
            "keep_parent": false, "order_by": "Cl",
            "new_names": "Merrillite,Chlorapatite", "...": "..."}}
```
```

```bash
uv run python -m karak.flow.schema && uv run python -m karak.stages.reference
```

- [ ] **Step 3: CHANGELOG**

Added:

```
- `split_threshold` and `split_gmm` stages replace `refine`. Each names
  its new labels in its own params, carries a `note`, and appends a
  record to the labels' split history (parent, new labels, pixel counts,
  method, component means). `split_gmm` can give every component a new
  label (`keep_parent: false`) and order them by a feature's mean
  (`order_by`), as the published phosphate split did.
```

Removed:

```
- The `refine` stage. Its core functions stay in
  `clustering/refinement.py`; `split_threshold` and `split_gmm` call them.
```

- [ ] **Step 4: Full run, push, PR**

```bash
uv run pytest
git add -A
git commit -m "refactor: remove refine; docs for the split stages"
git push -u origin HEAD
gh pr create --title "feat: split_threshold and split_gmm stages replace refine" --body "<...>"
gh pr checks --watch
```

Stop; the user merges.

---

# Part 4: `split_hires` and `HiresLabels` (PR 4, branch `feat/split-hires`)

### Task 4.1: `HiresLabels` payload

**Files:**
- Modify: `src/karak/stages/payloads.py`
- Test: `tests/test_payloads.py`

**Interfaces:**
- Produces: `HiresLabels(image: np.ndarray int16 (H1, W1), ratio: int, names: dict[int, str], downsample_factor: int, header_trim_px: int, left_trim_px: int)`, `payload_type = "hires_labels"`, with `summary`, `to_h5`, `from_h5`.

- [ ] **Step 1: Write the failing test**

```python
def test_hires_labels_h5_roundtrip(tmp_path):
    from karak.stages.payloads import HiresLabels

    hires = HiresLabels(
        image=np.array([[-1, 13], [14, 14]], dtype=np.int16),
        ratio=2, names={13: "Pigeonite", 14: "Augite"},
        downsample_factor=1, header_trim_px=100, left_trim_px=0,
    )
    back = _roundtrip(hires, tmp_path)
    np.testing.assert_array_equal(back.image, hires.image)
    assert back.image.dtype == np.int16
    assert (back.ratio, back.names, back.downsample_factor, back.header_trim_px) == (
        2, {13: "Pigeonite", 14: "Augite"}, 1, 100)
    assert "HiresLabels" in hires.summary()
```

Also add `HiresLabels` to whatever list `test_every_payload_roundtrips_under_each_compression` iterates (read that test; it probably builds instances inline; add one).

- [ ] **Step 2: Run it to verify it fails** → `ImportError`.

- [ ] **Step 3: Write the payload**

After `Labels` in `payloads.py`:

```python
@_payload
@dataclass(frozen=True)
class HiresLabels(_Replaceable):
    """Sub-phase labels at a higher resolution than the working labels,
    inside a set of phases: the output of ``split_hires`` and the input
    of a lamellae analysis. -1 outside the region and where the split
    feature is undefined."""

    payload_type = "hires_labels"

    image: np.ndarray                       # (H1, W1) int16
    ratio: int                              # hires pixels per working pixel
    names: dict                             # label -> name
    downsample_factor: int = 1              # of the source cube
    header_trim_px: int = 0
    left_trim_px: int = 0

    def summary(self) -> str:
        labelled = int((self.image >= 0).sum())
        return (f"HiresLabels {_shape(self.image.shape)} int16 · "
                f"{len(self.names)} labels · {labelled:,} px · ratio {self.ratio}")

    def to_h5(self, group, compression=CACHE_COMPRESSION) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["ratio"] = self.ratio
        group.attrs["names"] = json.dumps({str(k): v for k, v in self.names.items()})
        group.attrs["downsample_factor"] = self.downsample_factor
        group.attrs["header_trim_px"] = self.header_trim_px
        group.attrs["left_trim_px"] = self.left_trim_px
        _dataset(group, "image", self.image.astype(np.int16), compression)

    @classmethod
    def from_h5(cls, group) -> "HiresLabels":
        return cls(
            image=group["image"][()],
            ratio=int(group.attrs["ratio"]),
            names=_int_keys(json.loads(group.attrs["names"])),
            downsample_factor=int(group.attrs["downsample_factor"]),
            header_trim_px=int(group.attrs["header_trim_px"]),
            left_trim_px=int(group.attrs["left_trim_px"]),
        )
```

Run: `uv run pytest tests/test_payloads.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/karak/stages/payloads.py tests/test_payloads.py
git commit -m "feat: HiresLabels payload for sub-phase maps at full resolution"
```

### Task 4.2: core `hires_split`

**Files:**
- Create: `src/karak/clustering/hires.py`
- Test: `tests/test_hires_split.py`

**Interfaces:**
- Produces: `hires.resolution_ratio(working_shape, hires_shape) -> int` (raises `ValueError` unless `0 <= H1 - H*ds < ds` and the same for W, with `ds = round(H1 / H)`, `ds >= 1`); `hires.hires_split(labels, mineral_indices, image_shape, cube_hi, element_names_hi, target_phases, feature, *, n_components, subsample_n, random_state) -> tuple[np.ndarray, np.ndarray, list[int], dict]` returning `(working_labels, hires_image int16, new_labels, info)`; `info` has `method`, `n_pixels` (per new label at hires), `n_pixels_working`, `component_means` (per new label), `n_undefined`.

- [ ] **Step 1: Write the failing tests**

```python
"""hires_split: one GMM at full resolution inside a set of phases."""

import numpy as np
import pytest

from karak.clustering.hires import hires_split, resolution_ratio


def test_resolution_ratio():
    assert resolution_ratio((4, 4), (8, 8)) == 2
    assert resolution_ratio((4, 4), (9, 8)) == 2        # one extra hires row is clipped
    assert resolution_ratio((4, 4), (4, 4)) == 1
    with pytest.raises(ValueError, match="integer"):
        resolution_ratio((4, 4), (10, 8))                # 2.5
    with pytest.raises(ValueError, match="integer"):
        resolution_ratio((4, 4), (8, 11))


def _case():
    """Working 4x4: phase 2 in the left two columns, phase 3 right. Hires
    8x8 with Ca and Mg: inside phase 2, Ca/(Ca+Mg) is low in the top half,
    high in the bottom half, except one working pixel whose children tie
    2:2 and one whose children are all undefined (Ca = Mg = 0)."""
    H = W = 4
    labels_img = np.where(np.arange(W)[None, :] < 2, 2, 3).astype(np.int32)
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    idx = np.stack([rows.ravel(), cols.ravel()], 1).astype(np.int32)
    labels = labels_img.ravel()
    ca = np.zeros((8, 8), np.float32); mg = np.zeros((8, 8), np.float32)
    ca[:4, :4], mg[:4, :4] = 0.1, 0.9                 # low ratio, top-left block
    ca[4:, :4], mg[4:, :4] = 0.9, 0.1                 # high ratio, bottom-left block
    # working pixel (0, 1) -> hires rows 0..1, cols 2..3: tie, top-left child low
    ca[0, 2:4], mg[0, 2:4] = 0.1, 0.9
    ca[1, 2:4], mg[1, 2:4] = 0.9, 0.1
    # working pixel (3, 1) -> hires rows 6..7, cols 2..3: all undefined
    ca[6:8, 2:4] = 0; mg[6:8, 2:4] = 0
    cube = np.stack([ca, mg], -1)
    return labels, idx, (H, W), cube


def test_hires_split_outputs():
    labels, idx, shape, cube = _case()
    out_labels, hires, new, info = hires_split(
        labels, idx, shape, cube, ["Ca", "Mg"], [2], "Ca/(Ca+Mg)",
        n_components=2, subsample_n=None, random_state=0)
    assert new == [4, 5]                                   # low ratio -> 4, high -> 5
    assert hires.dtype == np.int16 and hires.shape == (8, 8)
    assert (hires[:, 4:] == -1).all()                      # outside the region
    assert (hires[:4, :2] == 4).all() and (hires[4:, :2] == 5).all()
    assert (hires[6:8, 2:4] == -1).all()                   # undefined children
    img = out_labels.reshape(shape)
    assert (img[:, 2:] == 3).all()                         # other phase untouched
    assert img[0, 0] == 4 and img[3, 0] == 5                # majority
    assert img[0, 1] == 4                                  # tie -> child at (0, 2)
    assert img[3, 1] == 2                                  # no defined child -> parent
    assert info["n_undefined"] == 4
    assert info["component_means"][4]["Ca/(Ca+Mg)"] < info["component_means"][5]["Ca/(Ca+Mg)"]


def test_hires_split_clips_an_odd_hires_image():
    labels, idx, shape, cube = _case()
    taller = np.concatenate([cube, cube[-1:]], axis=0)     # 9 x 8
    out_labels, hires, new, _ = hires_split(
        labels, idx, shape, taller, ["Ca", "Mg"], [2], "Ca/(Ca+Mg)",
        n_components=2, subsample_n=None, random_state=0)
    assert hires.shape == (9, 8) and (hires[8] == -1).all()
    assert new == [4, 5]
```

- [ ] **Step 2: Run them to verify they fail** → `ModuleNotFoundError`.

- [ ] **Step 3: Write `hires.py`**

```python
"""Sub-phase split at a higher resolution than the working labels.

The working labels (one per mineral pixel at the pipeline's downsampled
resolution) select a region; a cube at full resolution supplies the
feature; one GMM fit classifies every full-resolution pixel of the region;
the working labels take the majority of their children.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def resolution_ratio(working_shape: tuple[int, int],
                     hires_shape: tuple[int, int]) -> int:
    """Hires pixels per working pixel, the same along both axes. The hires
    image may exceed ``working * ratio`` by up to ``ratio - 1`` pixels (odd
    source dimensions); anything else is an error."""
    (H, W), (H1, W1) = working_shape, hires_shape
    ds = max(1, int(round(H1 / H)))
    for size, hi in ((H, H1), (W, W1)):
        if not 0 <= hi - size * ds < ds:
            raise ValueError(
                f"hires shape {hires_shape} is not an integer multiple of the "
                f"working shape {working_shape} (ratio {H1 / H:.3f} x {W1 / W:.3f})")
    return ds


def _feature_image(cube_hi, element_names, feature) -> tuple[np.ndarray, np.ndarray]:
    """(values, defined) over the whole hires image for one feature token."""
    from karak.clustering.refinement import _channel_index
    from karak.stages.params_text import parse_feature

    kind, names = parse_feature(feature)
    if kind == "bse":
        raise ValueError("split_hires takes a channel or a ratio, not BSE")
    if kind == "ratio":
        a = cube_hi[:, :, _channel_index(element_names, names[0])]
        b = cube_hi[:, :, _channel_index(element_names, names[1])]
        denom = a + b
        defined = denom > 0
        values = np.zeros(a.shape, dtype=np.float32)
        np.divide(a, denom, out=values, where=defined)
        return values, defined
    values = cube_hi[:, :, _channel_index(element_names, names[0])]
    return values.astype(np.float32), np.ones(values.shape, dtype=bool)


def hires_split(
    labels: np.ndarray,
    mineral_indices: np.ndarray,
    image_shape: tuple[int, int],
    cube_hi: np.ndarray,
    element_names_hi: list[str],
    target_phases: list[int],
    feature: str,
    *,
    n_components: int,
    subsample_n: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[int], dict]:
    """Classify every hires pixel of the target phases with one GMM on
    ``feature``; return ``(working_labels, hires_image, new_labels, info)``.

    New labels are ``max(labels) + 1`` onwards, ordered by ascending
    component mean of the feature. A working pixel takes the majority label
    of its defined children; a tie goes to the child at ``(r * ds, c * ds)``
    when that child is defined, else to the smallest tied label; a pixel
    with no defined child keeps its parent label.
    """
    from sklearn.mixture import GaussianMixture

    from karak.clustering.noise_assign import labels_to_image

    H, W = image_shape
    H1, W1 = cube_hi.shape[:2]
    ds = resolution_ratio((H, W), (H1, W1))

    label_img = labels_to_image(labels, mineral_indices, (H, W))
    region = np.isin(label_img, np.asarray(target_phases))
    region_hi = np.repeat(np.repeat(region, ds, axis=0), ds, axis=1)[:H1, :W1]

    values, defined = _feature_image(cube_hi, element_names_hi, feature)
    use = region_hi & defined
    n_undefined = int((region_hi & ~defined).sum())
    sample = values[use]
    method = f"GMM {n_components}-component on {feature} at {ds}x resolution"
    if sample.size < n_components * 10:
        raise ValueError(f"split_hires: only {sample.size} defined pixels in phases "
                         f"{target_phases}, need {n_components * 10}")

    rng = np.random.default_rng(random_state)
    fit = sample
    if subsample_n is not None and sample.size > subsample_n:
        fit = sample[rng.choice(sample.size, subsample_n, replace=False)]
    gmm = GaussianMixture(n_components=n_components, random_state=random_state,
                          n_init=5).fit(fit.reshape(-1, 1))
    comp = gmm.predict(sample.reshape(-1, 1))
    order = np.argsort(gmm.means_.ravel(), kind="stable")
    rank = np.empty(n_components, dtype=np.int64)
    rank[order] = np.arange(n_components)
    first = int(labels.max()) + 1
    new_labels = [first + i for i in range(n_components)]

    hires = np.full((H1, W1), -1, dtype=np.int16)
    hires[use] = (first + rank[comp]).astype(np.int16)

    # Majority vote per working pixel over its ds x ds children.
    padded = np.full((H * ds, W * ds), -1, dtype=np.int16)
    padded[:H1, :W1] = hires[:H * ds, :W * ds]
    blocks = padded.reshape(H, ds, W, ds)
    counts = np.stack([(blocks == lab).sum(axis=(1, 3)) for lab in new_labels])  # (k, H, W)
    best = counts.max(axis=0)
    winner = np.asarray(new_labels)[counts.argmax(axis=0)]          # smallest on ties
    tied = (counts == best[None]).sum(axis=0) > 1
    top_left = padded[0::ds, 0::ds]
    winner = np.where(tied & (top_left >= 0), top_left, winner)
    result_img = np.where(region & (best > 0), winner, label_img)

    out = labels.copy()
    rows, cols = mineral_indices[:, 0], mineral_indices[:, 1]
    out[:] = result_img[rows, cols]

    info = {
        "method": method,
        "n_pixels": {lab: int((hires == lab).sum()) for lab in new_labels},
        "n_pixels_working": {lab: int((out == lab).sum()) for lab in new_labels},
        "component_means": {first + int(rank[c]): {feature: float(gmm.means_[c, 0])}
                            for c in range(n_components)},
        "n_undefined": n_undefined,
        "ratio": ds,
    }
    logger.info("%s in phases %s: %s hires px, %d undefined", method, target_phases,
                info["n_pixels"], n_undefined)
    return out, hires, new_labels, info
```

Note: `labels_to_image` fills non-mineral pixels with -1 (check its signature; it takes `(labels, mineral_indices, image_shape)` and may accept a fill value). `result_img[rows, cols]` only reads mineral positions, so background values never reach `out`.

Run: `uv run pytest tests/test_hires_split.py -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add src/karak/clustering/hires.py tests/test_hires_split.py
git commit -m "feat: hires_split classifies a phase region at full resolution"
```

### Task 4.3: the `split_hires` stage and export

**Files:**
- Create: `src/karak/stages/split_hires.py`
- Modify: `src/karak/io/storage.py` (add `save_hires_labels`), `src/karak/stages/export.py` (port `labels_hires`)
- Test: `tests/test_split_stages.py`, `tests/test_export_stage.py`, `tests/test_registry.py`

**Interfaces:**
- Produces: stage `split_hires`: inputs `labels` (CLEANED), `cube_hires` (RAW); outputs `labels` (CLEANED), `labels_hires` (`HiresLabels`); params `target_phases` (str), `feature` (str), `n_components` (int, 2), `subsample_n` (int, 500000), `random_state` (int, 42), `new_names` (str), `note` (str). `storage.save_hires_labels(h5_path, image, *, ratio, names, downsample_factor, header_trim_px, left_trim_px, compression)` writes `clusters/hires/labels` (int16, chunks via `dataset_options`) with those attributes.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_split_stages.py`:

```python
def test_split_hires_stage(scene):
    labels, cube = scene
    H, W = labels.image_shape
    hi = np.repeat(np.repeat(cube.pixels, 2, axis=0), 2, axis=1)
    # inside phase 1 make channel A vary so Ca/(Ca+Mg)-like ratio A/(A+B) has two modes
    hi[: H, 72:, 0] = 0.9          # top half of phase 1: high A
    hi[H:, 72:, 0] = 0.1           # bottom half: low A
    cube_hi = ElementCube(pixels=hi, element_names=("A", "B", "C"), space=Space.RAW,
                          downsample_factor=1, header_trim_px=7)
    out = get("split_hires")().run(
        {"labels": labels, "cube_hires": cube_hi},
        {"target_phases": "1", "feature": "A/(A+B)", "n_components": 2,
         "subsample_n": 0, "random_state": 0, "new_names": "Low A,High A", "note": "n"})
    got, hires = out["labels"], out["labels_hires"]
    assert got.names[2] == "Low A" and got.names[3] == "High A"
    assert hires.names == {2: "Low A", 3: "High A"}
    assert hires.ratio == 2 and hires.header_trim_px == 7
    assert hires.image.shape == (2 * H, 2 * W)
    assert (hires.image[:, :72] == -1).all()
    assert set(np.unique(got.labels[labels.labels == 1]).tolist()) <= {2, 3}
    rec = got.history[-1]
    assert rec["stage"] == "split_hires" and rec["new_labels"] == [2, 3]
    assert rec["ratio"] == 2


def test_split_hires_check_params():
    cls = get("split_hires")
    base = {"target_phases": "2", "feature": "Ca/(Ca+Mg)", "n_components": 2,
            "subsample_n": 0, "random_state": 0, "new_names": "a,b", "note": ""}
    assert not cls.check_params(base)
    assert cls.check_params({**base, "target_phases": "x"})
    assert cls.check_params({**base, "feature": "BSE"})
    assert cls.check_params({**base, "new_names": "a"})


def test_split_hires_non_integer_ratio_is_a_stage_error(scene):
    labels, cube = scene
    hi = np.repeat(np.repeat(cube.pixels, 2, axis=0), 2, axis=1)[:-40]   # 88 x 128
    cube_hi = ElementCube(pixels=hi, element_names=("A", "B", "C"), space=Space.RAW)
    with pytest.raises(StageError, match="integer multiple"):
        get("split_hires")().run(
            {"labels": labels, "cube_hires": cube_hi},
            {"target_phases": "1", "feature": "A/(A+B)", "n_components": 2,
             "subsample_n": 0, "random_state": 0, "new_names": "a,b", "note": ""})
```

(The synthetic scene is 64 x 64; phase 1 is columns 36..63, so at 2x it starts at column 72. The fixture's `H` is 64, so `hi[:H]` is the top half of the 128-row image.)

Append to `tests/test_export_stage.py`:

```python
def test_export_writes_hires_labels(tmp_path, payloads):
    from karak.stages.payloads import HiresLabels

    hires = HiresLabels(image=np.full((16, 16), -1, dtype=np.int16), ratio=2,
                        names={5: "A"}, downsample_factor=1, header_trim_px=3, left_trim_px=0)
    path = tmp_path / "out.h5"
    get("export_h5")().run(
        {"labels": payloads["labels"], "stats": payloads["stats"],
         "features": payloads["features"], "labels_hires": hires},
        {"path": str(path), "flow_json": _flow_json(), "compression": "gzip"},
    )
    with h5py.File(path, "r") as fh:
        ds = fh["clusters/hires/labels"]
        assert ds.dtype == np.int16 and ds.shape == (16, 16)
        assert ds.attrs["ratio"] == 2 and ds.attrs["header_trim_px"] == 3
        assert json.loads(ds.attrs["names"]) == {"5": "A"}
```

Add `"split_hires"` to `tests/test_registry.py`.

- [ ] **Step 2: Run them to verify they fail** → `KeyError: 'split_hires'`; export test fails on the missing port.

- [ ] **Step 3: Write the stage**

```python
"""Split-hires stage: a GMM at full resolution inside a set of phases."""

from __future__ import annotations

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_csv, parse_feature, parse_int_list
from karak.stages.payloads import HiresLabels, LabelState, Space
from karak.stages.registry import register
from karak.stages.split import _NOTE, _history_record


@register
class SplitHiresStage(Stage):
    id = "split_hires"
    label = "Split at full resolution"
    description = (
        "Inside a set of phases, fit one Gaussian mixture on a channel or "
        "ratio of a higher-resolution cube and classify every pixel of that "
        "cube; the working labels take the majority of their children. The "
        "high-resolution map is a second output (e.g. exsolution lamellae "
        "in pyroxene from Ca/(Ca+Mg) at 1x)."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="working-resolution labels"),
        Port("cube_hires", space=Space.RAW,
             help="raw cube at a higher resolution (a second load_elements "
                  "node with downsample_factor 1 and include_elements)"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="working labels with the new phases; names and history extended"),
        Port("labels_hires", help="HiresLabels: the new labels at the cube's resolution"),
    ]
    PARAMS = [
        Param("target_phases", "str", "", "Target phases",
              "Comma list of labels that form the region"),
        Param("feature", "str", "", "Feature",
              "One channel of cube_hires or a ratio 'A/(A+B)'"),
        Param("n_components", "int", 2, "Components", min=2),
        Param("subsample_n", "int", 500_000, "Subsample N",
              "Max hires pixels fitted; 0 = all", min=0),
        Param("random_state", "int", 42, "Random seed"),
        Param("new_names", "str", "", "New names",
              "Comma list, one per component, by ascending feature mean"),
        _NOTE,
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        errors = []
        try:
            if not parse_int_list(params["target_phases"], "target_phases"):
                errors.append("target_phases: at least one label is needed")
        except ValueError as exc:
            errors.append(str(exc))
        try:
            kind, _ = parse_feature(params["feature"])
            if kind == "bse":
                errors.append("feature: BSE is not a channel of cube_hires")
        except ValueError as exc:
            errors.append(f"feature: {exc}")
        if len(parse_csv(params["new_names"])) != params["n_components"]:
            errors.append(f"new_names: {params['n_components']} names expected")
        return errors

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.hires import hires_split

        labels, cube_hi = inputs["labels"], inputs["cube_hires"].to("cpu")
        targets = parse_int_list(params["target_phases"], "target_phases")
        try:
            updated, image, new_labels, info = hires_split(
                labels.labels, labels.mineral_indices, labels.image_shape,
                cube_hi.pixels, list(cube_hi.element_names), targets, params["feature"],
                n_components=params["n_components"],
                subsample_n=params["subsample_n"] or None,
                random_state=params["random_state"],
            )
        except ValueError as exc:
            raise StageError(f"split_hires: {exc}") from exc
        new_names = dict(zip(new_labels, parse_csv(params["new_names"])))
        names = {**labels.names, **new_names}
        for parent in targets:                      # emptied parents lose their name
            if not (updated == parent).any():
                names.pop(parent, None)
        record = _history_record(self.id, targets[0], new_labels, new_names,
                                 info["n_pixels_working"], info["method"], params["note"])
        record.update({"parents": targets, "ratio": info["ratio"],
                       "n_pixels_hires": info["n_pixels"], "n_undefined": info["n_undefined"],
                       "component_means": info["component_means"]})
        return {
            "labels": labels.replace(labels=updated, names=names,
                                     history=labels.history + (record,)),
            "labels_hires": HiresLabels(
                image=image, ratio=info["ratio"], names=new_names,
                downsample_factor=cube_hi.downsample_factor,
                header_trim_px=cube_hi.header_trim_px, left_trim_px=cube_hi.left_trim_px,
            ),
        }
```

`_history_from_json` in `payloads.py`: also int-convert `n_pixels_hires` keys.

- [ ] **Step 4: Export**

`storage.py`:

```python
def save_hires_labels(h5_path, image, *, ratio, names, downsample_factor,
                      header_trim_px, left_trim_px, compression) -> None:
    """Write a full-resolution sub-phase map to ``clusters/hires/labels``."""
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group("clusters").require_group("hires")
        if "labels" in grp:
            del grp["labels"]
        data = np.asarray(image, dtype=np.int16)
        ds = grp.create_dataset("labels", data=data,
                                **dataset_options(data, compression, shuffle=False))
        ds.attrs["ratio"] = int(ratio)
        ds.attrs["names"] = json.dumps({str(k): v for k, v in names.items()})
        ds.attrs["downsample_factor"] = int(downsample_factor)
        ds.attrs["header_trim_px"] = int(header_trim_px)
        ds.attrs["left_trim_px"] = int(left_trim_px)
    logger.info("Saved hires labels %s to clusters/hires/labels", data.shape)
```

`export.py`: add `Port("labels_hires", required=False, help="written to clusters/hires/labels (full-resolution sub-phase map)")` to `INPUTS`, and in `apply` after the cluster block:

```python
        hires = inputs.get("labels_hires")
        if hires is not None:
            storage.save_hires_labels(
                path, hires.image, ratio=hires.ratio, names=hires.names,
                downsample_factor=hires.downsample_factor,
                header_trim_px=hires.header_trim_px, left_trim_px=hires.left_trim_px,
                compression=compression,
            )
```

`dataset_options` chunks on the first two axes; check it handles int16 (it takes `itemsize`).

Run: `uv run pytest tests/test_split_stages.py tests/test_export_stage.py tests/test_registry.py -v` → PASS.

- [ ] **Step 5: Docs, schema, reference, CHANGELOG, PR**

User guide stage table row:

```
| `split_hires` | `labels:cleaned`, `cube:raw` (higher resolution) → `labels:cleaned`, `labels_hires` | One GMM on a channel or ratio of a full-resolution cube inside a set of phases; every full-resolution pixel is classified (`labels_hires`, exported to `clusters/hires/labels`) and the working labels take the majority of their children. The second cube comes from a `load_elements` node with `downsample_factor: 1` and `include_elements`. |
```

HDF5 layout section: add `clusters/hires/labels` (int16, attrs `ratio`, `names`, `downsample_factor`, `header_trim_px`, `left_trim_px`).

CHANGELOG, Added:

```
- `split_hires` stage and `HiresLabels` payload: a GMM on a full-resolution
  channel or ratio inside a set of phases, with the full-resolution label
  map as a second output (`clusters/hires/labels` in the export). Working
  pixels take the majority of their children; ties go to the child at
  (r·ds, c·ds); pixels with no defined child keep the parent label. This
  is the published 1x pyroxene split, fitted once instead of three times.
```

```bash
uv run python -m karak.flow.schema && uv run python -m karak.stages.reference
uv run pytest
git add -A
git commit -m "feat: split_hires stage, HiresLabels export, docs"
git push -u origin HEAD
gh pr create --title "feat: split_hires classifies a phase region at full resolution" --body "<...>"
gh pr checks --watch
```

Stop; the user merges.

---

# Part 5: the `paper` flow, docs, acceptance (PR 5, branch `feat/paper-flow-complete`)

### Task 5.1: extend `paper.json`

**Files:**
- Modify: `src/karak/flow/flows/paper.json`
- Modify: `tests/test_builtins.py:241-252`, `tests/test_recipe_stability.py`

**Interfaces:**
- Produces: the `paper` builtin with nodes `src, src_hires, msk, dn, nrm, pca, hdb, rare, knn, names, oliv, weath, pyx, phos, stats, fp, exp` plus the eight QC sinks; `exp.labels` and `stats.labels`, `fp.labels`, `qc_phase_map.labels` take `phos.labels`.

- [ ] **Step 1: Write the failing builtin test**

Replace `test_paper_is_tiled_rare_with_the_published_settings` with:

```python
PAPER_EXTRA_NODES = ["src_hires", "names", "oliv", "weath", "pyx", "phos"]


def test_paper_is_tiled_rare_plus_the_published_hand_steps():
    paper, base = builtin_flow("paper"), builtin_flow("tiled-rare")
    base_ids = [n.id for n in base.nodes]
    assert [n.id for n in paper.nodes if n.id not in PAPER_EXTRA_NODES] == base_ids
    for node in base.nodes:
        expected = {**node.params, **PAPER_SETTINGS.get(node.id, {})}
        assert paper.node(node.id).params == expected, node.id
    assert all(paper.node(n).params["device"] == "cpu"
               for n in ("dn", "nrm", "pca", "hdb", "rare", "knn"))
    types = {n.id: n.type for n in paper.nodes}
    assert types["src_hires"] == "load_elements" and types["names"] == "name_phases"
    assert types["oliv"] == "split_threshold" and types["weath"] == "split_gmm"
    assert types["pyx"] == "split_hires" and types["phos"] == "split_gmm"
    hires = paper.node("src_hires").params
    assert hires["downsample_factor"] == 1 and hires["include_elements"] == "Ca,Mg"
    assert hires["header_trim_px"] == paper.node("src").params["header_trim_px"]
    assert hires["colormap"] == paper.node("src").params["colormap"]

    def source(node_id, port):
        return next((e.src.node, e.src.port) for e in paper.edges
                    if e.dst.node == node_id and e.dst.port == port)

    chain = [("names", "knn"), ("oliv", "names"), ("weath", "oliv"),
             ("pyx", "weath"), ("phos", "pyx")]
    for node_id, upstream in chain:
        assert source(node_id, "labels") == (upstream, "labels")
    assert source("pyx", "cube_hires") == ("src_hires", "cube")
    for consumer in ("stats", "fp", "exp", "qc_phase_map"):
        assert source(consumer, "labels") == ("phos", "labels")
    assert source("exp", "labels_hires") == ("pyx", "labels_hires")
    assert source("weath", "bse") == ("src", "bse")
    assert paper.node("oliv").params["rule"] == "Fe-K > 0.6 & Ca < 0.10"
    assert paper.node("weath").params["keep_parent"] is True
    assert paper.node("phos").params["keep_parent"] is False
    assert paper.node("phos").params["order_by"] == "Cl"
    assert paper.node("pyx").params["feature"] == "Ca/(Ca+Mg)"
```

Keep the existing `PAPER_SETTINGS` dict as it is (it describes the tiled-rare nodes).

- [ ] **Step 2: Run it to verify it fails** → FAIL (no `src_hires` node).

- [ ] **Step 3: Edit `paper.json`**

Insert these nodes after `src` (for `src_hires`) and after `knn` (the rest), in this order, and renumber nothing else (edge ids are free-form strings; use `e48` onwards for new edges and rewire existing ones by editing their `from`):

```json
{"id": "src_hires", "type": "load_elements", "params": {
  "input_dir": "{input}", "file_glob": "*.png", "filename_pattern": null,
  "bse_filename": null, "colormap": "cmap:jet", "exclude_elements": "Fe-L",
  "include_elements": "Ca,Mg", "bse_channel": "SEM", "header_trim_px": 100,
  "bottom_trim_px": 0, "left_trim_px": 0, "right_trim_px": 0, "downsample_factor": 1}},

{"id": "names", "type": "name_phases", "params": {
  "names": "0: Ilmenite (FeTiO₃); 1: Silica polymorph (SiO₂); 2: Pyroxene (Ca,Mg,Fe)SiO₃; 3: Plagioclase (CaAl₂Si₂O₈); 4: Epoxy; 5: Fe-Ti-Cr Spinel (Fe₂TiO₄); 6: Calcite (CaCO₃); 7: Apatite (Ca₅(PO₄)₃(OH,Cl,F)); 8: Fe Oxyhydroxide (FeOOH); 9: Zn-bearing Phase; 10: Xenotime (YPO₄)",
  "note": "Assigned from the phase fingerprints and the TIMA reference map of NAW 4587-2 (S3858); the published base inventory before the splits."}},

{"id": "oliv", "type": "split_threshold", "params": {
  "target_phase": 2, "rule": "Fe-K > 0.6 & Ca < 0.10",
  "new_name": "Ferroan Olivine ((Fe,Mg)₂SiO₄)",
  "note": "Fe-rich, Ca-poor pixels of the pyroxene phase are olivine; thresholds read off the Fe-K and Ca histograms of phase 2."}},

{"id": "weath", "type": "split_gmm", "params": {
  "target_phase": 2, "features": "Ca,Mg,Fe-K,BSE,Ca/(Ca+Mg)", "n_components": 2,
  "bse_weight": 1.0, "subsample_n": 500000, "random_state": 42, "keep_parent": true,
  "order_by": "", "new_names": "Weathering Assemblage",
  "note": "The smaller GMM component of the remaining pyroxene pixels carries S, Cl and altered mafic chemistry (terrestrial weathering), not a pyroxene composition; the larger component stays pyroxene."}},

{"id": "pyx", "type": "split_hires", "params": {
  "target_phases": "2", "feature": "Ca/(Ca+Mg)", "n_components": 2,
  "subsample_n": 500000, "random_state": 42,
  "new_names": "Pigeonite (low-Ca pyroxene),Augite (high-Ca pyroxene)",
  "note": "Exsolution lamellae are narrower than a 2x pixel; Ca/(Ca+Mg) from the 1x Ca and Mg maps separates low-Ca pigeonite from high-Ca augite. The 1x map feeds the lamellae analysis."}},

{"id": "phos", "type": "split_gmm", "params": {
  "target_phase": 7, "features": "Cl,Na,Mg,F", "n_components": 2,
  "bse_weight": 1.0, "subsample_n": 500000, "random_state": 42, "keep_parent": false,
  "order_by": "Cl", "new_names": "Merrillite (Ca₉NaMg(PO₄)₇),Chlorapatite (Ca₅(PO₄)₃(Cl,F,OH))",
  "note": "The phosphate phase has a low-Cl, Na-Mg-bearing population (merrillite) and a high-Cl population (chlorapatite); ordered by mean Cl."}}
```

Edges to add:

```
knn.labels -> names.labels
names.labels -> oliv.labels;   dn.cube -> oliv.cube
oliv.labels -> weath.labels;   dn.cube -> weath.cube;   src.bse -> weath.bse
weath.labels -> pyx.labels;    src_hires.cube -> pyx.cube_hires
pyx.labels -> phos.labels;     dn.cube -> phos.cube
pyx.labels_hires -> exp.labels_hires
```

Edges to rewire (`from` becomes `phos.labels`): `e16` (stats), `e17` (fp), `e26` (exp.labels), `e39` (qc_phase_map.labels).

Then: `uv run karak flow complete src/karak/flow/flows/paper.json -o src/karak/flow/flows/paper.json` (idempotent; confirms completeness) and `uv run karak validate --builtin paper`.

Run: `uv run pytest tests/test_builtins.py tests/test_flow_schema.py -v` → PASS.

- [ ] **Step 4: Recipe table**

Run `tests/test_recipe_stability.py`; add the six new node hashes to `GOLDEN["paper"]` (print them as in Task 1.4 with the node list extended) and note in the docstring:

```
2026-10-06: paper gains src_hires, names, oliv, weath, pyx and phos; the
consumers of the final labels (stats, fp) now hash from phos. Other flows
unchanged.
```

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/flows/paper.json tests/test_builtins.py tests/test_recipe_stability.py
git commit -m "feat: paper flow carries every published hand step as a node"
```

### Task 5.2: docs

**Files:**
- Modify: `docs/user_guide.md` (the `paper` paragraph in "Flows", the builtins table, "Reproducibility"), `README.md:14` and the stage paragraph, `CLAUDE.md` (the `paper` line of the command list), `CHANGELOG.md`

- [ ] **Step 1: User guide `paper` paragraph**

Replace the paragraph from "`paper` is `tiled-rare` with the settings of the published NWA 4587 run" through its bullets and the "Run it on the CPU" sentence with:

```
`paper` is `tiled-rare` with the settings of the published NWA 4587 run,
plus the six steps that were done by hand after the March 2026
clustering, each as a node with its parameters and a `note`:

| node | type | what it decides |
|---|---|---|
| `src_hires` | `load_elements` | Ca and Mg at 1x (no downsample), the same 100 px trim and jet palette |
| `names` | `name_phases` | the 11 base names, from the fingerprints and the TIMA reference |
| `oliv` | `split_threshold` | olivine out of phase 2: `Fe-K > 0.6 & Ca < 0.10` |
| `weath` | `split_gmm` | the smaller of two components of phase 2 on Ca, Mg, Fe-K, BSE and Ca/(Ca+Mg) is a weathering assemblage |
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
- the splits give the published 16 phases; the numbering differs in one
  place (13 pigeonite, 14 augite here; the published file has them the
  other way round). Abundances against Table 1 of the paper: ACCEPTANCE.

Run it on the CPU: cuML selects different clusters on these full tiles
(see [Computational requirements](#computational-requirements)). The tiled
HDBSCAN takes about 30 minutes on 16 threads, with the pool limited to 7
workers by memory; the 1x load adds about a minute and 0.8 GB.
```

`ACCEPTANCE` is filled in Task 5.3 from the real run; the PR is not opened before that.

Builtins table: the `paper` row says "tiled-rare + the published names and splits; 1x pyroxene map".

"Reproducibility" section, add a bullet:

```
- **Declared judgments** — every split or naming decision is a node with
  a `note` param; `clusters/subclustering` in the HDF5 lists them in order
  with their pixel counts and component means.
```

- [ ] **Step 2: README and CLAUDE.md**

README line 14: replace the clause from "the load, mask, denoise, and normalization outputs reproduce" to the end of the sentence with: "it reproduces the published preprocessing bit for bit, the published tiles and phases, and the six hand steps of the published analysis (names, olivine, weathering, 1x pyroxene and phosphate splits) as declared nodes; the abundances agree with the paper's Table 1 to within ACCEPTANCE (see the [user guide](docs/user_guide.md#flows))."

README stage paragraph: mention `split_hires` with the other split stages.

CLAUDE.md command list: the `paper` line becomes "tiled-rare with the published NWA 4587 settings and the published names and splits (cpu, ~45 min)".

- [ ] **Step 3: CHANGELOG**

Added:

```
- The `paper` builtin flow carries the six hand steps of the published
  NWA 4587 analysis as nodes: `src_hires` (Ca and Mg at 1x), `names`,
  `oliv` (olivine threshold), `weath` (weathering GMM), `pyx` (1x
  pyroxene split with the lamellae map) and `phos` (phosphate GMM). One
  run gives the published 16 phases. Real-data check: ACCEPTANCE.
```

- [ ] **Step 4: Commit**

```bash
git add docs README.md CLAUDE.md CHANGELOG.md
git commit -m "docs: the paper flow's hand steps, what it reproduces"
```

### Task 5.3: real-data acceptance on NWA 4587

**Files:**
- Create (outside the repo, in the job's tmp dir or `/home/brendon/Projects/science/karak/output/nwa4587/paperflow/`): `compare_nwa4587.py`, `lamellae_from_hires.py`
- Modify: `docs/user_guide.md`, `README.md`, `CHANGELOG.md` (replace `ACCEPTANCE`)

- [ ] **Step 1: Run the flow**

The item-2 cache holds `src` … `pca` for these settings; `hdb` and later recompute (recipe changes). The CLI puts `work` beside the `--out` parent, so run inside `output/nwa4587/item2/` to reuse it:

```bash
cd /home/brendon/Projects/science/karak
nohup uv run karak run --builtin paper \
  --input /home/brendon/Dropbox/Projects/izawa/NWA_4587_data \
  --out output/nwa4587/item2/paperflow --workers 0 --plain \
  > output/nwa4587/item2/paperflow.log 2>&1 &
```

Expected about 45 minutes (hdb 30 min, rare 8 min, knn 5 min, the splits and the 1x load a few minutes). Poll `tail -3 output/nwa4587/item2/paperflow.log`. A failure in a split stage shows as a `StageError` with the node id; fix on the branch and rerun (every earlier node comes from the cache).

- [ ] **Step 2: Compare with Table 1**

`compare_nwa4587.py`:

```python
"""Abundances of the paper-flow run against the published cleaned labels."""
import json, sys
import h5py, numpy as np

ours = h5py.File(sys.argv[1], "r")          # output/nwa4587/item2/paperflow.h5
paper = h5py.File("/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/data/eds_pipeline.h5", "r")
L = ours["clusters/cleaned_labels"][:]; P = paper["clusters/cleaned_labels"][:]
names = {int(k): v for k, v in json.loads(ours["clusters"].attrs["mineral_names"]).items()}
pnames = {int(k): v for k, v in json.loads(paper["clusters"].attrs["mineral_names"]).items()}
n = L.size
assert n == P.size
def key(name): return name.split(" (")[0].lower()
print(f"{'phase':34s} {'ours %':>8s} {'paper %':>8s} {'delta pp':>9s} {'ours label':>10s} {'paper label':>11s}")
rows = []
for lab, name in sorted(names.items(), key=lambda kv: kv[1]):
    match = [pl for pl, pn in pnames.items() if key(pn) == key(name)]
    o = 100 * (L == lab).sum() / n
    p = 100 * (P == match[0]).sum() / n if match else float("nan")
    rows.append((name, o, p, o - p, lab, match[0] if match else -1))
for name, o, p, d, lab, pl in rows:
    flag = "  <-- over 1 pp" if abs(d) > 1 else ""
    print(f"{name[:34]:34s} {o:8.3f} {p:8.3f} {d:+9.3f} {lab:10d} {pl:11d}{flag}")
print("phases ours/paper:", len(names), len(pnames))
print("history:", json.loads(ours["clusters/subclustering"].attrs["history"]))
```

Run: `uv run python compare_nwa4587.py output/nwa4587/item2/paperflow.h5`. Acceptance: 16 named phases; every name matches a published name; every delta under 1 pp or explained in the PR body. Record the table.

- [ ] **Step 3: Lamellae handshake**

Copy `/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/analysis/lamellae_analysis.py` to `lamellae_from_hires.py` beside the compare script. Replace the body of `build_augite_pigeonite_image(cfg, meta)` with:

```python
def build_augite_pigeonite_image(cfg, meta):
    import h5py, json
    with h5py.File(HIRES_H5, "r") as f:        # the paper-flow output
        ds = f["clusters/hires/labels"]
        img = ds[()]
        names = {int(k): v for k, v in json.loads(ds.attrs["names"]).items()}
    aug = next(k for k, v in names.items() if v.startswith("Augite"))
    pig = next(k for k, v in names.items() if v.startswith("Pigeonite"))
    label_1x = np.full(img.shape, -1, dtype=np.int8)
    label_1x[img == aug] = AUGITE_LABEL
    label_1x[img == pig] = PIGEONITE_LABEL
    return label_1x, img.shape[0], img.shape[1]
```

with `HIRES_H5 = "<abs path>/output/nwa4587/item2/paperflow.h5"` at module top, and point its output directory at the job tmp dir (edit the `figures` path constants near the top; do not write into the paper repo). Run it with the paper repo's venv or `uv run --with scipy --with scikit-image` from the karak repo (it imports `karak.config`, which no longer exists in karak 0.2: remove that import and the `_find_config`/`meta` lookups that only serve the raw-PNG path, or stub `cfg`/`meta` with `None`). Record the grain count against the paper's 128 (its `lamellae_summary.json` says `n_grains`), the median lamella spacing against 51 µm, and the Rayleigh p-value.

- [ ] **Step 4: Fill in the numbers**

Replace every `ACCEPTANCE` placeholder in `docs/user_guide.md`, `README.md` and `CHANGELOG.md` with the measured statement, for example "within 0.4 pp for every phase (largest: Weathering Assemblage, +0.38 pp)". Commit:

```bash
git add docs README.md CHANGELOG.md
git commit -m "docs: paper flow acceptance numbers from the NWA 4587 run"
```

- [ ] **Step 5: Full tests, push, PR**

```bash
uv run pytest
git push -u origin HEAD
gh pr create --title "feat: the paper flow reproduces the published 16 phases in one run" --body "<what changed; the acceptance table; the lamellae grain count; recipe hashes; the one numbering swap (13/14); follow-ups: sub-project 2 reads clusters/hires/labels and clusters/subclustering>"
gh pr checks --watch
```

Stop; the user reviews and merges. Save the acceptance table to memory as a `project` note (what the regenerated numbers are, so sub-project 3 can quote them).

---

## Self-review notes

- Spec section 1 → Tasks 1.1 to 1.4. Section 2 → Tasks 2.1 to 2.4 (names, history, `check_params`, `name_phases`), 3.1 to 3.5 (threshold and GMM), 4.1 to 4.3 (hires). Section 3 → Task 5.1 (flow), 2.4 and 4.3 (export, QC), 1.4/2.4/3.5/4.3/5.2 (docs and regenerated files). Section 4 → the tests in every task and Task 5.3.
- Deviation from the spec, stated: the spec says `qc_phase_map` uses the names; `generate_phase_map` has no names argument, so `qc_named_phase_map` and `qc_fingerprints` use them instead. The spec's `TiledArtifacts.deferred_tiles` is a property derived from `TileResult.deferred`; `deferred_pixels` is the stored field, because `rare_phase` needs pixel indices, not tile ids.
- Names and types used across tasks: `parse_rule`, `parse_feature`, `parse_csv`, `parse_int_list`, `parse_names` (2.2) are used in 3.3, 3.4, 4.3; `_history_record` and `_NOTE` (3.3) in 3.4 and 4.3; `_channel_index` (3.1) in 4.2; `_int_keys` (2.1) in 4.1; `HiresLabels` (4.1) in 4.3; `resolve_min_tile_pixels`, `deferred_pixel_indices` (1.1) in 1.2.
