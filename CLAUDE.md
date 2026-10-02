# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

Every change to the repository (code, tests, docs, shipped flows) goes
through a pull request. Never commit or push directly to `main`.

1. Branch from an up-to-date `main`:
   `git switch main && git pull && git switch -c <type>/<short-name>`,
   where `<type>` is `feat`, `fix`, `refactor`, `docs`, `test`, or `chore`.
2. Commit in small, tested steps. Run `uv run pytest` before each push.
   Update `CHANGELOG.md` (Unreleased) in the same branch as the change.
3. Push and open the PR: `git push -u origin HEAD`, then `gh pr create`.
   - Title: conventional-commit style (`feat: ...`, `fix: ...`); it becomes
     the squash commit message on `main`.
   - Body: what changed and why; how it was tested (commands and results,
     including any real-data check); rulings or open questions; follow-ups.
4. Wait for CI (`gh pr checks --watch`) and fix any failure on the branch.
5. Stop there. The user reviews and merges. Merge only when the user says
   so, and then with `gh pr merge --squash --delete-branch`.
6. After a merge: `git switch main && git pull`.

## Project

Karak is an automated mineralogy pipeline for SEM-EDS (Scanning Electron Microscopy with Energy Dispersive Spectroscopy) elemental map stacks. It transforms jet-colormapped elemental map PNGs into mineral phase maps with chemical fingerprints. Python >=3.12, built with Hatchling.

## Commands

```bash
uv sync --group dev                 # install with test dependencies
uv run pytest                       # run the test suite

karak flow init --builtin tiled -o my_flow.json      # complete flow JSON to edit
karak flow complete FLOW.json [-o OUT]               # fill missing params from templates
karak run --builtin global --input DIR --out BASE    # run the standard flow
karak run --builtin tiled|tiled-rare ...             # tiled variants
karak run FLOW.json --input DIR --out BASE           # run a custom flow
karak run ... --set NODE.PARAM=VALUE                 # override any node param
karak run ... --no-qc                                # skip QC figure sinks
karak run --builtin stepwise --input DIR --out BASE  # the growing step-by-step flow (load + mask + denoise + normalize, for now)
karak run ... --plain                      # line output instead of the live dashboard
uv run --extra view karak view BASE [--mask CSV]    # open a run's cached load outputs in napari
karak run ... --no-cache                             # ignore the node cache
karak run ... --workers 0                  # parallel stages on all cores (same results)
karak run ... --cache-compression gzip     # cache file filter: lzf (default), gzip, none
karak run ... --ram-budget 8               # GB of outputs held between steps (default: half of available memory)
karak run ... --device cuda                # GPU paths where stages support it
karak run ... --gpu-budget GB              # device memory held between steps (default 80% free)
karak validate (FLOW.json | --builtin NAME)          # structural validation
karak schema                                         # stage palette as JSON
karak bench --builtin tiled --input DIR --out BASE \
    --config baseline --config workers=0   # per-node timing comparison
karak bench --compare a.json b.json        # cross-machine table
```

`python -m karak.flow {run|validate|schema}` mirrors the flow subcommands.

## Architecture

Three layers; each depends only on the one below it.

1. **Numeric core** — `io/`, `preprocessing/`, `clustering/`,
   `identification/`, `qc/`. Pure functions on numpy arrays. No knowledge of
   stages or flows. Settings arrive as explicit arguments or as the
   default-free dataclass bundles in `core_params.py` (no default values:
   every run value comes from the flow via the stage). `provenance.py`
   holds version info for run records and HDF5.
2. **Stages** — `stages/`. One small class per operation declaring typed
   `PARAMS` (name, type, default, bounds, help) and named input/output
   `Port`s; `apply(inputs, params)` calls the core. `@register` +
   package autoload make them discoverable; `stages.list_stages()` emits the
   JSON palette. Payloads (`stages/payloads.py`) are frozen dataclasses with
   `space`/`state` tags that ports type-check, plus HDF5 serialization for
   the cache.
3. **Flow** — `flow/`. `graph.py` (Node/Edge/Graph + JSON round-trip),
   `validate.py` (pure structural validation, shared by CLI and any GUI),
   `executor.py` (topo-sort, `{input}`/`{out}`/`{work}`/`{flow}` token
   resolution, content-addressed per-node caching, refcounted payload
   eviction with spill-to-cache for >256 MB arrays), `builtins.py` (the
   loads the shipped flows + `override_params`/`apply_device`),
   `complete.py` (fill missing params from stage templates, v1 → v2),
   `record.py` (per-run `{out}/runs/<time>/flow.json` + `run.json`),
   `schema.py` (JSON Schema for flow files → `docs/flow.schema.json`,
   drift-tested; regenerate with `uv run python -m karak.flow.schema`
   after changing any stage's params),
   `flows/*.json` (the builtin flows: complete JSON, the source of truth).

`cli/main.py` is the CLI; `cli/reporter.py` holds all Rich output;
`cli/runner.py` is only the packaged entry-point re-export.

### Key design rules

- **Params are data**: every knob is a declared `Param`; never read an
  undeclared params key.
- **Flows are complete**: a flow JSON lists every param of every node
  (format version 2). `Param` defaults are templates for `karak flow
  init/complete` and direct `Stage.run()` calls only; a flow run never
  fills a value from code, and JSON null is a value only where the template
  is null. Adding a param to a stage means existing flows need
  `karak flow complete`; regenerate the shipped flows the same way.
- **Payloads are immutable**: `apply()` returns new payloads via
  `.replace()`; never mutate inputs (the cache depends on this).
- **Caching replaces checkpoints**: stage outputs live in
  `{work}/cache/<recipe-hash>__<port>.h5`; the provenance HDF5 is written by
  the `export_h5` sink, not used as runtime state.
- **Rich stays in the CLI**: stages report progress through the duck-typed
  `Reporter` (`flow/events.py`); matplotlib uses the Agg backend.
- **Adding a stage**: drop a `@register`ed `Stage` subclass into
  `stages/` — autoload picks it up; add a parity test against the core
  function it wraps (see `tests/test_stage_parity.py`); give every Port a
  `help` string; regenerate the doc with
  `uv run python -m karak.stages.reference` (a drift test pins
  `docs/stage_reference.md` to the registry).
- **Notebook-facing APIs** kept outside the flows:
  `identification.fingerprint`, `storage.save/load_mineral_names`,
  `preprocessing.denoise.compare_denoisers`.
