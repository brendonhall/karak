# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow

Always push commits to `main` after committing.

## Project

Karak is an automated mineralogy pipeline for SEM-EDS (Scanning Electron Microscopy with Energy Dispersive Spectroscopy) elemental map stacks. It transforms jet-colormapped elemental map PNGs into mineral phase maps with chemical fingerprints. Python >=3.12, built with Hatchling.

## Commands

```bash
uv sync --group dev                 # install with test dependencies
uv run pytest                       # run the test suite

karak run --builtin global --input DIR --out BASE    # run the standard flow
karak run --builtin tiled|tiled-rare ...             # tiled variants
karak run FLOW.json --input DIR --out BASE           # run a custom flow
karak run ... --set NODE.PARAM=VALUE                 # override any node param
karak run ... --no-qc                                # skip QC figure sinks
karak run ... --no-cache                             # ignore the node cache
karak validate (FLOW.json | --builtin NAME)          # structural validation
karak schema                                         # stage palette as JSON

karak -c config.yaml                # legacy YAML mode (runs via the flow engine)
karak --test-mode                   # fast validation: 4 elements, 4x downsample
karak --emit-flow -c config.yaml    # print the equivalent flow JSON
karak --clean                       # delete previous HDF5 + cache
```

`python -m karak.flow {run|validate|schema}` mirrors the flow subcommands.

## Architecture

Three layers; each depends only on the one below it.

1. **Numeric core** — `io/`, `preprocessing/`, `clustering/`,
   `identification/`, `qc/`. Pure functions on numpy arrays. No knowledge of
   stages or flows.
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
   three standard flows + `flow_from_config` YAML shim + `override_params`),
   `flows/*.json` (the same flows as shipped JSON, round-trip tested).

`cli/main.py` is the CLI; `cli/reporter.py` holds all Rich output;
`cli/runner.py` is only the packaged entry-point re-export.

### Key design rules

- **Params are data**: every knob is a declared `Param`; never read an
  undeclared params key. A drift test pins stage defaults to the legacy
  Pydantic defaults in `config.py`.
- **Payloads are immutable**: `apply()` returns new payloads via
  `.replace()`; never mutate inputs (the cache depends on this).
- **Caching replaces checkpoints**: stage outputs live in
  `{work}/cache/<recipe-hash>__<port>.h5`; the provenance HDF5 is written by
  the `export_h5` sink, not used as runtime state.
- **Rich stays in the CLI**: stages report progress through the duck-typed
  `Reporter` (`flow/events.py`); matplotlib uses the Agg backend.
- **Adding a stage**: drop a `@register`ed `Stage` subclass into
  `stages/` — autoload picks it up; add a parity test against the core
  function it wraps (see `tests/test_stage_parity.py`).
- **Notebook-facing APIs** kept outside the flows:
  `identification.fingerprint`, `storage.save/load_mineral_names`,
  `preprocessing.denoise.compare_denoisers`.
