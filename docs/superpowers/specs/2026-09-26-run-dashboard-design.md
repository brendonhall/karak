# Live run dashboard for `karak run`

Date: 2026-09-26. Status: approved design, pending spec review.

## Goal

`karak run` prints one line when a node starts and one when it ends. It
shows no parameters, no progress inside a stage, and nothing about what a
stage produced. The goal is a live dashboard that shows, for every run on
any dataset:

- the run context (flow, directories, workers, device, cache, version),
- the resolved parameters of the running step,
- progress inside the step,
- the output of each finished step,
- cache status, memory use, and recent stage log lines.

The dashboard is built step by step alongside a growing flow. The first
step is the load stage (`load_elements`), tested on the NWA 4587 paper
sample in `/home/brendon/Dropbox/Projects/izawa/NWA_4587_data`.

Decisions taken with the user:

- Improve `karak run` itself, not a separate launcher script.
- Run partial pipelines through a small shipped flow that grows one node
  at a time, named `stepwise` so the name stays accurate as it grows.
- A full live dashboard (Rich `Live`), not a scrolling log.
- Show output summaries, memory use, cache status, and stage log lines.

## Out of scope

- An interactive stepper that pauses between steps.
- Run records or manifests written to disk.
- Stages other than `load_elements` emitting in-stage progress. They get
  the dashboard (parameters, outputs, cache, memory, logs) for free; each
  one gains progress events when it joins the `stepwise` flow.

## Architecture

The design keeps the existing layering: stages and the executor emit
events through the duck-typed `Reporter` in `flow/events.py`; Rich stays in
`cli/`.

### Reporter events (`flow/events.py`)

Existing methods stay: `node_started`, `node_finished`, `progress`, `log`.
New methods, all no-ops on `NullReporter`:

| Method | Payload | Emitted by |
|---|---|---|
| `run_started(info: RunInfo)` | flow name, input dir, out base, work dir, workers, device, cache on/off, karak version, ordered node list `[(node_id, stage_type)]` | executor, before the first node |
| `node_params(node_id, params: list[ParamValue])` | per param: name, resolved value, `is_default` flag | executor, before running or loading a node |
| `node_cache(node_id, recipe_hash: str, cached: bool, cache_dir: str)` | cache decision for the node | executor, after hashing |
| `node_outputs(node_id, summaries: dict[str, str])` | one summary line per output port | executor, after outputs exist (ran or cached) |
| `run_finished(summary: dict, seconds: float)` | executor summary dict | executor, after the last node |
| `node_failed(node_id, message: str)` | error text | executor, before raising `FlowError` |

`RunInfo` and `ParamValue` are small frozen dataclasses in
`flow/events.py`. The executor already calls `reporter.node_started` only
for nodes that run; `node_params`, `node_cache`, and `node_outputs` fire
for cached nodes too. Reporters written against the old protocol keep
working: the executor calls new methods through a helper that skips a
method the reporter does not define.

The karak version comes from `importlib.metadata.version("karak")`, since
`karak.__version__` is stale.

### Payload summaries (`stages/payloads.py`)

Each payload class gets `summary() -> str`, one line. Illustrative examples:

- `ElementCube 6525×3990×19 float32 1.98 GB space=RAW`
- `BseImage 6525×3990 float32 104 MB`
- `MaskSet valid 48.5% · mineral 41.2%`
- `Labels 12,495,787 px · 11 phases · state=CLEANED`

The executor calls `summary()` when present and falls back to the class
name.

### In-stage progress for the load stage

`load_element_maps` (`io/loaders.py`) gains an optional
`on_file: Callable[[int, int, str], None] | None = None`, called as
`on_file(done, total, element)` after each file is loaded, in both the
serial and the worker-pool paths. The numeric core stays free of reporter
knowledge. `LoadElementsStage.apply` passes a callback that forwards to
`self.reporter.progress(node_id, done, total, element)`. Before loading,
the stage logs one discovery line: file count, elements, BSE channel, and
excluded channels.

The stage needs its node id to report progress; the executor sets
`stage.node_id` next to the existing `stage.reporter` and `stage.workers`.

### Log capture (`cli/main.py`)

For the duration of a run, a `logging.Handler` on the `karak` logger
forwards records at INFO and above to `reporter.log(level, message)`. It
is attached in a `try/finally` and removed afterwards, including on
failure, so notebooks and tests see no side effects.

### Memory sampling (`cli/dashboard.py`)

A daemon thread samples once per second: current RSS from
`/proc/self/statm` and peak RSS from `resource.getrusage`. Only the
standard library is used. If `/proc` is missing, the memory line shows
peak only. With `--workers` other than 1, the memory line is labelled
"main process" because pool workers are not counted.

### Dashboard (`cli/dashboard.py`)

`DashboardReporter` implements the full reporter protocol and renders a
Rich `Live` layout, refreshed about 4 times per second:

```
┌ karak run  stepwise  karak 0.2.0 ─────────────────────────────┐
│ input  …/NWA_4587_data     out  output/nwa4587                │
│ workers 16   device cpu   cache on   mem 2.4 GB (peak 4.1 GB) │
├ steps ────────────────────────────────────────────────────────┤
│ src   load_elements   ▶ 14/20   0:41                          │
├ params: src ──────────────────────────────────────────────────┤
│ input_dir …/NWA_4587_data   colormap tima:jet                 │
│ header_trim_px 0   downsample_factor 2   (6 more at defaults) │
├───────────────────────────────────────────────────────────────┤
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  14/20  Si  0:41   │
├ log ──────────────────────────────────────────────────────────┤
│ Loaded element 'Sb' (inverted via tima:jet): shape (6525, …)  │
│ Loaded element 'Si' (inverted via tima:jet): shape (6525, …)  │
└───────────────────────────────────────────────────────────────┘
```

- Steps table: one row per node; status is pending, running with
  `done/total`, done with seconds, cached with a short (8-char) recipe
  hash, failed, or interrupted.
- Params panel: the running step's parameters, non-default values first
  and highlighted; when space runs short, defaults collapse into
  `(N more at defaults)`.
- Progress bar: the latest `progress` event of the running step; hidden
  for steps that send none.
- Log panel: the last 5 log lines.
- Finished steps show their output summaries under their row.

When the run ends, `Live` stops and a static summary stays on screen: the
steps table with times and cache status, every output summary, total
time, peak memory, and the cache directory.

### Plain mode

`--plain` forces the existing line-based `RichReporter`, extended to print
the new events as lines (run header, params, outputs, cache). `karak run`
also picks plain mode when stdout is not a terminal, so `nohup` and piped
runs give readable logs.

### The `stepwise` flow

A new built-in flow `stepwise`, defined in `flow/builtins.py` and shipped
as `flow/flows/stepwise.json` (round-trip tested like the others). It
starts with the single node `src` (`load_elements`, `input_dir={input}`)
and gains one node per step as the dashboard work continues. Run it with:

```bash
karak run --builtin stepwise --input DIR --out BASE
```

## Error handling

- A stage raises: the executor emits `node_failed`, the dashboard marks
  the step failed in red and stops `Live`, the `FlowError` message prints
  below the final panel, exit code 1.
- Ctrl-C: `Live` stops cleanly, the running step shows as interrupted,
  exit code 130. Cache entries of completed steps stay valid, since the
  executor writes them only after a step finishes.
- The log handler and memory thread are always torn down in `finally`.

## Testing

- `summary()` on every payload type.
- Executor events: a recording reporter checks the event order on a
  small synthetic flow; a second run checks that the node reports
  `cached=True` with the same recipe hash and still emits `node_outputs`.
- A reporter that lacks the new methods still runs a flow (backward
  compatibility).
- `on_file` fires once per loaded file with the element name, serial and
  with workers.
- `DashboardReporter` fed a recorded event sequence, rendered on a
  recording Rich console: the output contains the run header, the
  non-default params, the progress text, output summaries, and the final
  summary; a failure sequence shows the failed status.
- Plain-mode selection when stdout is not a terminal.
- `stepwise` JSON round-trip and validation, as for the other built-ins.
- Real data: one run of `karak run --builtin stepwise` on
  `NWA_4587_data`, with the screen output shown to the user before the
  work is called done.

## Files

- `src/karak/flow/events.py`: new methods, `RunInfo`, `ParamValue`.
- `src/karak/flow/executor.py`: emit new events, set `stage.node_id`,
  tolerant reporter calls.
- `src/karak/stages/payloads.py`: `summary()` per payload.
- `src/karak/stages/base.py`: `node_id` attribute.
- `src/karak/io/loaders.py`: `on_file` callback.
- `src/karak/stages/load.py`: forward progress, discovery log line.
- `src/karak/cli/dashboard.py`: new, `DashboardReporter` and memory sampler.
- `src/karak/cli/reporter.py`: plain-mode lines for the new events.
- `src/karak/cli/main.py`: `--plain`, TTY detection, log capture, exit codes.
- `src/karak/flow/builtins.py`, `src/karak/flow/flows/stepwise.json`.
- `docs/user_guide.md`, `CHANGELOG.md`, `CLAUDE.md` command list.
- Tests as listed above.
