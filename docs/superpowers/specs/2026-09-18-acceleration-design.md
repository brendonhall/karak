# CPU/GPU acceleration for karak stages

Date: 2026-09-18
Status: approved (design review in chat, 2026-09-18)

## Goal

Add optional CPU multicore and CUDA GPU execution to the karak pipeline, one
stage at a time, with a benchmark command that measures each change. The
default serial CPU path stays byte-identical to today's behavior.

## Context and constraints

- Profiling on the NWA 4587 publication sample (13050x7980 px, 19 elements)
  showed: loader spends 50 s serial in read + invert + downsample; the
  colormap inversion itself is 0.52 s per image; the one-time LUT build is
  141 s but cached. Clustering is expected to dominate full runs; the bench
  command confirms before any stage is accelerated.
- Development machine has 12 CPU cores and no NVIDIA GPU. An RTX 4090 is
  reachable over SSH on a separate machine; host details arrive later. GPU
  code is written and unit-tested here (skipped without CUDA), validated and
  benchmarked there.
- Parity policy is tiered. Exact tier: elementwise and per-unit-parallel
  work (load, denoise, normalize, CPU tiling) must match the serial result,
  byte-identical for CPU parallelism, within float tolerance for GPU.
  Loose tier: cuML HDBSCAN and cuML KNN may produce structurally different
  results; their param help says so, and QC figures support comparison.
- The publication baseline is not touched: acceleration is opt-in, defaults
  reproduce current behavior.

## Design

### Two knobs, two homes

The rule: a knob that can change numeric results is a declared `Param`; a
knob that cannot is a runtime setting.

**`workers`** (CPU parallelism, never changes results):

- `flow.executor.run()` gains `workers: int | None = None`. `None` = serial
  (current behavior), `0` = all cores, `N` = N processes.
- The executor sets `stage.workers = workers` before `apply()`, the same
  mechanism as `stage.reporter`. `Stage` gains a class default
  `workers = None`.
- Never enters the recipe hash. Cached outputs stay valid across `workers`
  changes. A parity test pins byte-identical output for `workers=1` vs
  `workers=8`.

**`device`** (`"cpu"` | `"cuda"`, can change results):

- Declared as `Param("device", "str", "cpu", choices=["cpu", "cuda"])` only
  on stages that have a GPU path.
- Enters the recipe hash like any param: CPU and GPU outputs cache under
  different keys, and the provenance HDF5 records the device.

**CLI:**

- `karak run` gains `--workers N` and `--device cuda`. `--device` applies
  `override_params` only to nodes whose stage declares a `device` param, so
  it works on any flow. `--set node.device=cuda` targets one node. Both
  flags mirror into `python -m karak.flow run`.
- `device=cuda` without CuPy or without a visible GPU fails before the flow
  starts, with a clear error and an install hint, not mid-run. The check
  (`karak.accel.cuda_available()`) runs in the CLI pre-flight, next to the
  existing input checks. `flow/validate.py` stays purely structural: it
  validates that `device` values are among the declared choices, and never
  probes the environment.

### `karak/accel.py`

One new module holds the whole acceleration surface (about 80 lines):

- `cuda_available() -> bool`: CuPy imports and sees a device.
- `get_array_module(device)`: returns `numpy` or `cupy`; raises `StageError`
  with an install hint for unavailable `cuda`.
- `to_device(arr, device)` / `to_numpy(arr)`: transfers, no-ops on CPU.
- `resolve_workers(workers)`: `None` -> 1, `0` -> `os.cpu_count()`.

### Numeric core rules

Core functions stay pure, numpy in and numpy out. Two patterns:

1. CPU-parallel functions gain `workers: int = 1` and split their natural
   unit across a `ProcessPoolExecutor` (files, channels, tiles). The units
   are independent, so results are identical by construction.
2. GPU-capable functions gain `device: str = "cpu"` and compute through
   `xp = get_array_module(device)`, converting back with `to_numpy()`
   before returning. Payloads stay numpy; HDF5 serialization is unchanged.
   cuML paths (HDBSCAN, KNN) branch explicitly on `device` instead of the
   `xp` pattern because the estimator API differs from scikit-learn only in
   import path.

All GPU imports are lazy, inside functions.

### Dependencies

One optional extra in `pyproject.toml`: `karak[cuda]` with `cupy-cuda12x`,
`cuml-cu12`, `cucim-cu12`. The default dependency list is unchanged.

### Module-by-module paths

Ranked by measured or expected cost. The bench command gates each row: no
measured cost, no code.

| Module | CPU path (`workers`) | GPU path (`device`) | Parity |
|---|---|---|---|
| `io/loaders` | Process pool across files (read + invert + downsample per file). 50 s -> ~7 s expected on 12 cores. | None: PNG decode and PCIe dominate. | exact |
| `preprocessing/denoise` | Pool across the C channels (both cube denoisers already loop over channels). | CuPy/cuCIM bilateral on the whole cube. | exact |
| `clustering/tiling` | Pool across tiles (`run_tiled_hdbscan` already loops over tiles). Expected biggest win. | cuML HDBSCAN per tile. | CPU exact, GPU loose |
| `clustering/hdbscan_cluster` | Pass through `core_dist_n_jobs`. | cuML HDBSCAN. | GPU loose |
| `noise_assign`, `final_knn_assign` | Already `n_jobs=-1`; nothing to do. | cuML KNN, only if benchmarked as material. | GPU loose |
| `clustering/pca`, `preprocessing/compositional` | Benchmark first; BLAS already multithreads. | CuPy if justified. | exact |
| `qc/` figures | Possibly pool across sinks at executor level; deferred. | None. | n/a |

Excluded on purpose: `identification/fingerprint` (notebook-facing, cheap)
and the `mask` stage (morphology on one boolean image). Neither gets a knob.

### `karak bench`

The executor already times every node via
`reporter.node_finished(node_id, elapsed, cached_hit)`; bench wraps that.

```
karak bench --builtin tiled --input DIR --out BASE \
    --config baseline --config workers=0 --config workers=0,device=cuda \
    --repeats 3
karak bench --compare cpu_box.json gpu_box.json
```

- Each `--config` is a comma list of `workers=N` and `device=X`;
  `baseline` means serial CPU. `device=cuda` applies only to nodes that
  declare the param.
- Bench runs the flow once per config with the cache disabled and QC sinks
  skipped by default (`--qc` opts in). `--repeats N` keeps the per-node
  minimum.
- Output: `BASE_bench.json` (hostname, CPU count, GPU name, git revision,
  date, input path, per-config per-node seconds) plus a Rich table printed
  to the terminal: nodes as rows, configs as columns, speedup vs the first
  config. `--compare` renders the same table from two JSON files across
  machines.
- Placement: logic in `cli/bench.py`, subcommand registered in
  `cli/main.py`, mirrored in `python -m karak.flow`; the table rendering
  lives in `cli/reporter.py` per the repo rule.

## Testing

- Parity: loader and tiling produce byte-identical output for `workers=1`
  vs `workers=4` (synthetic data).
- GPU parity: exact-tier functions assert `np.allclose(cpu, gpu, atol=1e-5)`,
  marked `skipif(not cuda_available())`. Loose-tier functions get shape and
  sanity tests only.
- Bench smoke test: two configs on the synthetic tutorial data; assert the
  JSON schema and positive per-node times.
- Existing drift tests pin the new `device` param defaults; stage docs
  regenerate via `python -m karak.stages.reference`.

## Implementation order

1. `karak bench` subcommand (measurement before optimization).
2. Loader process pool (`workers`).
3. Denoise channel pool (CPU) + CuPy path (GPU).
4. Tiled clustering tile pool (CPU) + cuML HDBSCAN (GPU).
5. PCA / normalize / KNN, only where bench numbers justify.
6. GPU benchmark run on the 4090 host once SSH details arrive.

## Out of scope

- QC figure parallelism (deferred until bench shows it matters).
- Multi-GPU, distributed execution, and any change to flow JSON schemas.
- Driver or CUDA setup on the 4090 host.
