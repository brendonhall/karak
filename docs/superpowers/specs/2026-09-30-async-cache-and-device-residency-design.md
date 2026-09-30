# Asynchronous cache writes and device-resident payloads

Date: 2026-09-30. Status: approved in conversation; implementation plan
to follow.

Supersedes the "Payloads stay numpy" rule of
`2026-09-18-acceleration-design.md` (section "Numeric core rules") for
stages that declare a `device` param. Everything else in that spec stands.

## Goal

A chain of GPU stages moves the data to the device once, passes
device-resident results from node to node, and never stalls on a cache
write between nodes. The cache remains the restart and inspection
mechanism; its cost leaves the critical path.

## Measurements that drove the design

NWA 4587, 6525 x 3990 x 19 float32 (1.98 GB), RTX 4090, 16 CPU cores,
31 GB RAM, btrfs on NVMe. From the 2026-09-29 and 2026-09-30 runs:

| Item | Seconds |
|---|---|
| Read the input cube from the gzip cache | 5.3 |
| Host to device, 2 GB | 1.1 |
| Bilateral kernel, 19 channels | 0.3 |
| Device to host, 2 GB | 0.2 |
| Write the output cube to the gzip cache (h5py default chunks) | about 35 |

The write with other HDF5 settings, same cube, chunks (512, 512, 19):

| Setting | Write | Read | Size |
|---|---|---|---|
| gzip level 4 (today) | 25.6 s | 4.5 s | 549 MB |
| gzip level 1 | 14.5 s | 4.2 s | 574 MB |
| lzf | 10.8 s | 5.8 s | 822 MB |
| none | 0.8 s | 1.1 s | 1979 MB |

Today the executor spills any payload over 256 MB (hard-wired in
`flow.executor.run`), so every consumer of a large payload reads it back
from the cache. `_payload_nbytes` counts only numpy arrays.

## Rulings

- GPU memory pressure: fall back to host RAM (and to the cache when RAM is
  short too), log one line, continue. A run always completes.
- Default cache compression: `lzf`, switchable from the CLI.
- GPU paths in scope: denoise, normalize, pca, hdbscan. PCA on the GPU is
  a CuPy port of the sklearn algorithm (exact tier), not cuML PCA.
- GPU tests run only on the user's machine; CI has no GPU
  (`skipif(not cuda_available())`).
- Approach: arrays live where they are and the executor places them
  (approach A below). Rejected: a side table of device mirrors keyed by
  (node, port) (two copies of every large payload, executor-aware
  stages); per-stage device caches (leak across runs, one node only).

## Section 1: asynchronous cache writer, format, RAM residency

### CacheWriter (`flow/cache.py`)

- One background thread, one FIFO queue.
- `submit(recipe, port, host_payload, summary, upstream, compression)`
  enqueues. The thread calls the existing `store_payload` (tmp file then
  `os.replace`, so `has_payload` only ever sees complete files) and
  `store_summary`.
- `wait_for(recipe, port)` blocks until that entry is on disk. `wait()`
  drains the queue. Both re-raise the first exception the thread hit.
- Each finished write is reported through `reporter.log`
  (`cache: dn.cube written in 10.8 s`). The thread never touches the
  reporter: it appends to a thread-safe list, and the executor drains
  that list on the main thread at every node boundary and at the end of
  the run (the Rich dashboard is not thread-safe). The run summary gains
  the total writer seconds.
- One thread suffices: the write is disk-bound and HDF5 compression runs
  in C outside the GIL, so the next node computes while the file is
  written.

### Executor

- After `stage.run`, for each output: take a host copy
  (`payload.to("cpu")`, a no-op for numpy), compute the summary from it,
  submit it to the writer, put the original payload in the store. The
  node finishes at once.
- The record lists the output file path at node finish, as today. The
  file may lag until the run ends. `karak view` already falls back to a
  cache scan when a listed file is missing, and the executor drains the
  writer before it returns, so a completed run's record is complete.
- A spilled payload that reloads from the cache first calls
  `wait_for` on its entry.
- A `finally` drains the writer on success, failure and Ctrl-C, so
  completed outputs reach the cache and the next invocation resumes from
  the crash point (the user guide's promise). A second Ctrl-C during the
  drain stops it; a leftover `.h5.tmp` is ignored by `has_payload` and
  overwritten by the next run.

### Format

- Every payload's `to_h5(group, compression)` takes the compression
  (`"lzf"`, `"gzip"`, `None`). Default `lzf`. Large datasets use chunks
  of (512, 512, C) for cubes and (512, 512) for images and masks.
- `karak run --cache-compression {lzf,gzip,none}` passes through
  `flow.executor.run(cache_compression=...)` to the writer.
- Reads are unchanged: HDF5 records the filter per dataset, so existing
  gzip cache files stay valid. Recipe hashes exclude the compression.

### RAM residency

- The per-payload spill threshold becomes a total budget. The store
  tracks the bytes it holds on the host. A new payload spills only when
  adding it would exceed the RAM budget.
- Default budget: half of `MemAvailable` (from `/proc/meminfo`) at run
  start. `karak run --ram-budget GB` overrides it; `run(ram_budget=...)`
  in bytes.
- With `--no-cache` nothing can spill, as today. The store logs one line
  when a payload spills (`store: dn.cube (1.98 GB) spilled to cache,
  budget 11.0 GB`).

## Section 2: placement and device memory

### Payloads

- `_Replaceable` gains `device` ("cpu" or "cuda", from the type of the
  first non-None array field) and `to(device)`, which returns the same
  payload with every array field moved (`cupy.asarray` to the device,
  `.get()` to the host), or `self` when already there. `None` fields stay
  `None`. Dataclass fields, `space` and `state` tags are unchanged, so
  port checks are unaffected.
- `summary()` and `to_h5()` are called only on host payloads (the
  executor guarantees it). They keep their numpy code.

### Executor placement

- A node's placement is its `device` param when the stage declares one,
  otherwise `cpu`.
- Before `apply()`, every input is moved to the node's placement. A CPU
  stage never sees a CuPy array. Outputs stay where the stage produced
  them.
- The record stores each output's placement next to its summary
  (`"device": "cuda"`).

### Device budget

- `_payload_nbytes` counts numpy and CuPy arrays, reported separately.
- The store keeps a device budget, default 80 % of free device memory at
  run start (`cupy.cuda.runtime.memGetInfo`), overridable with
  `karak run --gpu-budget GB` / `run(gpu_budget=...)`.
- When a device payload would exceed the device budget, the store moves
  it to host (`to("cpu")`), logs one line, and continues; the RAM rules
  then apply. A later GPU consumer gets it moved back by placement.
- Refcounting is unchanged. Device memory returns to the CuPy pool at
  release; the executor calls `cupy.get_default_memory_pool()
  .free_all_blocks()` once at the end of a run that used the device.
- The dashboard's memory line gains the device figure
  (`gpu 2.1 / 24 GB`) when the run has a CUDA node
  (`cli/memory.py` reads `memGetInfo` when cupy is importable).
- The cache writer always receives a host copy, so one device-to-host
  transfer per output (0.2 s per cube) sits on the main thread; the write
  does not.

## Section 3: stage contracts and the four GPU paths

### Contract

- `apply()` receives inputs already on the node's placement.
- A stage without a `device` param is unchanged and always sees numpy.
- A stage with `device="cuda"` receives CuPy arrays and returns CuPy
  arrays. It picks its array module from the data with the new helper
  `accel.xp(array)` (numpy for `np.ndarray`, cupy for `cupy.ndarray`) and
  never calls `cp.asarray` or `to_numpy` at its boundary.
- Core functions follow the same rule for their GPU paths: array in, the
  same kind of array out. CPU paths keep "numpy in, numpy out".
- New `accel` helpers: `xp(array)`, `device_of(array)`,
  `free_device_memory()`, `device_memory_info()`.

### denoise

- `bilateral_denoise_cube` drops its transfers. The mask fill, the joint
  guide and the re-zero run with `xp`; `bilateral_cupy` takes the device
  channel as now. Anisotropic diffusion on CUDA raises as today.
- The `on_channel` progress callback is unchanged.

### normalize

- Gains `Param("device", "str", "cpu", choices=("cpu", "cuda"))`.
- `zscore_normalize` becomes `xp` code: mineral-pixel mean and standard
  deviation per channel, `z = (x - mean) / std`, zeros outside the mask.
  `means` and `stds` are returned on the host (19 values) and ride on the
  payload as today.
- Exact tier. The reductions over 12 M float32 values differ from numpy's
  pairwise summation at about 1e-6 relative; parity is `atol=1e-4` on the
  z-scores and 1e-6 on means and stds.

### pca

- Gains `Param("device", "str", "cpu", choices=("cpu", "cuda"))`.
- CUDA path, sklearn's algorithm shape on CuPy: the same subsample indices
  from the host generator (`np.random.default_rng(random_state)`),
  centring, the C x C covariance in float64, `cupy.linalg.eigh`,
  descending order, sklearn's sign convention (the largest-magnitude
  loading of each component positive), then the projection of all mineral
  pixels on the device. Explained variance ratios come from the
  eigenvalues. `auto_n_components` and `select_components` are unchanged
  (they read the host ratios and slice the device features).
- cuML PCA is not used: loose tier, an extra API, no gain on a 19-column
  problem.
- Exact tier at `atol=1e-4` on the features, 1e-6 on the ratios.
- `PCAFeatures.features` (12 M x k float32, about 400 MB) stays on the
  device for HDBSCAN; `mineral_indices` too.

### hdbscan

- cuML accepts CuPy input; `run_hdbscan` drops its `cp.asarray` and
  returns labels and probabilities as device arrays. `noise_assign` is a
  CPU stage, so the executor moves them to the host. The tiled variant
  slices tiles on the device.

### Hash impact

- The new `device` params on normalize and pca change the recipe hashes
  of those nodes and everything downstream in every flow. The shipped
  flows are regenerated with `karak flow complete`; the golden hashes in
  `tests/test_recipe_stability.py` are updated; `docs/flow.schema.json`
  and `docs/stage_reference.md` are regenerated.
- `--device cuda` reaches the two new params through `apply_device`
  with no CLI change.

### Unchanged

QC sinks, export, view and every other stage stay numpy; the executor
converts their inputs.

## Section 4: testing and validation

### Writer and format (CPU-only, run in CI)

- `CacheWriter`: entries appear complete or not at all; `wait_for` blocks
  until the entry exists; a writer exception surfaces at the next wait;
  `wait()` drains in FIFO order.
- Executor: the next node starts before the previous output is on disk
  (a slow fake `store_payload` and a timeline of events); a spilled
  reload waits for its entry; a failing node still gets the earlier
  outputs written; Ctrl-C during a stage drains the queue and still exits
  130.
- Format: `to_h5` with `lzf`, `gzip` and `None` round-trips every payload
  class; a gzip file written before this change still loads;
  `--cache-compression` reaches the writer.
- RAM budget: payloads spill only when the total would exceed the budget
  (replaces `test_payload_store_spills_large_payloads`); `--ram-budget`
  reaches the store.

### Placement (CPU-only where possible)

- `payload.device` and `payload.to()` for every payload class, including
  `None` optional fields; `to("cpu")` on a numpy payload returns `self`.
- The executor moves inputs to a node's placement, with a fake device
  module in tests so CI covers the placement logic without a GPU; a CPU
  stage downstream of a fake-device stage receives numpy.
- Device budget fallback: a payload above the budget lands on the host
  and the log line appears.

### GPU parity (the user's RTX 4090 only)

- normalize: z-scores at `atol=1e-4`, means and stds at 1e-6.
- pca: features at `atol=1e-4`, explained variance ratios at 1e-6,
  component signs equal to sklearn's.
- denoise: the existing four-method parity test, now with device inputs.
- hdbscan: shape and label-count sanity on device features (loose tier).
- Chain: `stepwise` with `--device cuda` keeps the cube on the device from
  denoise to PCA; the record shows `"device": "cuda"` on those outputs.

### Real data, NWA 4587

- The `global` flow on `--device cuda` against the CPU run: per-node
  seconds, writer seconds, peak device memory, and the parity numbers
  above on the real cube.
- The same flow on the CPU with the new writer and `lzf`, against the
  2026-09-29 timings (`dn` 70.8 s on 16 cores, of which about 35 s was
  the write).

### Drift and docs

Regenerated schema and stage reference; updated golden hashes; user
guide paragraphs on the writer, compression, the two budgets and the
residency rule; CHANGELOG entries.

## Out of scope

- Multi-GPU, streams, or overlapping transfers with compute.
- GPU paths for noise_assign, refine, fingerprints, QC and export.
- Changing `method` defaults in the shipped flows.
- Reporting the scikit-image spatial table finding upstream.
