# Device-Resident Payloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Post-review changes (2026-10-02, PR #7):** the device-budget fallback covers outputs held between steps only; after a fallback the executor frees CuPy's pool before the next node; an out-of-memory error at placement fails the node with a `FlowError`. `--set NODE.device` overrides `--device`. See the spec's Section 2 and Rulings.

**Goal:** A chain of GPU stages moves the data to the device once and passes device-resident payloads from node to node, with normalize and pca joining denoise and hdbscan on the GPU.

**Architecture:** Payload array fields may hold numpy or CuPy arrays; `payload.device` and `payload.to(device)` move them. The executor places every input on the consuming node's device (its `device` param, else `cpu`) before `apply()`, keeps outputs where the stage produced them, gives the cache writer and the summaries a host copy, and enforces a device memory budget with host fallback. GPU stages and cores pick their array module from the data (`accel.xp`).

**Tech Stack:** Python 3.12, numpy, CuPy 14 (`cupy-cuda12x`, optional `cuda` extra), cuML 26.08 (HDBSCAN), scikit-learn (CPU PCA), pytest. GPU tests run only on the user's RTX 4090 machine (`skipif(not cuda_available())`); CI has no GPU.

**Spec:** `docs/superpowers/specs/2026-09-30-async-cache-and-device-residency-design.md`, sections 2, 3 and 4. Requires the async cache writer plan (`2026-09-30-async-cache-writer.md`) merged first: this plan's executor changes assume `CacheWriter`, `ram_budget` and `write_seconds` exist.

## Global Constraints

- Every change goes through a pull request from a branch off `main`; `uv run pytest` before each push; `CHANGELOG.md` updated in the same branch (CLAUDE.md).
- Payloads are immutable; `apply()` never mutates inputs; `to()` returns a new payload or `self`.
- GPU imports only inside functions; a plain install never imports cupy or cuml.
- Exact-tier parity: normalize `atol=1e-4` on z-scores, 1e-6 on means and stds; pca `atol=1e-4` on features, 1e-6 on explained variance ratios; denoise `atol=1e-5` as today. Loose tier (hdbscan): shape and sanity only.
- `summary()` and `to_h5()` are only ever called on host payloads; the executor guarantees it.
- `device` params enter the recipe hash. Adding `device` to normalize and pca changes the golden hashes of `nrm`, `pca`, and every downstream node in every flow; Task 8 updates them once.
- Rich stays in `cli/`; the writer thread never touches the reporter.
- Commit messages end with the session's attribution lines.

## Review Focus

1. `--device cuda` on a machine without CuPy, reached through `run()` directly (not the CLI pre-flight): placement must raise a `StageError` that names the node, not an `ImportError`. Pinned in Task 3 (`test_placement_without_cuda_names_the_node`).
2. Payloads with no array fields (`ClusterStats`, `Fingerprints`) or with `None` optional fields (`ElementCube.means`, `MaskSet.valid_mask`, `Labels.probabilities`): `device` is `cpu`, `to()` keeps `None`. Pinned in Task 2 (`test_payloads_without_arrays_are_cpu`, `test_to_keeps_none_fields`).
3. A cache hit feeding a CUDA node: the loaded output is a host payload and must be moved by placement. Pinned in Task 3 (`test_cache_hit_output_is_placed_on_the_consumer_device`).
4. A device budget smaller than one payload: the output lands on the host with a log line, and the next GPU consumer still works. Pinned in Task 3 (`test_device_budget_fallback_moves_the_output_to_host`).
5. A CPU stage downstream of a GPU stage receives numpy and its output is host-resident, so the cache writer never sees a device array. Pinned in Task 3 (`test_cpu_stage_downstream_of_a_device_stage_gets_numpy`).

---

### Task 1: `accel` helpers for array-type dispatch and device memory

**Files:**
- Modify: `src/karak/accel.py`
- Test: `tests/test_accel.py`

**Interfaces:**
- Produces:

```python
def is_device_array(arr) -> bool          # type(arr).__module__ starts with "cupy"
def device_of(arr) -> str                 # "cuda" for a device array, else "cpu"
def xp(arr)                               # cupy for a device array (import inside), else numpy
def to_host(arr)                          # alias of to_numpy
def device_memory_info() -> tuple[int, int] | None   # (free, total) bytes, None without CUDA
def free_device_memory() -> None          # cupy default pool free_all_blocks(); no-op without CUDA
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_accel.py`:

```python
class _FakeDeviceArray:
    """Stands in for cupy.ndarray: the module name is what accel checks."""

    __module__ = "cupy"

    def __init__(self, host):
        self.host = np.asarray(host)
        self.shape, self.dtype, self.nbytes = self.host.shape, self.host.dtype, self.host.nbytes

    def get(self):
        return self.host


def test_is_device_array_by_module_name():
    assert not accel.is_device_array(np.zeros(3))
    assert accel.is_device_array(_FakeDeviceArray(np.zeros(3)))
    assert not accel.is_device_array(None)
    assert not accel.is_device_array([1, 2])


def test_device_of():
    assert accel.device_of(np.zeros(3)) == "cpu"
    assert accel.device_of(_FakeDeviceArray(np.zeros(3))) == "cuda"


def test_xp_is_numpy_for_host_arrays():
    assert accel.xp(np.zeros(3)) is np


def test_xp_for_a_device_array_uses_get_array_module(monkeypatch):
    fake_module = object()
    monkeypatch.setattr(accel, "get_array_module", lambda device: fake_module)
    assert accel.xp(_FakeDeviceArray(np.zeros(3))) is fake_module


def test_to_host_brings_a_device_array_back():
    arr = _FakeDeviceArray(np.arange(3))
    np.testing.assert_array_equal(accel.to_host(arr), np.arange(3))


def test_device_memory_info_none_without_cuda(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    assert accel.device_memory_info() is None
    accel.free_device_memory()   # no-op, no error


@pytest.mark.skipif(not accel.cuda_available(), reason="no CUDA")
def test_device_memory_info_on_a_gpu():
    free, total = accel.device_memory_info()
    assert 0 < free <= total
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_accel.py -q`
Expected: FAIL with `AttributeError: module 'karak.accel' has no attribute 'is_device_array'` and similar.

- [ ] **Step 3: Implement the helpers**

Append to `src/karak/accel.py`:

```python
def is_device_array(arr) -> bool:
    """True for a CuPy array (checked by module name, so no import)."""
    return type(arr).__module__.split(".")[0] == "cupy"


def device_of(arr) -> str:
    return "cuda" if is_device_array(arr) else "cpu"


def xp(arr):
    """The array module of ``arr``: cupy for a device array, else numpy."""
    return get_array_module("cuda") if is_device_array(arr) else np


to_host = to_numpy


def device_memory_info() -> tuple[int, int] | None:
    """(free, total) bytes on GPU 0, or None without a usable CUDA stack."""
    if not cuda_available():
        return None
    import cupy

    free, total = cupy.cuda.runtime.memGetInfo()
    return int(free), int(total)


def free_device_memory() -> None:
    """Return the CuPy pool's cached blocks to the driver. No-op without CUDA."""
    if not cuda_available():
        return
    import cupy

    cupy.get_default_memory_pool().free_all_blocks()
```

Update the module docstring: "The whole GPU surface of karak lives here" still holds; add one line naming the array-type helpers.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_accel.py -q`
Expected: all pass (the GPU one skips without CUDA).

- [ ] **Step 5: Commit**

```bash
git add src/karak/accel.py tests/test_accel.py
git commit -m "feat: accel helpers for array-type dispatch and device memory"
```

---

### Task 2: `payload.device`, `payload.to()`, and `payload_nbytes`

**Files:**
- Modify: `src/karak/stages/payloads.py` (`_Replaceable`, lines 51-53; add `payload_nbytes` near `format_bytes`)
- Modify: `src/karak/flow/executor.py:39-45` (`_payload_nbytes` delegates)
- Test: `tests/test_payloads.py`, `tests/test_executor.py`

**Interfaces:**
- Consumes: `accel.is_device_array`, `accel.to_numpy`, `accel.get_array_module` from Task 1.
- Produces:

```python
class _Replaceable:
    def replace(self, **changes)
    @property
    def device(self) -> str            # "cuda" if any top-level array field is a device array, else "cpu"
    def to(self, device: str)          # new payload with every top-level array field moved; self when nothing moves

def payload_nbytes(payload) -> tuple[int, int]   # (host_bytes, device_bytes) over top-level array fields
```

Top-level means direct dataclass fields whose value is an `np.ndarray` or a device array. Nested containers (`TiledArtifacts.tile_results`, `Fingerprints.data`) are host-only by construction (CPU stages produce them) and are not walked.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_payloads.py`:

```python
import karak.accel as accel
from karak.stages.payloads import ClusterStats, MaskSet, payload_nbytes


class _FakeDeviceArray:
    __module__ = "cupy"

    def __init__(self, host):
        self.host = np.asarray(host)
        self.shape, self.dtype, self.nbytes = self.host.shape, self.host.dtype, self.host.nbytes

    def get(self):
        return self.host


class _FakeCupy:
    """Enough of cupy for to(): asarray wraps, .get() unwraps."""

    @staticmethod
    def asarray(arr):
        return arr if isinstance(arr, _FakeDeviceArray) else _FakeDeviceArray(arr)


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(accel, "get_array_module",
                        lambda device: _FakeCupy if device == "cuda" else np)


def test_host_payload_is_cpu_and_to_cpu_is_self():
    cube = _cube()
    assert cube.device == "cpu"
    assert cube.to("cpu") is cube


def test_to_cuda_moves_every_array_field_and_back(fake_cuda):
    cube = _cube().replace(means=np.ones(3, np.float32), stds=np.ones(3, np.float32))
    on_device = cube.to("cuda")
    assert on_device is not cube
    assert on_device.device == "cuda"
    assert isinstance(on_device.pixels, _FakeDeviceArray)
    assert isinstance(on_device.means, _FakeDeviceArray)
    assert on_device.element_names == cube.element_names    # non-array fields untouched
    assert on_device.to("cuda") is on_device
    back = on_device.to("cpu")
    assert back.device == "cpu"
    np.testing.assert_array_equal(back.pixels, cube.pixels)
    np.testing.assert_array_equal(back.means, cube.means)


def test_to_keeps_none_fields(fake_cuda):
    masks = MaskSet(mineral_mask=np.ones((2, 2), bool), valid_mask=None)
    on_device = masks.to("cuda")
    assert on_device.valid_mask is None
    assert isinstance(on_device.mineral_mask, _FakeDeviceArray)
    assert on_device.to("cpu").valid_mask is None


def test_payloads_without_arrays_are_cpu(fake_cuda):
    stats = ClusterStats(stats={"n": 1})
    assert stats.device == "cpu"
    assert stats.to("cuda") is stats


def test_payload_nbytes_splits_host_and_device(fake_cuda):
    cube = _cube()
    host, device = payload_nbytes(cube)
    assert (host, device) == (cube.pixels.nbytes, 0)
    host, device = payload_nbytes(cube.to("cuda"))
    assert (host, device) == (0, cube.pixels.nbytes)
    assert payload_nbytes(ClusterStats(stats={})) == (0, 0)
```

`_cube()` in this file builds a 3-channel cube (check its `names` default; adjust the `np.ones(3)` sizes to the channel count it uses).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_payloads.py -q`
Expected: FAIL with `ImportError: cannot import name 'payload_nbytes'` and `AttributeError: 'ElementCube' object has no attribute 'device'`.

- [ ] **Step 3: Implement**

In `src/karak/stages/payloads.py`, replace `_Replaceable`:

```python
class _Replaceable:
    """``replace`` for frozen dataclasses, plus host/device placement of
    the top-level array fields."""

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    def _array_fields(self):
        from karak.accel import is_device_array

        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray) or is_device_array(value):
                yield field.name, value

    @property
    def device(self) -> str:
        from karak.accel import is_device_array

        return "cuda" if any(is_device_array(v) for _, v in self._array_fields()) else "cpu"

    def to(self, device: str):
        """This payload with every array field on ``device``; ``self`` when
        nothing has to move."""
        from karak.accel import get_array_module, is_device_array, to_numpy

        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
        changes = {}
        for name, value in self._array_fields():
            if device == "cuda" and not is_device_array(value):
                changes[name] = get_array_module("cuda").asarray(value)
            elif device == "cpu" and is_device_array(value):
                changes[name] = to_numpy(value)
        return dataclasses.replace(self, **changes) if changes else self


def payload_nbytes(payload) -> tuple[int, int]:
    """(host bytes, device bytes) held by a payload's top-level array fields."""
    from karak.accel import is_device_array

    host = device = 0
    for _, value in payload._array_fields():
        if is_device_array(value):
            device += int(value.nbytes)
        else:
            host += int(value.nbytes)
    return host, device
```

In `src/karak/flow/executor.py`, replace `_payload_nbytes` with:

```python
def _payload_nbytes(payload) -> int:
    """Host bytes a payload holds (the RAM budget counts these)."""
    from karak.stages.payloads import payload_nbytes

    return payload_nbytes(payload)[0]
```

and drop the now-unused `dataclasses` import if nothing else in the module uses it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_payloads.py tests/test_executor.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/karak/stages/payloads.py src/karak/flow/executor.py tests/test_payloads.py
git commit -m "feat: payload.device and payload.to() move array fields between host and device"
```

---

### Task 3: Executor placement, device budget, and record placement

**Files:**
- Modify: `src/karak/flow/executor.py` (`PayloadStore`; `run` signature; `_execute` miss path, record call, end of run)
- Modify: `src/karak/flow/__main__.py` (`--gpu-budget` flag, settings, run call)
- Modify: `src/karak/cli/memory.py`, `src/karak/cli/dashboard.py:196-215` (device figure)
- Test: `tests/test_executor.py`, `tests/test_cli_run.py`, `tests/test_cli_dashboard.py`

**Interfaces:**
- Consumes: Task 1 helpers; Task 2 `to()`, `device`, `payload_nbytes`; the writer plan's `CacheWriter`, `ram_budget`.
- Produces:
  - `run(..., gpu_budget: int | None = None)`: `None` means 80 % of free device memory at run start when a node uses CUDA; never used otherwise.
  - `PayloadStore(consumers, reload=None, ram_budget=None, on_spill=None, gpu_budget=None, on_device_fallback=None)`; `held_device_bytes: int`.
  - Placement: before `apply()`, `inputs[port] = payload.to(placement)` where `placement = params.get("device", "cpu")`. A `StageError` from `to("cuda")` is re-raised as `FlowError("node 'dn' (denoise): device='cuda' ...")`.
  - The record's per-output entry gains `"device": "cpu" | "cuda"`.
  - Log lines: `store: dn.cube (1.98 GB) moved to host, gpu budget 19.6 GB`.
  - `karak run --gpu-budget GB`; record settings `"gpu_budget_gb"`.
  - `cli.memory.device_memory() -> tuple[int, int] | None` (used, total) and the dashboard header shows `gpu 2.1 / 24 GB` when `RunInfo.device` contains `cuda`.

- [ ] **Step 1: Write the failing executor tests**

Append to `tests/test_executor.py` (reuse `_FakeDeviceArray` and `_FakeCupy` by importing them from `tests/test_payloads.py`, or copy the two classes here; copying is fine):

```python
import numpy as np

import karak.accel as accel
from karak.stages.payloads import BseImage


class _FakeDeviceArray:
    __module__ = "cupy"

    def __init__(self, host):
        self.host = np.asarray(host)
        self.shape, self.dtype, self.nbytes = self.host.shape, self.host.dtype, self.host.nbytes

    def get(self):
        return self.host


class _FakeCupy:
    @staticmethod
    def asarray(arr):
        return arr if isinstance(arr, _FakeDeviceArray) else _FakeDeviceArray(arr)


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(accel, "get_array_module",
                        lambda device: _FakeCupy if device == "cuda" else np)
    monkeypatch.setattr(accel, "device_memory_info", lambda: (10 * 2**30, 24 * 2**30))
    monkeypatch.setattr(accel, "free_device_memory", lambda: None)


SEEN: dict = {}


class FakeImageSource(Stage):
    id = "fake_image_source"
    label = "Fake image source"
    OUTPUTS = [Port("bse")]
    PARAMS = [Param("n", "int", 4)]

    def apply(self, inputs, params):
        return {"bse": BseImage(pixels=np.ones((params["n"], params["n"]), np.float32))}


class FakeDeviceDouble(Stage):
    """Doubles the image; on cuda it must receive and return device arrays."""
    id = "fake_device_double"
    label = "Fake device double"
    INPUTS = [Port("bse")]
    OUTPUTS = [Port("bse")]
    PARAMS = [Param("device", "str", "cpu", choices=("cpu", "cuda"))]

    def apply(self, inputs, params):
        bse = inputs["bse"]
        SEEN[self.node_id] = accel.device_of(bse.pixels)
        if params["device"] == "cuda":
            assert isinstance(bse.pixels, _FakeDeviceArray)
            return {"bse": bse.replace(pixels=_FakeDeviceArray(bse.pixels.host * 2))}
        return {"bse": bse.replace(pixels=bse.pixels * 2)}


class FakeHostSum(Stage):
    id = "fake_host_sum"
    label = "Fake host sum"
    INPUTS = [Port("bse")]
    OUTPUTS = [Port("num")]
    PARAMS = []

    def apply(self, inputs, params):
        pixels = inputs["bse"].pixels
        SEEN[self.node_id] = type(pixels).__name__
        assert isinstance(pixels, np.ndarray)
        return {"num": ClusterStats(stats={"value": float(pixels.sum())})}


@pytest.fixture(autouse=True)
def _device_stages():
    for cls in (FakeImageSource, FakeDeviceDouble, FakeHostSum):
        registry.register(cls)      # spell as the existing _fake_stages fixture does
    SEEN.clear()
    yield


def _device_chain(device="cuda", n=4):
    return complete(Graph(name="dev", nodes=(
        Node(id="src", type="fake_image_source", params={"n": n}),
        Node(id="d1", type="fake_device_double", params={"device": device}),
        Node(id="d2", type="fake_device_double", params={"device": device}),
        Node(id="s", type="fake_host_sum", params={}),
        Node(id="out", type="fake_sink", params={"out": "{out}"}),
    ), edges=(
        Edge(id="e1", src=Endpoint("src", "bse"), dst=Endpoint("d1", "bse")),
        Edge(id="e2", src=Endpoint("d1", "bse"), dst=Endpoint("d2", "bse")),
        Edge(id="e3", src=Endpoint("d2", "bse"), dst=Endpoint("s", "bse")),
        Edge(id="e4", src=Endpoint("s", "num"), dst=Endpoint("out", "num")),
    )))


def test_inputs_are_placed_on_the_node_device_and_outputs_stay_there(tmp_path, fake_cuda):
    run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))
    assert SEEN["d1"] == "cuda" and SEEN["d2"] == "cuda"   # moved once, then stayed
    assert SEEN["s"] == "ndarray"                            # CPU consumer got numpy
    assert RECORD[-1][1] == 4 * 4 * 4.0                      # 1 * 2 * 2 summed over 16 px


def test_cpu_stage_downstream_of_a_device_stage_gets_numpy(tmp_path, fake_cuda):
    """Also: the cache writer only ever sees host payloads."""
    import h5py

    run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))
    for path in (tmp_path / "w" / "cache").glob("*__bse.h5"):
        with h5py.File(path) as fh:
            assert fh["payload"]["pixels"].shape == (4, 4)


def test_record_stores_each_output_placement(tmp_path, fake_cuda):
    import json

    from karak.flow.record import RunRecord

    graph = _device_chain()
    record = RunRecord(str(tmp_path / "o"), graph, argv=[], source="t",
                       tokens={}, settings={}, overrides={})
    run(graph, out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"), record=record)
    data = json.loads((record.path / "run.json").read_text())
    assert data["nodes"]["d1"]["outputs"]["bse"]["device"] == "cuda"
    assert data["nodes"]["src"]["outputs"]["bse"]["device"] == "cpu"


def test_cache_hit_output_is_placed_on_the_consumer_device(tmp_path, fake_cuda):
    """Outputs loaded from the cache are host payloads; placement moves them."""
    run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))
    SEEN.clear()
    run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))
    assert "d1" not in SEEN and "d2" not in SEEN      # cache hits
    assert SEEN["s"] == "ndarray"                      # host payload from disk, placed on cpu
    SEEN.clear()
    run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"),
        cache=False)
    assert SEEN["d1"] == "cuda" and SEEN["d2"] == "cuda"


def test_placement_without_cuda_names_the_node(tmp_path, monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    with pytest.raises(FlowError, match=r"node 'd1' \(fake_device_double\).*karak\[cuda\]"):
        run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))


def test_device_budget_fallback_moves_the_output_to_host(tmp_path, fake_cuda):
    class Logger:
        def __init__(self):
            self.lines = []

        def log(self, level, msg):
            self.lines.append(msg)

        def __getattr__(self, name):
            return lambda *a, **k: None

    reporter = Logger()
    run(_device_chain(n=64), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"),
        reporter=reporter, gpu_budget=1024)
    assert SEEN["d2"] == "cuda"       # moved back to the device for the consumer
    assert any(line.startswith("store: d1.bse (") and "moved to host" in line
               for line in reporter.lines)


def test_gpu_budget_defaults_to_80_percent_of_free_memory(tmp_path, fake_cuda):
    import karak.flow.executor as executor

    seen = {}
    real = executor.PayloadStore

    class Spy(real):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    executor.PayloadStore = Spy
    try:
        run(_device_chain(), out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))
    finally:
        executor.PayloadStore = real
    assert seen["gpu_budget"] == int(0.8 * 10 * 2**30)


def test_no_gpu_budget_without_a_cuda_node(tmp_path, fake_cuda):
    import karak.flow.executor as executor

    seen = {}
    real = executor.PayloadStore

    class Spy(real):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    executor.PayloadStore = Spy
    try:
        run(_device_chain(device="cpu"), out_base=str(tmp_path / "o"),
            work_dir=str(tmp_path / "w"))
    finally:
        executor.PayloadStore = real
    assert seen["gpu_budget"] is None
```

Append to `tests/test_cli_run.py`:

```python
def test_gpu_budget_flag_is_gigabytes(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update(kwargs)
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(executor, "run", fake_run)
    out = tmp_path / "o"
    assert main(["run", "--builtin", "stepwise", "--out", str(out), "--plain",
                 "--gpu-budget", "2"]) == 0
    assert seen["gpu_budget"] == 2 * 2**30
    data = json.loads((out / "runs" / "latest" / "run.json").read_text())
    assert data["settings"]["gpu_budget_gb"] == 2.0
```

Append to `tests/test_cli_dashboard.py` (look at how that file builds a `DashboardReporter` and a `RunInfo`; copy its helper):

```python
def test_header_shows_the_gpu_figure_for_cuda_runs(monkeypatch):
    import karak.cli.memory as memory

    monkeypatch.setattr(memory, "device_memory", lambda: (2_100_000_000, 24_000_000_000))
    reporter = _reporter()                         # the file's helper
    reporter.run_started(_info(device="cuda"))     # the file's helper, with device
    text = reporter._memory_text()
    assert "gpu 2.1 / 24.0 GB" in text
    reporter.run_started(_info(device="cpu"))
    assert "gpu" not in reporter._memory_text()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_executor.py tests/test_cli_run.py tests/test_cli_dashboard.py -q -k "placed or placement or downstream or record_stores or cache_hit or fallback or gpu_budget or gpu_figure"`
Expected: FAIL. `SEEN["d1"] == "cuda"` fails (today the stage sees numpy), `gpu_budget` is an unexpected keyword, the argparse flag is unknown, `memory.device_memory` does not exist.

- [ ] **Step 3: Implement**

`src/karak/cli/memory.py`, append:

```python
def device_memory() -> tuple[int, int] | None:
    """(used, total) bytes on GPU 0, or None without a usable CUDA stack."""
    from karak.accel import device_memory_info

    info = device_memory_info()
    if info is None:
        return None
    free, total = info
    return total - free, total
```

`src/karak/cli/dashboard.py`, in `_memory_text` (line 196), after building the host text:

```python
        if info is not None and "cuda" in info.device:
            from karak.cli.memory import device_memory

            gpu = device_memory()
            if gpu is not None:
                used, total = gpu
                text += f"   gpu {_gb(used)} / {_gb(total)}"
        return text
```

(restructure the method so both branches assign `text` and fall through to this block; `_gb` is the file's existing helper and prints one decimal with the unit, so the assertion reads `gpu 2.1 GB / 24.0 GB`; adjust the test string to match `_gb`'s exact output once seen.)

`src/karak/flow/executor.py`:

1. `PayloadStore.__init__` gains `gpu_budget: int | None = None, on_device_fallback=None`; add `self.held_device_bytes = 0`, `self._gpu_budget = gpu_budget`, `self._on_device_fallback = on_device_fallback`, and `self._device_sizes: dict = {}`.

2. `PayloadStore.put`, before the host-budget check:

```python
        from karak.stages.payloads import payload_nbytes

        host_bytes, device_bytes = payload_nbytes(payload)
        if device_bytes and self._gpu_budget is not None \
                and self.held_device_bytes + device_bytes > self._gpu_budget:
            payload = payload.to("cpu")
            if self._on_device_fallback is not None:
                self._on_device_fallback(node, port, device_bytes, self._gpu_budget)
            host_bytes, device_bytes = payload_nbytes(payload)
        nbytes = host_bytes
```

and account `self.held_device_bytes += device_bytes` next to `self.held_bytes += nbytes` (store `self._device_sizes[key] = device_bytes`; subtract both in `release`). The RAM budget check uses `host_bytes` only.

3. `run()` gains `gpu_budget: int | None = None` and passes it to `_execute`.

4. In `_execute`, the `devices` set is already computed before `run_started`. After it:

```python
    uses_cuda = "cuda" in devices
    if uses_cuda and gpu_budget is None:
        from karak.accel import device_memory_info

        info = device_memory_info()
        gpu_budget = None if info is None else int(0.8 * info[0])

    def _on_device_fallback(node, port, nbytes, budget):
        _emit(reporter, "log", "info",
              f"store: {node}.{port} ({format_bytes(nbytes)}) moved to host, "
              f"gpu budget {format_bytes(budget)}")
```

and pass `gpu_budget=gpu_budget if uses_cuda else None, on_device_fallback=_on_device_fallback` to `PayloadStore(...)`. Note `run_started` must still be emitted before any node runs; keep the order: compute `devices`, compute the budget, build the store, emit `run_started`.

5. Placement in the miss path, replacing the `inputs = {...}` comprehension:

```python
            placement = str(params.get("device", "cpu"))
            inputs = {}
            for e in graph.in_edges(node_id):
                payload = store.get(e.src.node, e.src.port)
                try:
                    inputs[e.dst.port] = payload.to(placement)
                except StageError as exc:
                    _emit(reporter, "node_failed", node_id, str(exc))
                    if record is not None:
                        record.node(node_id, status="failed", error=str(exc))
                    raise FlowError(f"node {node_id!r} ({node.type}): {exc}") from exc
```

Import `StageError` from `karak.stages.base` at the top of the module. Payloads with no array fields return `self` from `to()`, so sinks and stats are unaffected.

6. Host copies for the writer and the summaries (this replaces the writer plan's `summaries = {name: _summarize(p) ...}` line):

```python
            host_outputs = {name: p.to("cpu") for name, p in outputs.items()}
            summaries = {name: _summarize(p) for name, p in host_outputs.items()}
            if cache and not is_sink:
                upstream = _upstream_recipes(graph, node_id, hashes)
                for port_name, payload in host_outputs.items():
                    writer.submit(node_hash, port_name, payload,
                                  summaries[port_name], upstream,
                                  label=f"{node_id}.{port_name}")
```

`store.put(node_id, port_name, payload)` keeps putting the original (device) `outputs`.

7. Record placement: keep a `placements = {port: p.device for port, p in outputs.items()}` (cache hits: all `"cpu"`) and add `"device": placements.get(port, "cpu")` to each output entry in the `record.node(... outputs={...})` call.

8. At the end of `_execute`, in the `finally` after the writer drains:

```python
        if uses_cuda:
            from karak.accel import free_device_memory

            free_device_memory()
```

`src/karak/flow/__main__.py`: after `--ram-budget`:

```python
    run_parser.add_argument(
        "--gpu-budget", type=float, default=None, metavar="GB",
        help="Device memory for stage outputs held between GPU steps "
             "(default: 80%% of free device memory at start).",
    )
```

Add `"gpu_budget_gb": args.gpu_budget` to the record settings and `gpu_budget=None if args.gpu_budget is None else int(args.gpu_budget * 2**30)` to the `run_flow(...)` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_executor.py tests/test_cli_run.py tests/test_cli_dashboard.py tests/test_run_record.py -q`
Expected: all pass.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/executor.py src/karak/flow/__main__.py src/karak/cli/memory.py src/karak/cli/dashboard.py tests/test_executor.py tests/test_cli_run.py tests/test_cli_dashboard.py
git commit -m "feat: executor places inputs on each node's device; device budget with host fallback"
```

---

### Task 4: denoise accepts device inputs

**Files:**
- Modify: `src/karak/preprocessing/denoise.py` (`_joint_guide`, `bilateral_denoise_cube` CUDA branch, `denoise_cube` docstring)
- Test: `tests/test_denoise_gpu.py`, `tests/test_denoise_workers.py`

**Interfaces:**
- Consumes: `accel.xp`, `accel.is_device_array` (Task 1).
- Produces: `bilateral_denoise_cube(cube, mask, ..., device=)` accepts numpy or CuPy `cube`, `mask`, `guide`; with `device="cuda"` it returns a CuPy array; with `device="cpu"` and CuPy inputs it raises `StageError("device='cpu' received a device array; the executor places inputs")`. `denoise_cube` the same.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_denoise_gpu.py`:

```python
@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("method", ["bilateral", "bilateral_sym",
                                    "joint_bilateral_total", "joint_bilateral_bse"])
def test_device_inputs_stay_on_the_device_and_match_cpu(method):
    import cupy as cp

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    guide = np.random.default_rng(9).random(cube.shape[:2], dtype=np.float32)
    cfg = denoise_cfg(method=method, sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    cpu = denoise_cube(cube, mask, cfg, guide=guide)
    gpu = denoise_cube(cp.asarray(cube), cp.asarray(mask), cfg,
                       guide=cp.asarray(guide), device="cuda")
    assert isinstance(gpu, cp.ndarray)
    assert gpu.dtype == cp.float32
    np.testing.assert_allclose(cpu, cp.asnumpy(gpu), atol=1e-5)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_host_inputs_on_cuda_still_work_and_return_a_device_array():
    import cupy as cp

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cfg = denoise_cfg(method="bilateral", sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    out = denoise_cube(cube, mask, cfg, device="cuda")
    assert isinstance(out, cp.ndarray)
```

Append to `tests/test_denoise_workers.py`:

```python
def test_cpu_path_rejects_a_device_array_with_a_clear_error():
    from conftest import denoise_cfg
    from karak.preprocessing.denoise import denoise_cube
    from karak.stages.base import StageError

    class FakeDeviceArray:
        __module__ = "cupy"
        shape = (4, 4, 2)

    cfg = denoise_cfg(method="bilateral", sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    with pytest.raises(StageError, match="device array"):
        denoise_cube(FakeDeviceArray(), np.ones((4, 4), bool), cfg)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_denoise_gpu.py tests/test_denoise_workers.py -q`
Expected: the GPU tests fail on `isinstance(gpu, cp.ndarray)` (today the function returns numpy) and the CPU test fails because no `StageError` is raised (skimage raises something else, or nothing).

- [ ] **Step 3: Implement**

In `src/karak/preprocessing/denoise.py`:

`_joint_guide` becomes array-module neutral:

```python
def _joint_guide(cube, mask, method, guide):
    """The mask-filled guide image for a joint method, or None. Runs on
    whatever array module ``cube`` lives on."""
    from karak.accel import xp as _xp

    if method not in JOINT_METHODS:
        return None
    if method == "joint_bilateral_total":
        guide = cube.sum(axis=-1)
    elif guide is None:
        from karak.errors import StageError
        raise StageError(
            "method='joint_bilateral_bse' needs the BSE image: connect the "
            "edge src.bse -> dn.bse (the denoise stage's optional 'bse' port)"
        )
    xp = _xp(cube)
    guide = xp.ascontiguousarray(xp.asarray(guide), dtype=xp.float32).copy()
    guide[~mask] = guide[mask].mean()
    return guide
```

In `bilateral_denoise_cube`, replace the CUDA branch's transfers:

```python
    from karak.accel import is_device_array

    if device == "cpu" and (is_device_array(cube) or is_device_array(mask)):
        from karak.errors import StageError
        raise StageError(
            "device='cpu' received a device array; the executor places "
            "inputs on the stage's device, so run this stage with device='cuda'"
        )
    if device == "cuda":
        from karak.accel import get_array_module
        from karak.preprocessing.bilateral import bilateral_cupy

        cp = get_array_module(device)  # raises StageError without CUDA
        gpu_cube = cp.asarray(cube)          # no-op for a device array
        gpu_mask = cp.asarray(mask)
        gpu_guide = None if guide is None else cp.asarray(guide)
        out = cp.zeros_like(gpu_cube)
        for i in range(C):
            channel = gpu_cube[:, :, i].copy()
            channel[~gpu_mask] = cp.nanmean(channel[gpu_mask])
            out[:, :, i] = bilateral_cupy(
                channel, channel if gpu_guide is None else gpu_guide,
                sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                exact_skimage=(method == "bilateral"),
            )
            if on_channel is not None:
                on_channel(i + 1, C, i)
        out[~gpu_mask] = 0.0
        logger.info(...)   # unchanged
        return out.astype(cube.dtype)        # stays on the device
```

Move the `guide = _joint_guide(cube, mask, method, guide)` call so it runs after the device check (it already runs before the branch; keep it there, it is now module-neutral). `cube.dtype` is available on both array types.

Update the `denoise_cube` docstring `device` entry: "cpu or cuda; with cuda the result is a CuPy array and CuPy inputs are accepted".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra cuda pytest tests/test_denoise_gpu.py tests/test_denoise_workers.py tests/test_bilateral.py -q` (on the GPU machine) and `uv run pytest -q`.
Expected: all pass; on CI the GPU tests skip.

- [ ] **Step 5: Commit**

```bash
git add src/karak/preprocessing/denoise.py tests/test_denoise_gpu.py tests/test_denoise_workers.py
git commit -m "feat: denoise takes and returns device arrays on cuda"
```

---

### Task 5: normalize on the GPU

**Files:**
- Modify: `src/karak/preprocessing/compositional.py:20-66` (`zscore_normalize`), `src/karak/stages/normalize.py:27-43`
- Test: `tests/test_normalize_gpu.py` (new), plus the existing normalize tests must stay green (`grep -l zscore_normalize tests/`)

**Interfaces:**
- Consumes: `accel.xp`, `accel.to_numpy`.
- Produces: `zscore_normalize(cube, mask) -> (normalized, means, stds)` works for numpy and CuPy `cube`/`mask`; `normalized` is on the input's device; `means` and `stds` are always numpy `(C,)` float32. `NormalizeStage.PARAMS` gains `Param("device", "str", "cpu", "Device", "cpu or cuda (needs karak[cuda])", choices=("cpu", "cuda"))`; the stage reads nothing from it (placement is the executor's job), but the param must exist so the executor places the inputs and the hash separates the two paths.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_normalize_gpu.py`:

```python
"""z-score normalization on the device: exact tier."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.accel import cuda_available
from karak.preprocessing.compositional import zscore_normalize
from karak.stages.normalize import NormalizeStage


def test_normalize_declares_a_device_param():
    names = [p.name for p in NormalizeStage.PARAMS]
    assert names == ["method", "device"]
    device = NormalizeStage.PARAMS[1]
    assert device.default == "cpu" and device.choices == ("cpu", "cuda")


def test_means_and_stds_are_host_arrays_on_cpu():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    normalized, means, stds = zscore_normalize(cube, mask)
    assert isinstance(means, np.ndarray) and isinstance(stds, np.ndarray)
    assert normalized.dtype == np.float32


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_device_zscore_matches_cpu():
    import cupy as cp

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cpu, means, stds = zscore_normalize(cube, mask)
    gpu, gmeans, gstds = zscore_normalize(cp.asarray(cube), cp.asarray(mask))
    assert isinstance(gpu, cp.ndarray) and gpu.dtype == cp.float32
    assert isinstance(gmeans, np.ndarray) and isinstance(gstds, np.ndarray)
    np.testing.assert_allclose(cp.asnumpy(gpu), cpu, atol=1e-4)
    np.testing.assert_allclose(gmeans, means, atol=1e-6)
    np.testing.assert_allclose(gstds, stds, atol=1e-6)
    assert not cp.asnumpy(gpu)[~mask].any()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_normalize_gpu.py -q`
Expected: `test_normalize_declares_a_device_param` fails (`["method"]`); the GPU test fails on `isinstance(gpu, cp.ndarray)` or inside numpy code on a CuPy array.

- [ ] **Step 3: Implement**

`zscore_normalize` in `src/karak/preprocessing/compositional.py`:

```python
def zscore_normalize(cube, mask):
    """Per-channel z-score normalization over mineral pixels.

    Works on numpy or CuPy arrays; the normalized cube comes back on the
    input's device, ``means`` and ``stds`` always as numpy ``(C,)`` float32.
    (docstring body otherwise unchanged)
    """
    from karak.accel import to_numpy, xp as _xp

    xp = _xp(cube)
    H, W, C = cube.shape
    mineral_data = cube[mask]  # (N, C)

    means = mineral_data.mean(axis=0)
    stds = mineral_data.std(axis=0)

    stds_safe = stds.copy()
    zero_std = stds_safe < 1e-10
    if bool(zero_std.any()):
        n_zero = int(zero_std.sum())
        logger.warning(
            "%d channels have near-zero std -- setting std to 1.0 to avoid division by zero",
            n_zero,
        )
        stds_safe[zero_std] = 1.0

    normalized = xp.zeros((H, W, C), dtype=xp.float32)
    normalized[mask] = ((mineral_data - means) / stds_safe).astype(xp.float32)

    logger.info(...)   # keep the existing log line, with to_numpy on anything it formats
    return normalized, to_numpy(means).astype(np.float32), to_numpy(stds).astype(np.float32)
```

Keep the existing tail of the function (any logging after the assignment) and make sure any value it formats goes through `float(...)` or `to_numpy`.

In `src/karak/stages/normalize.py`, add to `PARAMS`:

```python
        Param("device", "str", "cpu", "Device",
              "cpu or cuda (the executor moves the inputs; needs karak[cuda])",
              choices=("cpu", "cuda")),
```

`apply` is unchanged; `cube.replace(pixels=normalized, ...)` carries a device array when placed on cuda.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_normalize_gpu.py -q` and `uv run pytest -q`.
Expected: `test_normalize_gpu.py` passes (the GPU test skips on CI). The full suite now fails in `tests/test_recipe_stability.py`, `tests/test_flow_schema.py` (`test_builtin_flows_conform`), `tests/test_builtins.py::test_shipped_flows_are_complete_version_2`, and the two drift tests for `docs/stage_reference.md` and `docs/flow.schema.json`: expected until Task 8. Confirm those are the only failures.

- [ ] **Step 5: Commit**

```bash
git add src/karak/preprocessing/compositional.py src/karak/stages/normalize.py tests/test_normalize_gpu.py
git commit -m "feat: normalize runs on the device; device param on the stage"
```

---

### Task 6: PCA on the GPU

**Files:**
- Modify: `src/karak/clustering/pca.py` (`fit_pca`; add `PCAModel`, `_covariance_pca`), `src/karak/stages/pca.py:36-77`
- Test: `tests/test_pca_gpu.py` (new); existing PCA tests stay green (`grep -l fit_pca tests/`)

**Interfaces:**
- Consumes: `accel.xp`, `accel.to_numpy`.
- Produces:

```python
@dataclass(frozen=True)
class PCAModel:                       # what fit_pca returns on the device path
    components_: np.ndarray           # (k, C) float64, host
    mean_: np.ndarray                 # (C,) float64, host
    explained_variance_ratio_: np.ndarray   # (k,) float64, host

def _covariance_pca(spectra, n_components, xp) -> tuple[components, mean, evr]  # arrays on xp
def fit_pca(normalized_cube, mineral_mask, config) -> (model, features, mineral_indices)
    # numpy inputs: sklearn model as today
    # CuPy inputs: PCAModel; features (N, k) CuPy float32; mineral_indices (N, 2) CuPy int32
```

`PCAStage.PARAMS` gains `Param("device", "str", "cpu", "Device", ..., choices=("cpu", "cuda"))` after `random_state`. `PCAFeatures.explained_variance_ratio` stays a host array; `features` and `mineral_indices` are device arrays on cuda.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pca_gpu.py`:

```python
"""PCA on the device: a CuPy port of sklearn's covariance path, exact tier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA

from conftest import make_synthetic_scene, pca_cfg
from karak.accel import cuda_available
from karak.clustering.pca import _covariance_pca, fit_pca
from karak.stages.pca import PCAStage


def _spectra():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2000, 4)).astype(np.float32)
    mix = np.array([[1, 0.5, 0, 0], [0, 1, 0.3, 0], [0, 0, 1, 0.2], [0.1, 0, 0, 1]], np.float32)
    return base @ mix


def test_pca_declares_a_device_param():
    names = [p.name for p in PCAStage.PARAMS]
    assert names[-1] == "device"
    assert PCAStage.PARAMS[-1].choices == ("cpu", "cuda")


def test_covariance_pca_matches_sklearn_on_numpy():
    """The port is checked against sklearn on the CPU, so CI covers it."""
    spectra = _spectra()
    components, mean, evr = _covariance_pca(spectra, 4, np)
    ref = PCA(n_components=4, random_state=42).fit(spectra)
    np.testing.assert_allclose(mean, ref.mean_, atol=1e-6)
    np.testing.assert_allclose(evr, ref.explained_variance_ratio_, atol=1e-6)
    np.testing.assert_allclose(components, ref.components_, atol=1e-5)   # signs included
    ours = (spectra - mean) @ components.T
    np.testing.assert_allclose(ours, ref.transform(spectra), atol=1e-4)


def test_fit_pca_on_numpy_still_returns_an_sklearn_model():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    model, features, idx = fit_pca(cube, mask, pca_cfg(n_components=None, subsample_fraction=None, random_state=42))
    assert isinstance(model, PCA)
    assert features.dtype == np.float32


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("subsample", [None, 0.5])
def test_fit_pca_on_the_device_matches_cpu(subsample):
    import cupy as cp

    from karak.clustering.pca import PCAModel

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cfg = pca_cfg(n_components=None, subsample_fraction=subsample, random_state=42)
    model, features, idx = fit_pca(cube, mask, cfg)
    gmodel, gfeatures, gidx = fit_pca(cp.asarray(cube), cp.asarray(mask), cfg)
    assert isinstance(gmodel, PCAModel)
    assert isinstance(gfeatures, cp.ndarray) and gfeatures.dtype == cp.float32
    assert isinstance(gidx, cp.ndarray)
    np.testing.assert_allclose(gmodel.explained_variance_ratio_,
                               model.explained_variance_ratio_, atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(gfeatures), features, atol=1e-4)
    np.testing.assert_array_equal(cp.asnumpy(gidx), idx)
```

Check `pca_cfg` in `tests/conftest.py:73` for its field names (`n_components`, `subsample_fraction`, `random_state`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pca_gpu.py -q`
Expected: FAIL with `ImportError: cannot import name '_covariance_pca'` and the device-param assertion.

- [ ] **Step 3: Implement**

In `src/karak/clustering/pca.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PCAModel:
    """The fitted PCA of the device path (host arrays, float64)."""

    components_: np.ndarray
    mean_: np.ndarray
    explained_variance_ratio_: np.ndarray


def _covariance_pca(spectra, n_components: int, xp):
    """PCA through the covariance eigendecomposition, on ``xp``.

    The same algorithm sklearn's ``svd_solver="covariance_eigh"`` uses
    (its ``auto`` choice for tall, narrow inputs): centre, C x C
    covariance in float64, ``eigh``, descending order, and sklearn's sign
    convention (the largest-magnitude loading of each component is
    positive). Returns (components (k, C), mean (C,), explained variance
    ratio (k,)) as ``xp`` float64 arrays.
    """
    x = xp.asarray(spectra, dtype=xp.float64)
    n = x.shape[0]
    mean = x.mean(axis=0)
    centred = x - mean
    cov = centred.T @ centred / (n - 1)
    values, vectors = xp.linalg.eigh(cov)
    order = xp.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    components = vectors.T[:n_components]
    values = xp.maximum(values, 0.0)
    evr = values[:n_components] / values.sum()
    signs = xp.sign(components[xp.arange(n_components),
                               xp.argmax(xp.abs(components), axis=1)])
    signs = xp.where(signs == 0, 1.0, signs)
    components = components * signs[:, None]
    return components, mean, evr
```

In `fit_pca`, after the subsample block (`fit_spectra` chosen) and before `pca_model = PCA(...)`:

```python
    from karak.accel import is_device_array, to_numpy, xp as _xp

    if is_device_array(normalized_cube):
        xp = _xp(normalized_cube)
        components, mean, evr = _covariance_pca(fit_spectra, n_components, xp)
        pca_features = ((mineral_spectra.astype(xp.float64) - mean) @ components.T
                        ).astype(xp.float32)
        model = PCAModel(components_=to_numpy(components), mean_=to_numpy(mean),
                         explained_variance_ratio_=to_numpy(evr))
        logger.info("PCA fitted on the device: %d components, cumulative variance: %.1f%%",
                    n_components, float(model.explained_variance_ratio_.sum()) * 100)
        return model, pca_features, mineral_indices
```

Make the earlier lines device-neutral: `rows, cols = xp.where(mineral_mask)` and `mineral_indices = xp.stack([rows, cols], axis=1).astype(xp.int32)` (define `xp = _xp(normalized_cube)` at the top of the function; for numpy inputs it is `np` and nothing else changes). The subsample indices come from the host generator as today: `fit_idx = rng.choice(...)` is a numpy array; index a CuPy array with it via `xp.asarray(fit_idx)`.

In `src/karak/stages/pca.py`, add to `PARAMS` after `random_state`:

```python
        Param("device", "str", "cpu", "Device",
              "cpu or cuda (the executor moves the inputs; needs karak[cuda])",
              choices=("cpu", "cuda")),
```

`apply` is unchanged: `model.explained_variance_ratio_` is a host array on both paths, and `select_components(features, n_keep)` is a slice that works on CuPy. `image_shape=cube.pixels.shape[:2]` works on both.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pca_gpu.py -q` (and `uv run --extra cuda pytest tests/test_pca_gpu.py -q` on the GPU machine).
Expected: pass. If `test_covariance_pca_matches_sklearn_on_numpy` fails on component signs only, check sklearn's `svd_flip` in the installed version (`sklearn/utils/extmath.py`): with `u_based_decision=False` the sign comes from the max-abs entry of each row of `components_`; that is what the code above does. If sklearn's `auto` solver picked a different path for this input size, the components still agree up to sign; then compare `np.abs(components)` and the transformed features up to a per-column sign, and keep the row-based convention.

- [ ] **Step 5: Commit**

```bash
git add src/karak/clustering/pca.py src/karak/stages/pca.py tests/test_pca_gpu.py
git commit -m "feat: PCA on the device through a CuPy covariance eigendecomposition"
```

---

### Task 7: HDBSCAN consumes and produces device arrays

**Files:**
- Modify: `src/karak/clustering/hdbscan_cluster.py:56-71`, `src/karak/clustering/tiling.py` (the `device == "cuda"` per-tile call, around lines 584-615)
- Test: `tests/test_hdbscan_gpu.py`

**Interfaces:**
- Consumes: `accel.xp`.
- Produces: `run_hdbscan(pca_features, config, device="cuda")` accepts numpy or CuPy features and returns CuPy `labels` (int32) and `probabilities` (float32). With `device="cpu"` and CuPy input it raises the same `StageError` as denoise ("received a device array").

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hdbscan_gpu.py`:

```python
@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_device_features_in_device_labels_out():
    import cupy as cp

    labels, probs, _ = run_hdbscan(
        cp.asarray(_features()), hdbscan_cfg(min_cluster_size=100), device="cuda",
    )
    assert isinstance(labels, cp.ndarray) and labels.dtype == cp.int32
    assert isinstance(probs, cp.ndarray) and probs.dtype == cp.float32
    assert len(set(cp.asnumpy(labels).tolist()) - {-1}) == 2


def test_cpu_path_rejects_device_features():
    class FakeDeviceArray:
        __module__ = "cupy"
        shape = (10, 3)

    with pytest.raises(StageError, match="device array"):
        run_hdbscan(FakeDeviceArray(), hdbscan_cfg(min_cluster_size=5), device="cpu")
```

Update `test_cuml_hdbscan_shapes_and_sanity` in the same file: the return values are now CuPy arrays, so wrap them with `cp.asnumpy` before the `dtype`/`shape` assertions (or assert on `labels.dtype == cp.int32`).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra cuda pytest tests/test_hdbscan_gpu.py -q` (GPU machine) and `uv run pytest tests/test_hdbscan_gpu.py -q`.
Expected: the device test fails on `isinstance(labels, cp.ndarray)`; the CPU test fails because no `StageError` is raised.

- [ ] **Step 3: Implement**

In `src/karak/clustering/hdbscan_cluster.py`, replace the CUDA branch:

```python
    from karak.accel import is_device_array

    if device == "cpu" and is_device_array(pca_features):
        from karak.errors import StageError
        raise StageError(
            "device='cpu' received a device array; the executor places "
            "inputs on the stage's device, so run this stage with device='cuda'"
        )
    if device == "cuda":
        from karak.accel import get_array_module

        cp = get_array_module(device)  # raises StageError without CUDA
        from cuml.cluster import HDBSCAN as CumlHDBSCAN

        min_samples = (config.min_samples if config.min_samples is not None
                       else config.min_cluster_size)
        model = CumlHDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
        )
        model.fit(cp.asarray(pca_features, dtype=cp.float32))
        labels = cp.asarray(model.labels_).astype(cp.int32)
        probabilities = cp.asarray(model.probabilities_).astype(cp.float32)
        return labels, probabilities, model
```

Update the docstring's Returns: "on cuda, CuPy arrays".

In `src/karak/clustering/tiling.py`, the per-tile branch (`device == "cuda"`, the `run_hdbscan(tile_features, hdb_cfg, device=device)` call around line 613) receives `pca_features[tile.pixel_indices]`; with CuPy features this slice is a CuPy array and the labels come back as CuPy. The tile bookkeeping after that call uses `np.sum` and `.tolist()`: bring the tile results to the host there with `to_numpy(tile_labels)` and `to_numpy(tile_probs)` (import `to_numpy` from `karak.accel`), so the registry merge stays numpy code and `TiledArtifacts` stays host-only, as the spec says. The final assembled `labels` array the tiled path returns should be on the same device as the input features: build it with `xp = _xp(pca_features)`; the stage wraps it in `Labels`.

Check `HdbscanTiledStage.apply` (`stages/cluster.py:112-144`): it passes `cube.pixels` to `run_tiled_hdbscan` for the fingerprint means; that code indexes the cube with tile pixel indices and calls `.mean(axis=0)`; with a device cube it yields device means, so convert with `to_numpy` where the registry entry is built.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra cuda pytest tests/test_hdbscan_gpu.py tests/test_tiling*.py -q` (GPU machine) and `uv run pytest -q`.
Expected: pass, apart from the hash and drift tests that Task 8 fixes.

- [ ] **Step 5: Commit**

```bash
git add src/karak/clustering/hdbscan_cluster.py src/karak/clustering/tiling.py tests/test_hdbscan_gpu.py
git commit -m "feat: cuML HDBSCAN takes and returns device arrays"
```

---

### Task 8: Shipped flows, golden hashes, schema, and stage reference

**Files:**
- Modify: `src/karak/flow/flows/{global,tiled,tiled-rare,stepwise}.json`, `tests/test_recipe_stability.py`, `docs/flow.schema.json`, `docs/stage_reference.md`
- Test: `uv run pytest`

- [ ] **Step 1: Regenerate the shipped flows**

```bash
for f in src/karak/flow/flows/*.json; do uv run karak flow complete "$f" -o "$f"; done
git diff --stat src/karak/flow/flows/
```

Expected: each flow gains `"device": "cpu"` on `nrm` and `pca` (stepwise: `nrm`/`pca` are not in it yet, so stepwise is unchanged). Confirm with `git diff src/karak/flow/flows/global.json`.

- [ ] **Step 2: Capture the new golden hashes**

```bash
uv run python - <<'EOF'
from karak.flow.builtins import builtin_flow
from karak.flow.executor import plan_recipes
from tests.test_recipe_stability import TOKENS   # or copy TOKENS from that file
for name in ("global", "tiled", "tiled-rare", "stepwise"):
    print(name, plan_recipes(builtin_flow(name), TOKENS))
EOF
```

(if `tests` is not importable, paste `TOKENS` from `tests/test_recipe_stability.py:24-25`). Update `_SHARED["nrm"]`, `_SHARED["pca"]` and every downstream hash (`hdb`, `rare`, `knn`, `stats`, `fp`) per flow in `GOLDEN`. `src`, `msk` and `dn` must be unchanged; if they changed, stop and find out why before editing anything. Add a paragraph to the test's docstring: "2026-09-30: normalize and pca gained a `device` param, so `nrm`, `pca` and everything downstream carry new hashes."

- [ ] **Step 3: Regenerate the docs**

```bash
uv run python -m karak.stages.reference
uv run python -m karak.flow.schema
```

- [ ] **Step 4: Run the suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/flows tests/test_recipe_stability.py docs/flow.schema.json docs/stage_reference.md
git commit -m "chore: device param on normalize and pca in the shipped flows; golden hashes and docs regenerated"
```

---

### Task 9: Docs, changelog, and the real-data validation

**Files:**
- Modify: `docs/user_guide.md` (stage table rows for `normalize` and `pca`; the caching section after the RAM-budget paragraph; the "Computational requirements" section around line 358), `CHANGELOG.md`, `CLAUDE.md` (commands block: `--gpu-budget`), `docs/superpowers/specs/2026-09-18-acceleration-design.md` (one line under "Numeric core rules" pointing to the new spec)
- Test: the GPU chain test below, then the NWA 4587 runs

- [ ] **Step 1: Write the chain test (GPU machine)**

Append to `tests/test_cli_run.py`:

```python
@pytest.mark.skipif(not __import__("karak.accel", fromlist=["cuda_available"]).cuda_available(),
                    reason="no CUDA")
def test_stepwise_on_cuda_keeps_the_cube_on_the_device(tmp_path, scene):
    data_dir, colormap = scene
    out = tmp_path / "out" / "run"
    assert main(["run", "--builtin", "stepwise", "--input", str(data_dir),
                 "--out", str(out), "--plain", "--device", "cuda",
                 "--set", f"src.colormap={colormap}"]) == 0
    nodes = json.loads((out / "runs" / "latest" / "run.json").read_text())["nodes"]
    assert nodes["src"]["outputs"]["cube"]["device"] == "cpu"
    assert nodes["dn"]["outputs"]["cube"]["device"] == "cuda"
```

(When the `stepwise` flow has gained `nrm` and `pca` by the time this runs, extend the assertion to those nodes.)

- [ ] **Step 2: Write the docs**

User guide, caching section, after the RAM-budget paragraph:

```markdown
With `--device cuda`, a step's outputs stay on the GPU for the next GPU
step: the executor moves each input to the consuming step's device (its
`device` parameter, or the CPU when it has none), so a CPU step always
sees host arrays and a chain such as denoise → normalize → pca → hdbscan
moves the cube to the device once. Device outputs are held up to a
budget, 80 % of the free device memory at start (`--gpu-budget GB`);
beyond it an output moves to host RAM (logged as `store: dn.cube (1.98
GB) moved to host, gpu budget 19.6 GB`) and the RAM rules apply. The
cache always receives a host copy. The run record notes each output's
placement (`"device": "cuda"`), and the dashboard shows the device memory
next to the host figure.
```

Stage table: `normalize` and `pca` rows gain "On `--device cuda` runs on the GPU (exact tier)". Computational requirements: replace "No GPU is required; all computation is CPU-based" with "No GPU is required. With the `cuda` extra and `--device cuda`, denoise, normalize, PCA and HDBSCAN run on the GPU."

In `docs/superpowers/specs/2026-09-18-acceleration-design.md`, under "Numeric core rules", add: "Superseded for stages with a `device` param by `2026-09-30-async-cache-and-device-residency-design.md`: GPU paths take and return device arrays; payloads may hold them."

CHANGELOG, Unreleased / Added:

```markdown
- Device-resident payloads: with `--device cuda`, stage outputs stay on the
  GPU for the next GPU step. The executor places every input on the
  consuming step's device; CPU steps always see host arrays; the cache gets
  a host copy. `payload.device` and `payload.to()`; `--gpu-budget GB`
  (default 80 % of free device memory) with host fallback; the run record
  and the dashboard show placement and device memory.
- normalize and pca gained a `device` param and GPU paths: a CuPy z-score
  (exact tier, 1e-4) and a CuPy port of sklearn's covariance PCA (exact
  tier, 1e-4). cuML HDBSCAN now takes and returns device arrays. The
  recipe hashes of `nrm`, `pca` and downstream nodes changed; the shipped
  flows were regenerated.
```

Add `karak run ... --gpu-budget GB` to the `CLAUDE.md` commands block.

- [ ] **Step 3: Run the suite on the GPU machine**

Run: `uv run --extra cuda pytest -q`
Expected: all pass (no skips on this machine except the `no CUDA` ones being taken).

- [ ] **Step 4: Real-data validation on NWA 4587**

```bash
# CPU reference (writer + lzf from the previous plan)
uv run karak run --builtin global --input /home/brendon/Dropbox/Projects/izawa/NWA_4587_data \
    --out output/nwa4587/global_cpu --workers 0 --plain --no-qc 2>&1 | tail -40
# GPU chain
uv run karak run --builtin global --input /home/brendon/Dropbox/Projects/izawa/NWA_4587_data \
    --out output/nwa4587/global_gpu --device cuda --plain --no-qc 2>&1 | tail -40
```

Then compare the cached outputs of `nrm` and `pca` between the two runs with a short script (load both `run.json` files, open the listed cache files with `h5py`, `np.abs(a - b).max()` for the normalized cube, the PCA features and the explained variance ratios) and report: per-node seconds for both runs, `cache: ... written in` lines, the `gpu` figure from a dashboard run, any `moved to host` or `spilled` lines, and the three max differences against the tolerances (1e-4, 1e-4, 1e-6). HDBSCAN is loose tier: report the cluster count and noise fraction for both.

- [ ] **Step 5: Commit, push, open the PR**

```bash
git add docs/user_guide.md CHANGELOG.md CLAUDE.md docs/superpowers/specs/2026-09-18-acceleration-design.md tests/test_cli_run.py
git commit -m "docs: device-resident payloads, GPU normalize and pca"
git push -u origin HEAD
gh pr create --title "feat: device-resident payloads with GPU normalize and pca" --body "<what changed, why, tests, the NWA 4587 numbers, rulings, follow-ups>"
gh pr checks --watch
```

Stop there; the user reviews and merges.
