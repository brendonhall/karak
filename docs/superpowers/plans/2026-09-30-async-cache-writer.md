# Asynchronous Cache Writer, lzf Format and RAM Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Post-review changes (2026-10-02, PR #6):** `CacheWriter.close()` discards the queue and re-raises on a second Ctrl-C; tmp files are per-process and cleaned; writer errors become `FlowError`s at the next node; the writer queue is bounded by `max_pending_bytes` (set to the RAM budget) and drops each payload once written. See the spec's Section 1.

**Goal:** Take the cache write off the executor's critical path, make the cache files cheaper to write, and keep large payloads in RAM between nodes.

**Architecture:** A `CacheWriter` thread in `flow/cache.py` writes payloads FIFO through the existing tmp-then-rename `store_payload`. The executor submits a host copy of each output to the writer, puts the original in the `PayloadStore`, and drains the writer in a `finally`. `to_h5` takes a compression argument (default `lzf`). The per-payload 256 MB spill threshold becomes a total RAM budget.

**Tech Stack:** Python 3.12, numpy, h5py, `threading` / `queue` (standard library), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-30-async-cache-and-device-residency-design.md`, sections "Section 1" and "Section 4 / Writer and format". Plan B (`2026-09-30-device-residency.md`) builds on this plan.

## Global Constraints

- Every change goes through a pull request from a branch off `main` (CLAUDE.md). Run `uv run pytest` before each push. Update `CHANGELOG.md` (Unreleased) in the same branch.
- Payloads are immutable; `apply()` never mutates inputs.
- Recipe hashes exclude the compression setting; no golden hash in `tests/test_recipe_stability.py` changes in this plan.
- The writer thread never calls the reporter; the Rich dashboard is not thread-safe. Log lines queue up and the executor drains them on the main thread.
- `has_payload` must only ever see complete files: every write goes through a tmp file and `os.replace`.
- Rich stays in `cli/`; nothing under `flow/` imports Rich.
- Commit messages end with the attribution lines from the session (`Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and the `Claude-Session:` line).

## Review Focus

1. Disk full or a permission error during a background write: the run must fail with the writer's exception at the next drain, not finish "ok" with a missing cache file. Pinned in Task 2 (`test_writer_exception_surfaces_at_wait`) and Task 3 (`test_writer_failure_fails_the_run`).
2. Two `karak run` processes writing the same cache entry at once (same recipe): today they share one tmp name and can corrupt each other. Pinned in Task 2: the tmp name carries the pid (`test_tmp_name_is_unique_per_process`).
3. Ctrl-C while the writer still holds an output: the finished outputs must land on disk, and the exit code stays 130. Pinned in Task 3 (`test_interrupt_drains_the_writer_and_exits_130`).
4. `--no-cache`: no writer, no spill, nothing written. Pinned in Task 3 (`test_no_cache_creates_no_writer_and_writes_nothing`).
5. A machine without `/proc/meminfo` (macOS, some containers): the RAM budget falls back to 8 GB instead of crashing. Pinned in Task 5 (`test_default_budget_without_proc_meminfo`).

---

### Task 1: `to_h5` compression argument, chunking, and `store_payload(compression=)`

**Files:**
- Modify: `src/karak/stages/payloads.py` (every `to_h5`; lines 93-104, 135-138, 161-171, 204-215, 249-261, 292-318)
- Modify: `src/karak/flow/cache.py:44-60` (`store_payload`)
- Test: `tests/test_payloads.py`, `tests/test_cache.py`

**Interfaces:**
- Produces: `Payload.to_h5(group, compression="lzf")` on every payload class; `store_payload(recipe, port, payload, cache_dir, upstream=None, compression="lzf") -> Path`; module constants `payloads.CACHE_COMPRESSION = "lzf"` and `payloads.COMPRESSIONS = ("lzf", "gzip", "none")`; helper `payloads._dataset(group, name, data, compression)`.

- [ ] **Step 1: Write the failing round-trip tests**

Append to `tests/test_payloads.py`:

```python
@pytest.mark.parametrize("compression", ["lzf", "gzip", "none"])
def test_every_payload_roundtrips_under_each_compression(tmp_path, compression):
    """Each payload class writes with the requested filter and reads back."""
    import h5py

    from karak.stages.payloads import (
        BseImage, ClusterStats, Labels, LabelState, MaskSet, PCAFeatures,
    )

    cube = _cube()
    payloads = [
        cube,
        BseImage(pixels=np.zeros((4, 5), np.float32)),
        MaskSet(mineral_mask=np.ones((4, 5), bool), valid_mask=None),
        PCAFeatures(features=np.zeros((7, 3), np.float32),
                    mineral_indices=np.zeros((7, 2), np.int32),
                    image_shape=(4, 5),
                    explained_variance_ratio=np.array([0.5, 0.3, 0.2]),
                    n_kept=2),
        Labels(labels=np.zeros(7, np.int32), probabilities=None,
               mineral_indices=np.zeros((7, 2), np.int32),
               image_shape=(4, 5), state=LabelState.RAW),
        ClusterStats(stats={"n": 1}),
    ]
    for payload in payloads:
        path = tmp_path / f"{type(payload).__name__}.h5"
        with h5py.File(path, "w") as fh:
            payload.to_h5(fh.create_group("p"), compression=compression)
        with h5py.File(path, "r") as fh:
            back = payload_from_h5(fh["p"])
            pixels = fh["p"].get("pixels")
            if pixels is not None:
                expected = None if compression == "none" else compression
                assert pixels.compression == expected
                assert pixels.chunks is not None
        assert type(back) is type(payload)


def test_to_h5_defaults_to_lzf(tmp_path):
    import h5py

    with h5py.File(tmp_path / "d.h5", "w") as fh:
        _cube().to_h5(fh.create_group("p"))
    with h5py.File(tmp_path / "d.h5", "r") as fh:
        assert fh["p"]["pixels"].compression == "lzf"


def test_gzip_file_written_before_the_change_still_loads(tmp_path):
    """HDF5 records the filter per dataset; the reader does not care."""
    import h5py

    cube = _cube()
    with h5py.File(tmp_path / "old.h5", "w") as fh:
        g = fh.create_group("p")
        g.attrs["payload_type"] = "element_cube"
        g.attrs["element_names"] = list(cube.element_names)
        g.attrs["space"] = "raw"
        g.attrs["downsample_factor"] = 2
        g.attrs["header_trim_px"] = 100
        g.attrs["left_trim_px"] = 0
        g.create_dataset("pixels", data=cube.pixels, compression="gzip")
    with h5py.File(tmp_path / "old.h5", "r") as fh:
        back = payload_from_h5(fh["p"])
    np.testing.assert_array_equal(back.pixels, cube.pixels)
```

Check the top of `tests/test_payloads.py` imports `pytest`, `np`, `payload_from_h5`, `_cube`; add any that are missing. Confirm the `Labels` and `PCAFeatures` constructor field names against `src/karak/stages/payloads.py:184-272` before running (they are `features, mineral_indices, image_shape, explained_variance_ratio, n_kept` and `labels, probabilities, mineral_indices, image_shape, state`).

Append to `tests/test_cache.py`:

```python
def test_store_payload_passes_the_compression_through(tmp_path):
    import h5py

    from karak.stages.payloads import BseImage

    payload = BseImage(pixels=np.zeros((4, 4), np.float32))
    path = store_payload("r1", "bse", payload, tmp_path, compression="none")
    with h5py.File(path) as fh:
        assert fh["payload"]["pixels"].compression is None
    path = store_payload("r2", "bse", payload, tmp_path)
    with h5py.File(path) as fh:
        assert fh["payload"]["pixels"].compression == "lzf"
```

Add `import numpy as np` at the top of `tests/test_cache.py` if it is not there.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_payloads.py tests/test_cache.py -q`
Expected: FAIL with `TypeError: to_h5() got an unexpected keyword argument 'compression'` and the same for `store_payload`.

- [ ] **Step 3: Implement the compression argument and chunking**

In `src/karak/stages/payloads.py`, after `payload_from_h5`, add:

```python
CACHE_COMPRESSION = "lzf"
COMPRESSIONS = ("lzf", "gzip", "none")


def _dataset(group, name, data, compression=CACHE_COMPRESSION):
    """Create a chunked, optionally compressed dataset.

    Chunks are (512, 512) over the first two axes (the image axes), whole
    along the rest, so a cube and its per-channel reads stay cheap.
    """
    data = np.asarray(data)
    if compression not in COMPRESSIONS:
        raise ValueError(f"compression must be one of {COMPRESSIONS}, got {compression!r}")
    if data.ndim < 2 or data.size == 0:
        return group.create_dataset(name, data=data)
    chunks = tuple(min(512, n) for n in data.shape[:2]) + tuple(data.shape[2:])
    kwargs = {} if compression == "none" else {"compression": compression}
    return group.create_dataset(name, data=data, chunks=chunks, **kwargs)
```

Then change every `to_h5(self, group)` to `to_h5(self, group, compression=CACHE_COMPRESSION)` and replace each `group.create_dataset(<name>, data=<arr>, compression="gzip")` with `_dataset(group, <name>, <arr>, compression)`. The datasets that were written without compression (`means`, `stds`, `explained_variance_ratio`, `mean_fingerprint`) stay `group.create_dataset(...)`. For `TiledArtifacts.to_h5`, `local_labels` becomes `_dataset(sub, "local_labels", tr.local_labels, compression)`. `ClusterStats.to_h5` and `Fingerprints.to_h5` accept and ignore `compression`.

In `src/karak/flow/cache.py`, change `store_payload`:

```python
def store_payload(
    recipe: str, port: str, payload, cache_dir: str | Path,
    upstream: dict | None = None, compression: str = "lzf",
) -> Path:
    """Write a payload to the cache. ``upstream`` maps the producing
    node's input ports to the recipe hashes they consumed; it is stored
    as an attribute so a cache scan can tell which outputs belong together.
    ``compression`` is the HDF5 filter: "lzf" (default), "gzip" or "none"."""
    path = payload_path(recipe, port, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with h5py.File(tmp, "w") as fh:
        group = fh.create_group("payload")
        payload.to_h5(group, compression=compression)
        if upstream:
            group.attrs["upstream"] = json.dumps(dict(sorted(upstream.items())))
    os.replace(tmp, path)
    return path


def _tmp_path(path: Path) -> Path:
    """A tmp name unique to this process, so two runs that race on the same
    entry never write the same file."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_payloads.py tests/test_cache.py -q`
Expected: all pass.

Run: `uv run pytest -q`
Expected: all pass. `tests/test_cli_view.py` globs `*__*.h5`; a `.tmp` name never matches it.

- [ ] **Step 5: Commit**

```bash
git add src/karak/stages/payloads.py src/karak/flow/cache.py tests/test_payloads.py tests/test_cache.py
git commit -m "feat: cache files take a compression setting, default lzf, chunked"
```

---

### Task 2: `CacheWriter`

**Files:**
- Modify: `src/karak/flow/cache.py` (append)
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `store_payload(..., compression=)`, `store_summary` from Task 1.
- Produces:

```python
class CacheWriter:
    def __init__(self, cache_dir, compression="lzf"): ...
    def submit(self, recipe, port, payload, summary, upstream, label=None) -> None
    def wait_for(self, recipe, port) -> None      # blocks until that entry is on disk; re-raises a writer error
    def wait(self) -> None                         # drains everything; re-raises the first writer error
    def close(self) -> None                        # wait(), then stop the thread; idempotent
    def drain_log(self) -> list[str]               # log lines produced since the last call
    seconds: float                                 # total time spent writing
    per_label: dict[str, float]                    # "dn.cube" -> seconds written
```

Lines in `drain_log` look like `cache: dn.cube written in 10.8 s`. The writer records `(recipe, port)` in `self.written` once an entry is on disk.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache.py`:

```python
import threading
import time

from karak.flow.cache import CacheWriter, has_payload


def _bse(n=4):
    from karak.stages.payloads import BseImage

    return BseImage(pixels=np.zeros((n, n), np.float32))


def test_writer_writes_entries_complete_or_not_at_all(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    gate = threading.Event()
    real = cache.store_payload

    def slow_store(*args, **kwargs):
        gate.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "BseImage 4×4", {"cube": "abc"})
    time.sleep(0.05)
    assert not has_payload("r1", "bse", tmp_path)   # still in the queue
    assert not list(tmp_path.glob("*__bse.h5"))
    gate.set()
    writer.wait_for("r1", "bse")
    assert has_payload("r1", "bse", tmp_path)
    assert load_summary("r1", "bse", tmp_path) == "BseImage 4×4"
    assert ("r1", "bse") in writer.written
    writer.close()


def test_writer_wait_drains_in_fifo_order(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    order = []
    real = cache.store_payload

    def recording_store(recipe, port, *args, **kwargs):
        order.append((recipe, port))
        return real(recipe, port, *args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", recording_store)
    writer = CacheWriter(tmp_path)
    for i in range(5):
        writer.submit(f"r{i}", "bse", _bse(), "", {})
    writer.wait()
    assert order == [(f"r{i}", "bse") for i in range(5)]
    assert writer.seconds > 0
    writer.close()


def test_writer_exception_surfaces_at_wait(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "", {})
    with pytest.raises(OSError, match="disk full"):
        writer.wait()
    with pytest.raises(OSError, match="disk full"):   # sticky until close
        writer.wait_for("r1", "bse")
    writer.close()


def test_writer_log_lines_name_the_entry_and_the_time(tmp_path):
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "cube", _bse(), "", {}, label="dn.cube")
    writer.wait()
    lines = writer.drain_log()
    assert len(lines) == 1
    assert lines[0].startswith("cache: dn.cube written in ")
    assert lines[0].endswith(" s")
    assert writer.per_label["dn.cube"] > 0
    assert writer.drain_log() == []
    writer.close()


def test_writer_close_is_idempotent_and_wait_after_close_is_a_noop(tmp_path):
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "", {})
    writer.close()
    writer.close()
    writer.wait()
    assert has_payload("r1", "bse", tmp_path)


def test_tmp_name_is_unique_per_process(tmp_path):
    import os

    from karak.flow.cache import _tmp_path, payload_path

    tmp = _tmp_path(payload_path("r1", "bse", tmp_path))
    assert tmp.name == f"r1__bse.h5.{os.getpid()}.tmp"
```

Add `import pytest` to `tests/test_cache.py` if missing; `load_summary` is already imported there (check the top of the file).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cache.py -q`
Expected: FAIL with `ImportError: cannot import name 'CacheWriter'`.

- [ ] **Step 3: Implement `CacheWriter`**

Append to `src/karak/flow/cache.py`:

```python
import queue
import threading
import time


class CacheWriter:
    """Writes payloads to the cache on one background thread, FIFO.

    ``submit`` enqueues a host payload; the thread calls ``store_payload``
    and ``store_summary``. ``wait_for`` and ``wait`` block on the disk state
    and re-raise the first error the thread hit. The thread never talks to
    a reporter: it appends log lines that ``drain_log`` hands back to the
    caller's thread.
    """

    def __init__(self, cache_dir: str | Path, compression: str = "lzf"):
        self.cache_dir = Path(cache_dir)
        self.compression = compression
        self.written: set[tuple[str, str]] = set()
        self.seconds = 0.0
        self.per_label: dict[str, float] = {}   # "dn.cube" -> seconds
        self._queue: queue.Queue = queue.Queue()
        self._done = threading.Condition()
        self._error: BaseException | None = None
        self._log: list[str] = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="karak-cache-writer", daemon=True)
        self._thread.start()

    def submit(self, recipe: str, port: str, payload, summary: str,
               upstream: dict | None, label: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("CacheWriter is closed")
        self._queue.put((recipe, port, payload, summary, upstream,
                         label or f"{recipe[:8]}.{port}"))

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            recipe, port, payload, summary, upstream, label = item
            started = time.monotonic()
            try:
                if self._error is None:
                    store_payload(recipe, port, payload, self.cache_dir,
                                  upstream=upstream,
                                  compression=self.compression)
                    store_summary(recipe, port, summary, self.cache_dir)
            except BaseException as exc:   # surfaces at the next wait
                self._error = exc
            else:
                if self._error is None:
                    elapsed = time.monotonic() - started
                    self.seconds += elapsed
                    with self._done:
                        self.written.add((recipe, port))
                        self.per_label[label] = self.per_label.get(label, 0.0) + elapsed
                        self._log.append(
                            f"cache: {label} written in {elapsed:.1f} s")
                        self._done.notify_all()
            finally:
                with self._done:
                    self._done.notify_all()
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def wait_for(self, recipe: str, port: str) -> None:
        """Block until (recipe, port) is on disk; raise a writer error."""
        with self._done:
            while (recipe, port) not in self.written and self._error is None:
                if self._queue.unfinished_tasks == 0:
                    break          # never submitted, or already failed
                self._done.wait(0.05)
        self._raise_if_failed()

    def wait(self) -> None:
        if self._thread.is_alive():
            self._queue.join()
        self._raise_if_failed()

    def drain_log(self) -> list[str]:
        with self._done:
            lines, self._log = self._log, []
        return lines

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.join()
        finally:
            self._queue.put(None)
            self._thread.join()
        self._raise_if_failed()
```

`queue.Queue.unfinished_tasks` has been a public attribute of CPython's queue since 2.5; read it under the condition lock only, as above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/cache.py tests/test_cache.py
git commit -m "feat: CacheWriter writes cache entries on a background thread"
```

---

### Task 3: The executor uses the writer

**Files:**
- Modify: `src/karak/flow/executor.py` (`run` 175-212, `_execute` 215-378: the store construction, the miss path, the record call, the end of the loop)
- Test: `tests/test_executor.py`, `tests/test_cli_run.py`

**Interfaces:**
- Consumes: `CacheWriter` from Task 2.
- Produces: `run(..., cache_compression: str = "lzf")`; each node's summary entry gains `"write_seconds": float` (0.0 for sinks, cache hits and `--no-cache`); `reporter.log("info", line)` receives the writer's lines on the main thread; the run summary dict is unchanged otherwise.

- [ ] **Step 1: Write the failing executor tests**

Append to `tests/test_executor.py`:

```python
import threading
import time


def _chain(tmp_path, n=3):
    """src -> add1 -> add2 (-> ...) with a sink, all fake stages."""
    nodes = [Node(id="src", type="fake_source", params={"value": 1, "path": "{input}"})]
    edges = []
    prev = "src"
    for i in range(1, n):
        nodes.append(Node(id=f"add{i}", type="fake_add", params={"add": i}))
        edges.append(Edge(id=f"e{i}", src=Endpoint(prev, "num"), dst=Endpoint(f"add{i}", "num")))
        prev = f"add{i}"
    nodes.append(Node(id="out", type="fake_sink", params={"out": "{out}"}))
    edges.append(Edge(id="es", src=Endpoint(prev, "num"), dst=Endpoint("out", "num")))
    return complete(Graph(name="chain", nodes=tuple(nodes), edges=tuple(edges)))


class _Timeline:
    """A reporter that records node starts and the writer's log lines."""

    def __init__(self):
        self.events = []

    def node_started(self, node_id, label):
        self.events.append(("start", node_id, time.monotonic()))

    def log(self, level, msg):
        self.events.append(("log", msg, time.monotonic()))

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_next_node_starts_before_the_previous_output_is_written(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    real = cache.store_payload
    finished = {}

    def slow_store(recipe, port, *args, **kwargs):
        time.sleep(0.3)
        path = real(recipe, port, *args, **kwargs)
        finished[(recipe, port)] = time.monotonic()
        return path

    monkeypatch.setattr(cache, "store_payload", slow_store)
    reporter = _Timeline()
    summary = run(_chain(tmp_path), input_path="x", out_base=str(tmp_path / "o"),
                  work_dir=str(tmp_path / "w"), reporter=reporter)
    starts = {e[1]: e[2] for e in reporter.events if e[0] == "start"}
    src_written = min(finished.values())
    assert starts["add1"] < src_written          # add1 ran while src's file was pending
    assert all(v.get("write_seconds", 0) >= 0 for v in summary.values())
    assert summary["src"]["write_seconds"] >= 0.3
    assert summary["out"]["write_seconds"] == 0.0
    logs = [e[1] for e in reporter.events if e[0] == "log"]
    assert any(line.startswith("cache: src.num written in") for line in logs)
    # and every file exists once run() returns
    assert len(list((tmp_path / "w" / "cache").glob("*__num.h5"))) == 3


def test_spilled_reload_waits_for_the_pending_write(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    real = cache.store_payload

    def slow_store(*args, **kwargs):
        time.sleep(0.3)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)
    # ram_budget=0 spills everything, so add1 must reload src's output
    summary = run(_chain(tmp_path), input_path="x", out_base=str(tmp_path / "o"),
                  work_dir=str(tmp_path / "w"), ram_budget=0)
    assert summary["add2"]["cached"] is False
    assert RECORD[-1][1] == 1 + 1 + 2   # the sink saw the right value


def test_writer_failure_fails_the_run(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    with pytest.raises(OSError, match="disk full"):
        run(_chain(tmp_path), input_path="x", out_base=str(tmp_path / "o"),
            work_dir=str(tmp_path / "w"))


def test_failing_node_still_gets_earlier_outputs_written(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    real = cache.store_payload

    def slow_store(*args, **kwargs):
        time.sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)

    class Boom(Stage):
        id = "fake_boom"
        label = "Boom"
        INPUTS = [Port("num")]
        OUTPUTS = [Port("num")]
        PARAMS = []

        def apply(self, inputs, params):
            raise RuntimeError("boom")

    registry.register(Boom)
    graph = complete(Graph(name="boom", nodes=(
        Node(id="src", type="fake_source", params={"value": 1, "path": "x"}),
        Node(id="b", type="fake_boom", params={}),
    ), edges=(Edge(id="e", src=Endpoint("src", "num"), dst=Endpoint("b", "num")),)))
    with pytest.raises(FlowError, match="boom"):
        run(graph, input_path="x", out_base=str(tmp_path / "o"),
            work_dir=str(tmp_path / "w"))
    assert list((tmp_path / "w" / "cache").glob("*__num.h5"))   # src landed


def test_interrupt_drains_the_writer(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    real = cache.store_payload

    def slow_store(*args, **kwargs):
        time.sleep(0.2)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)

    class Interrupt(Stage):
        id = "fake_interrupt"
        label = "Interrupt"
        INPUTS = [Port("num")]
        OUTPUTS = [Port("num")]
        PARAMS = []

        def apply(self, inputs, params):
            raise KeyboardInterrupt

    registry.register(Interrupt)
    graph = complete(Graph(name="kbi", nodes=(
        Node(id="src", type="fake_source", params={"value": 1, "path": "x"}),
        Node(id="k", type="fake_interrupt", params={}),
    ), edges=(Edge(id="e", src=Endpoint("src", "num"), dst=Endpoint("k", "num")),)))
    with pytest.raises(KeyboardInterrupt):
        run(graph, input_path="x", out_base=str(tmp_path / "o"),
            work_dir=str(tmp_path / "w"))
    assert list((tmp_path / "w" / "cache").glob("*__num.h5"))


def test_no_cache_creates_no_writer_and_writes_nothing(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    def forbidden(*args, **kwargs):
        raise AssertionError("store_payload must not be called with cache=False")

    monkeypatch.setattr(cache, "store_payload", forbidden)
    summary = run(_chain(tmp_path), input_path="x", out_base=str(tmp_path / "o"),
                  work_dir=str(tmp_path / "w"), cache=False)
    assert not (tmp_path / "w" / "cache").exists()
    assert all(v["write_seconds"] == 0.0 for v in summary.values())
```

Check how `registry.register` is spelled in `tests/test_executor.py`'s `_fake_stages` fixture (it registers `FakeSource`, `FakeAdd`, `FakeSink`) and use the same call for `Boom` and `Interrupt`. If the registry rejects duplicate ids on re-registration across tests, register them inside the fixture instead.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_executor.py -q -k "written or spilled_reload or writer_failure or earlier_outputs or interrupt_drains or no_cache_creates"`
Expected: FAIL. The first fails on `assert starts["add1"] < src_written` (today the write is synchronous) or on the missing `write_seconds` key; `test_spilled_reload_waits_for_the_pending_write` fails with `TypeError: run() got an unexpected keyword argument 'ram_budget'` (that keyword arrives in Task 5; until then, temporarily pass `spill_threshold=0` in this one test and switch it to `ram_budget=0` in Task 5).

- [ ] **Step 3: Implement the writer in the executor**

In `src/karak/flow/executor.py`:

1. Import `CacheWriter` from `karak.flow.cache` (add to the existing import block).

2. Add `cache_compression: str = "lzf"` to `run()` and pass it to `_execute`.

3. In `_execute`, after `cache_dir = Path(work_dir) / "cache"`:

```python
    writer = CacheWriter(cache_dir, compression=cache_compression) if cache else None

    def _drain_writer_log() -> None:
        if writer is not None:
            for line in writer.drain_log():
                _emit(reporter, "log", "info", line)
```

4. Change `_reload` so a spilled payload waits for its file:

```python
    def _reload(node: str, port: str):
        recipe = hashes[(node, port)]
        if writer is not None:
            writer.wait_for(recipe, port)
        return load_payload(recipe, port, cache_dir)
```

5. Wrap the node loop (from `for node_id in _topo_order(graph):` to the `run_finished` emit) in `try: ... finally:`:

```python
    try:
        for node_id in _topo_order(graph):
            ...                      # the existing loop body, with the change in 6
            _drain_writer_log()
    finally:
        if writer is not None:
            try:
                writer.close()       # drains; re-raises a writer error
            finally:
                _drain_writer_log()
                for node_id, entry in summary.items():
                    entry["write_seconds"] = sum(
                        seconds for label, seconds in writer.per_label.items()
                        if label.startswith(node_id + ".")
                    )
```

`writer.per_label` is filled by the writer thread (Task 2); reading it after `close()` needs no lock.

6. Replace the miss-path write (the block starting `if cache and not is_sink:` with the `store_payload` and `store_summary` calls) with:

```python
            if cache and not is_sink:
                upstream = _upstream_recipes(graph, node_id, hashes)
                for port_name, payload in outputs.items():
                    writer.submit(node_hash, port_name, payload,
                                  summaries[port_name], upstream,
                                  label=f"{node_id}.{port_name}")
```

and initialise every node's summary entry with `"write_seconds": 0.0` where `summary[node_id] = {"cached": cached_hit, "seconds": elapsed}` is set (add the key there).

7. `run()` needs no change for exceptions: the `finally` in `_execute` drains before the exception propagates to `run()`'s handlers, so the record still finishes as "failed" or "interrupted" after the outputs are on disk.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_executor.py tests/test_cache.py -q`
Expected: all pass.

Run: `uv run pytest -q`
Expected: all pass. `tests/test_run_record.py::test_ok_run_records_complete_flow_params_and_outputs` asserts the output file exists after `run()` returns; the `finally` drain keeps that true.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/executor.py tests/test_executor.py tests/test_cache.py src/karak/flow/cache.py
git commit -m "feat: executor writes cache entries through the background writer"
```

---

### Task 4: `--cache-compression` on the CLI

**Files:**
- Modify: `src/karak/flow/__main__.py:97-125` (run parser) and `:208-218` (the `run_flow` call and `RunRecord` settings)
- Test: `tests/test_cli_run.py`

**Interfaces:**
- Consumes: `run(cache_compression=)` from Task 3.
- Produces: `karak run --cache-compression {lzf,gzip,none}` (default `lzf`); the run record's `settings` gains `"cache_compression"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_run.py`:

```python
def test_cache_compression_flag_reaches_the_executor_and_the_record(tmp_path, monkeypatch):
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
                 "--cache-compression", "none"]) == 0
    assert seen["cache_compression"] == "none"
    data = json.loads((out / "runs" / "latest" / "run.json").read_text())
    assert data["settings"]["cache_compression"] == "none"


def test_cache_compression_defaults_to_lzf(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update(kwargs)
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(executor, "run", fake_run)
    assert main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"), "--plain"]) == 0
    assert seen["cache_compression"] == "lzf"
```

Look at `test_keyboard_interrupt_exits_130` in the same file for the `main` import and the monkeypatch target; `karak.flow.__main__` imports `run as run_flow` inside `main`, so patching `executor.run` works.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_run.py -q -k cache_compression`
Expected: FAIL with `error: unrecognized arguments: --cache-compression none` (argparse exits 2) for the first, and `KeyError: 'cache_compression'` for the second.

- [ ] **Step 3: Implement the flag**

In `src/karak/flow/__main__.py`, after the `--no-cache` argument:

```python
    run_parser.add_argument(
        "--cache-compression", choices=["lzf", "gzip", "none"], default="lzf",
        help="HDF5 filter for cache files (lzf: fast; gzip: small; none: "
             "fastest, largest). Reads accept any.",
    )
```

Add `"cache_compression": args.cache_compression` to the `settings=` dict of `RunRecord(...)`, and `cache_compression=args.cache_compression` to the `run_flow(...)` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli_run.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/__main__.py tests/test_cli_run.py
git commit -m "feat: karak run --cache-compression {lzf,gzip,none}"
```

---

### Task 5: RAM budget replaces the per-payload spill threshold

**Files:**
- Create: `src/karak/flow/budget.py`
- Modify: `src/karak/flow/executor.py` (`PayloadStore` 48-94, `run` signature, store construction in `_execute`), `src/karak/flow/__main__.py` (flag, settings, run call)
- Test: `tests/test_budget.py` (new), `tests/test_executor.py`, `tests/test_cli_run.py`

**Interfaces:**
- Produces:
  - `budget.available_host_memory() -> int | None` (bytes from `/proc/meminfo` `MemAvailable`, None when unreadable)
  - `budget.default_ram_budget() -> int` (half of available, or `8 * 2**30` when unknown)
  - `PayloadStore(consumers, reload=None, ram_budget: int | None = None, on_spill=None)`: `None` means never spill; `on_spill(node, port, nbytes, budget)` is called once per spilled payload
  - `PayloadStore.held_bytes: int`
  - `run(..., ram_budget: int | None = None)`: `None` means `default_ram_budget()` when caching, never spill otherwise
  - `karak run --ram-budget GB` (float, default: half of available memory)
  - `spill_threshold` is removed from `run()` and `PayloadStore`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_budget.py`:

```python
"""RAM budget helpers."""

from __future__ import annotations

from karak.flow import budget


def test_available_host_memory_reads_meminfo(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\nMemAvailable:   20000000 kB\n")
    monkeypatch.setattr(budget, "MEMINFO", str(meminfo))
    assert budget.available_host_memory() == 20000000 * 1024


def test_default_budget_is_half_of_available(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   20000000 kB\n")
    monkeypatch.setattr(budget, "MEMINFO", str(meminfo))
    assert budget.default_ram_budget() == 20000000 * 1024 // 2


def test_default_budget_without_proc_meminfo(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "MEMINFO", str(tmp_path / "missing"))
    assert budget.available_host_memory() is None
    assert budget.default_ram_budget() == 8 * 2**30
```

In `tests/test_executor.py`, replace `test_payload_store_spills_large_payloads` with:

```python
def test_payload_store_spills_only_when_the_total_exceeds_the_budget():
    import numpy as np

    from karak.stages.payloads import BseImage

    one_kb = BseImage(pixels=np.zeros((16, 16), dtype=np.float32))   # 1024 bytes
    reloads, spills = [], []

    def reload(node, port):
        reloads.append((node, port))
        return one_kb

    store = PayloadStore({("a", "bse"): 1, ("b", "bse"): 1, ("c", "bse"): 2},
                         reload=reload, ram_budget=2048,
                         on_spill=lambda *args: spills.append(args))
    store.put("a", "bse", one_kb)
    store.put("b", "bse", one_kb)
    assert store.held_bytes == 2048
    store.put("c", "bse", one_kb)            # would make 3072 > 2048: spilled
    assert ("c", "bse") not in store._in_ram
    assert spills == [("c", "bse", 1024, 2048)]
    assert store.get("a", "bse") is one_kb   # released: 1024 held
    assert store.held_bytes == 1024
    assert store.get("c", "bse") is one_kb and store.get("c", "bse") is one_kb
    assert len(reloads) == 2                 # once per consumer of the spilled one


def test_payload_store_without_a_budget_never_spills():
    import numpy as np

    from karak.stages.payloads import BseImage

    big = BseImage(pixels=np.zeros((256, 256), dtype=np.float32))
    store = PayloadStore({("a", "bse"): 1}, reload=lambda n, p: None, ram_budget=None)
    store.put("a", "bse", big)
    assert ("a", "bse") in store._in_ram


def test_run_passes_the_ram_budget_and_logs_spills(tmp_path):
    class Logger:
        def __init__(self):
            self.lines = []

        def log(self, level, msg):
            self.lines.append(msg)

        def __getattr__(self, name):
            return lambda *a, **k: None

    reporter = Logger()
    run(_chain(tmp_path), input_path="x", out_base=str(tmp_path / "o"),
        work_dir=str(tmp_path / "w"), reporter=reporter, ram_budget=0)
    assert any(line.startswith("store: src.num (") and "spilled to cache" in line
               for line in reporter.lines)
```

Also switch `test_spilled_reload_waits_for_the_pending_write` (Task 3) to `ram_budget=0` if it still says `spill_threshold=0`.

Append to `tests/test_cli_run.py`:

```python
def test_ram_budget_flag_is_gigabytes(tmp_path, monkeypatch):
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
                 "--ram-budget", "1.5"]) == 0
    assert seen["ram_budget"] == int(1.5 * 2**30)
    data = json.loads((out / "runs" / "latest" / "run.json").read_text())
    assert data["settings"]["ram_budget_gb"] == 1.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_budget.py tests/test_executor.py tests/test_cli_run.py -q -k "budget or spills"`
Expected: FAIL with `ModuleNotFoundError: No module named 'karak.flow.budget'`, `TypeError: PayloadStore.__init__() got an unexpected keyword argument 'ram_budget'`, and an argparse error for `--ram-budget`.

- [ ] **Step 3: Implement the budget module, the store, and the flag**

Create `src/karak/flow/budget.py`:

```python
"""Host memory budget for payloads held between nodes (standard library)."""

from __future__ import annotations

MEMINFO = "/proc/meminfo"
FALLBACK_BUDGET = 8 * 2**30


def available_host_memory() -> int | None:
    """``MemAvailable`` in bytes, or None where /proc/meminfo is absent."""
    try:
        with open(MEMINFO) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def default_ram_budget() -> int:
    """Half of the available memory, or 8 GB when it cannot be read."""
    available = available_host_memory()
    return FALLBACK_BUDGET if available is None else available // 2
```

In `src/karak/flow/executor.py`, replace `PayloadStore`:

```python
class PayloadStore:
    """Refcounted in-RAM payload store with a total memory budget.

    ``consumers`` maps (node, port) -> number of downstream consumers.
    ``get`` decrements the count and evicts the payload once it reaches 0.
    A payload is spilled (not held; ``reload`` fetches it from the cache
    per consumer) when holding it would push the total above
    ``ram_budget`` bytes. ``ram_budget=None`` never spills.
    """

    def __init__(self, consumers: dict, reload=None,
                 ram_budget: int | None = None, on_spill=None):
        self._remaining = dict(consumers)
        self._in_ram: dict = {}
        self._sizes: dict = {}
        self._spilled: set = set()
        self._reload = reload
        self._budget = ram_budget
        self._on_spill = on_spill
        self.held_bytes = 0

    def put(self, node: str, port: str, payload) -> None:
        key = (node, port)
        if self._remaining.get(key, 0) <= 0:
            return  # unconsumed output — drop immediately
        nbytes = _payload_nbytes(payload)
        if (
            self._budget is not None
            and self._reload is not None
            and self.held_bytes + nbytes > self._budget
        ):
            self._spilled.add(key)
            if self._on_spill is not None:
                self._on_spill(node, port, nbytes, self._budget)
            return
        self._in_ram[key] = payload
        self._sizes[key] = nbytes
        self.held_bytes += nbytes

    def get(self, node: str, port: str):
        key = (node, port)
        if key in self._spilled:
            payload = self._reload(node, port)
        else:
            payload = self._in_ram[key]
        self.release(node, port)
        return payload

    def release(self, node: str, port: str) -> None:
        """Decrement the consumer count without fetching (cache-hit path)."""
        key = (node, port)
        remaining = self._remaining.get(key, 0) - 1
        self._remaining[key] = remaining
        if remaining <= 0:
            if self._in_ram.pop(key, None) is not None:
                self.held_bytes -= self._sizes.pop(key, 0)
            self._spilled.discard(key)
```

Update the module docstring's memory-model paragraph to describe the budget. In `run()`, replace `spill_threshold: int = 256 * 1024 * 1024` with `ram_budget: int | None = None` and pass it on. In `_execute`, replace the store construction:

```python
    from karak.flow.budget import default_ram_budget
    from karak.stages.payloads import format_bytes

    def _on_spill(node, port, nbytes, budget):
        _emit(reporter, "log", "info",
              f"store: {node}.{port} ({format_bytes(nbytes)}) spilled to cache, "
              f"budget {format_bytes(budget)}")

    store = PayloadStore(
        consumers,
        reload=_reload,
        ram_budget=(default_ram_budget() if ram_budget is None else ram_budget)
                   if cache else None,
        on_spill=_on_spill,
    )
```

Check `format_bytes` exists in `payloads.py:56-64` (it does) and prints `1.98 GB` style. Grep `spill_threshold` across `src/` and `tests/` and remove every remaining use (`cli/bench.py` and `flow/__main__.py` do not pass it; `tests/test_executor.py` had the one test replaced above).

In `src/karak/flow/__main__.py`, after `--cache-compression`:

```python
    run_parser.add_argument(
        "--ram-budget", type=float, default=None, metavar="GB",
        help="Host memory to hold stage outputs between steps before "
             "spilling them to the cache (default: half of available memory).",
    )
```

Add `"ram_budget_gb": args.ram_budget` to the record settings and `ram_budget=None if args.ram_budget is None else int(args.ram_budget * 2**30)` to the `run_flow(...)` call.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_budget.py tests/test_executor.py tests/test_cli_run.py -q`
Expected: all pass.

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/budget.py src/karak/flow/executor.py src/karak/flow/__main__.py tests/test_budget.py tests/test_executor.py tests/test_cli_run.py
git commit -m "feat: total RAM budget for held payloads replaces the 256 MB spill threshold"
```

---

### Task 6: Docs, changelog, and the real-data check

**Files:**
- Modify: `docs/user_guide.md` (caching paragraph around lines 252-270; the CLI synopsis around 276-282; the dashboard paragraph 290-298), `CHANGELOG.md` (Unreleased / Added), `CLAUDE.md` (the commands block: add the two flags)
- Test: `uv run pytest`, then the NWA 4587 run

- [ ] **Step 1: Write the user guide paragraphs**

In the caching section of `docs/user_guide.md`, after the paragraph that explains the recipe hash, add:

```markdown
Cache files are written on a background thread, so the next step starts
while the previous step's outputs are still being written; a run waits for
the writer before it returns (also after a failure or Ctrl-C, so completed
outputs always land). Files use the HDF5 `lzf` filter by default
(`--cache-compression gzip` for smaller files, `none` for the fastest
writes and reads; readers accept any). Each finished write appears in the
log lines as `cache: dn.cube written in 10.8 s`.

Between steps, outputs stay in RAM up to a budget, half of the available
memory by default (`--ram-budget GB` to change it). An output that would
exceed the budget is spilled: it is not held, and each consumer reads it
back from the cache; the log says so (`store: dn.cube (1.98 GB) spilled to
cache, budget 11.0 GB`). With `--no-cache` nothing is written or spilled.
```

Add `[--cache-compression lzf|gzip|none] [--ram-budget GB]` to the `karak run` synopsis line, and the same two flags to the commands block in `CLAUDE.md` with one-line comments.

- [ ] **Step 2: Write the changelog entries**

Under `## [Unreleased]` / `### Added` in `CHANGELOG.md`:

```markdown
- Cache writes run on a background thread: the next step starts while the
  previous outputs are written; the run drains the writer before it
  returns, also on failure and Ctrl-C. Log lines report each write.
- Cache files default to the HDF5 `lzf` filter (2.4x faster writes than
  gzip on the NWA 4587 cube at 1.5x the size); `karak run
  --cache-compression {lzf,gzip,none}`. Datasets are chunked (512, 512).
  Existing gzip cache files stay readable.
- A total RAM budget for outputs held between steps (`--ram-budget GB`,
  default half of the available memory) replaces the fixed 256 MB
  per-payload spill threshold; spills are logged.
```

- [ ] **Step 3: Run the suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 4: Real-data check on NWA 4587 (CPU)**

Run:

```bash
uv run karak run flows/nwa4587_denoise.json \
    --input /home/brendon/Dropbox/Projects/izawa/NWA_4587_data \
    --out output/nwa4587/run --workers 0 --plain --no-cache 2>&1 | tail -20
uv run karak run flows/nwa4587_denoise.json \
    --input /home/brendon/Dropbox/Projects/izawa/NWA_4587_data \
    --out output/nwa4587/run --workers 0 --plain 2>&1 | tail -20
```

Record in the PR body: `dn` seconds (compute) and its `cache: dn.cube written in ... s` line, against the 2026-09-29 baseline (`dn` 70.8 s including the write), and whether `src.cube` (1.98 GB) stayed in RAM (no `spilled` line) on this 31 GB machine. Note the file size of the new `lzf` cube in `output/nwa4587/work/cache/`.

- [ ] **Step 5: Commit, push, open the PR**

```bash
git add docs/user_guide.md CHANGELOG.md CLAUDE.md
git commit -m "docs: background cache writer, lzf cache files, RAM budget"
git push -u origin HEAD
gh pr create --title "feat: background cache writer, lzf cache files, RAM budget" --body "<what changed, why, tests, the NWA 4587 numbers, rulings>"
gh pr checks --watch
```

Stop there; the user reviews and merges.
