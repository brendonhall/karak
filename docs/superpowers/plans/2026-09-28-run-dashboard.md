# Live Run Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `karak run` shows a live Rich dashboard (run context, parameters, in-step progress, outputs, cache status, memory, log lines), with a line-based fallback, starting with a one-node `stepwise` flow that runs the load stage.

**Architecture:** The executor and the load stage emit richer events through the existing duck-typed `Reporter` seam in `flow/events.py`; the executor calls new events through a tolerant helper so old reporters keep working. Payloads describe themselves with `summary()`, and the cache stores that text next to each payload so cached steps can show it without loading gigabytes. Rich stays in `cli/`: `DashboardReporter` (live) and the extended `RichReporter` (plain) consume the events; `cli/main.py` picks one, captures `karak` log records, and maps failures to exit codes.

**Tech Stack:** Python ≥3.12, numpy, Rich 14 (`Live(get_renderable=...)`), pytest, stdlib `logging`, `threading`, `resource`.

**Spec:** `docs/superpowers/specs/2026-09-26-run-dashboard-design.md`

## Global Constraints

- Rich is imported only under `src/karak/cli/`. Nothing in `flow/`, `stages/`, or `io/` imports Rich.
- The numeric core (`io/`) knows nothing about reporters; it takes plain callbacks.
- Payloads stay immutable; `summary()` only reads.
- Old reporters that define only `node_started`, `node_finished`, `progress`, `log` must still run a flow.
- Memory sampling uses the standard library only (no psutil).
- Version string: `importlib.metadata.version("karak")`, fallback `"unknown"`.
- The growing flow is named `stepwise`; its first and only node is `src` (`load_elements`, `input_dir="{input}"`).
- Exit codes: 0 success, 1 flow failure, 130 Ctrl-C.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push to `main` after each commit (project rule).
- Run tests with `uv run pytest`.

## Review Focus

1. **A step whose output nobody consumes is cached** (the `stepwise` case today): the executor loads no payload for it, yet the dashboard must still show its output summary. Pinned in Task 3 (`test_cached_leaf_still_reports_summaries`).
2. **Cache entries written before this change** have no summary sidecar: a cached step must show a placeholder, not crash. Pinned in Task 3 (`test_cached_entry_without_sidecar_shows_placeholder`).
3. **Very long parameter values** (the export sink's `{flow}` param holds the whole flow JSON): the panel must truncate them. Pinned in Task 7 and Task 8 (`test_long_param_values_are_truncated`).
4. **Failure before the first node runs** (validation error): `run_started` never fires; `close()` and the error message must still work and logging must be restored. Pinned in Task 9 (`test_run_validation_failure_exits_1`).
5. **`close()` called more than once** (the `except` branch and then `finally`): it must be idempotent and print "interrupted" at most once. Pinned in Task 7 and Task 8 (`test_close_is_idempotent`).

## File Structure

- `src/karak/stages/payloads.py` (modify): `format_bytes`, `summary()` on all 8 payload classes.
- `src/karak/flow/cache.py` (modify): `store_summary`, `load_summary` sidecar text files.
- `src/karak/flow/events.py` (modify): `RunInfo`, `ParamValue`, six new protocol methods, `NullReporter` no-ops.
- `src/karak/flow/executor.py` (modify): emit new events, set `stage.node_id`, tolerant `_emit`.
- `src/karak/stages/base.py` (modify): `node_id` attribute.
- `src/karak/io/loaders.py` (modify): `on_file` callback, discovery log line.
- `src/karak/stages/load.py` (modify): forward per-file progress to the reporter.
- `src/karak/flow/builtins.py`, `src/karak/flow/flows/stepwise.json` (modify/create): `stepwise` flow.
- `src/karak/cli/memory.py` (create): RSS sampling.
- `src/karak/cli/logs.py` (create): log capture context manager.
- `src/karak/cli/reporter.py` (modify): plain lines for new events, `close()`.
- `src/karak/cli/dashboard.py` (create): `DashboardReporter`.
- `src/karak/flow/__main__.py`, `src/karak/cli/main.py` (modify): `--plain`, reporter selection, log capture, exit codes.
- Docs: `docs/user_guide.md`, `CHANGELOG.md`, `CLAUDE.md`.
- Tests: `tests/test_payload_summary.py`, `tests/test_cache.py`, `tests/test_executor.py`, `tests/test_loader_workers.py`, `tests/test_builtins.py`, `tests/test_cli_memory.py`, `tests/test_cli_logs.py`, `tests/test_cli_reporter.py`, `tests/test_cli_dashboard.py`, `tests/test_cli_run.py`.

---

### Task 1: Payload summaries

**Files:**
- Modify: `src/karak/stages/payloads.py`
- Test: `tests/test_payload_summary.py` (create)

**Interfaces:**
- Produces: `karak.stages.payloads.format_bytes(n: int) -> str`; `summary() -> str` on `ElementCube`, `BseImage`, `MaskSet`, `PCAFeatures`, `Labels`, `TiledArtifacts`, `ClusterStats`, `Fingerprints`. Task 3 calls `summary()`; `ClusterStats(stats={"value": 12}).summary() == "ClusterStats 1 entries"` is relied on by Task 3 tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_payload_summary.py`:

```python
"""One-line summaries that the run dashboard shows for each output."""

from __future__ import annotations

import numpy as np

from karak.stages.payloads import (
    BseImage,
    ClusterStats,
    ElementCube,
    Fingerprints,
    Labels,
    LabelState,
    MaskSet,
    PCAFeatures,
    Space,
    TiledArtifacts,
    format_bytes,
)


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2_400) == "2 kB"
    assert format_bytes(104_000_000) == "104 MB"
    assert format_bytes(1_978_000_000) == "1.98 GB"


def test_element_cube_summary():
    cube = ElementCube(
        pixels=np.zeros((10, 20, 3), np.float32),
        element_names=("A", "B", "C"),
        space=Space.RAW,
    )
    assert cube.summary() == "ElementCube 10×20×3 float32 2 kB space=raw"


def test_bse_summary():
    bse = BseImage(pixels=np.zeros((1000, 1000), np.float32))
    assert bse.summary() == "BseImage 1000×1000 float32 4 MB"


def test_mask_set_summary():
    mineral = np.zeros((10, 10), bool)
    mineral.flat[:25] = True
    valid = np.zeros((10, 10), bool)
    valid[:5] = True
    assert MaskSet(mineral_mask=mineral).summary() == "MaskSet mineral 25.0%"
    assert (
        MaskSet(mineral_mask=mineral, valid_mask=valid).summary()
        == "MaskSet mineral 25.0% · valid 50.0%"
    )


def test_pca_summary():
    pca = PCAFeatures(
        features=np.zeros((100, 3), np.float32),
        mineral_indices=np.zeros((100, 2), np.int32),
        image_shape=(10, 10),
        explained_variance_ratio=np.array([0.5, 0.3, 0.1, 0.1]),
        n_kept=3,
    )
    assert pca.summary() == "PCAFeatures 100 px × 3 components (90.0% variance)"


def test_labels_summary():
    labels = Labels(
        labels=np.array([0, 0, 1, 2, -1, -1], np.int32),
        probabilities=None,
        mineral_indices=np.zeros((6, 2), np.int32),
        image_shape=(2, 3),
        state=LabelState.RAW,
    )
    assert labels.summary() == "Labels 6 px · 3 phases · 2 noise · state=raw"


def test_labels_summary_without_noise():
    labels = Labels(
        labels=np.array([0, 1], np.int32),
        probabilities=None,
        mineral_indices=np.zeros((2, 2), np.int32),
        image_shape=(1, 2),
        state=LabelState.CLEANED,
    )
    assert labels.summary() == "Labels 2 px · 2 phases · state=cleaned"


def test_tiled_cluster_stats_fingerprints_summaries():
    tiled = TiledArtifacts(tile_results=(), phase_registry=(), tile_size=1024)
    assert tiled.summary() == "TiledArtifacts 0 tiles · 0 phases · tile_size=1024"
    assert ClusterStats(stats={"a": 1, "b": 2}).summary() == "ClusterStats 2 entries"
    fp = Fingerprints(data={"p1": {}}, similar_pairs=[])
    assert fp.summary() == "Fingerprints 1 entries · 0 similar pairs"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_payload_summary.py -v`
Expected: FAIL with `ImportError: cannot import name 'format_bytes'`.

- [ ] **Step 3: Implement**

In `src/karak/stages/payloads.py`, add after `class _Replaceable` (before the first `@_payload`):

```python
def format_bytes(n: int) -> str:
    """Human-readable byte count: '1.98 GB', '104 MB', '2 kB', '512 B'."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{n} B"


def _shape(shape) -> str:
    return "×".join(str(int(s)) for s in shape)
```

Add one `summary` method to each class (place it after the dataclass fields, before `to_h5`):

`ElementCube`:
```python
    def summary(self) -> str:
        return (
            f"ElementCube {_shape(self.pixels.shape)} {self.pixels.dtype} "
            f"{format_bytes(self.pixels.nbytes)} space={self.space.value}"
        )
```

`BseImage`:
```python
    def summary(self) -> str:
        return (
            f"BseImage {_shape(self.pixels.shape)} {self.pixels.dtype} "
            f"{format_bytes(self.pixels.nbytes)}"
        )
```

`MaskSet`:
```python
    def summary(self) -> str:
        text = f"MaskSet mineral {100 * self.mineral_mask.mean():.1f}%"
        if self.valid_mask is not None:
            text += f" · valid {100 * self.valid_mask.mean():.1f}%"
        return text
```

`PCAFeatures`:
```python
    def summary(self) -> str:
        kept = float(np.sum(self.explained_variance_ratio[: self.n_kept]))
        return (
            f"PCAFeatures {self.features.shape[0]:,} px × {self.n_kept} "
            f"components ({kept:.1%} variance)"
        )
```

`Labels`:
```python
    def summary(self) -> str:
        labels = self.labels
        phases = np.unique(labels[labels >= 0]).size
        noise = int((labels < 0).sum())
        text = f"Labels {labels.size:,} px · {phases} phases"
        if noise:
            text += f" · {noise:,} noise"
        return text + f" · state={self.state.value}"
```

`TiledArtifacts`:
```python
    def summary(self) -> str:
        return (
            f"TiledArtifacts {len(self.tile_results)} tiles · "
            f"{len(self.phase_registry)} phases · tile_size={self.tile_size}"
        )
```

`ClusterStats`:
```python
    def summary(self) -> str:
        return f"ClusterStats {len(self.stats)} entries"
```

`Fingerprints`:
```python
    def summary(self) -> str:
        return (
            f"Fingerprints {len(self.data)} entries · "
            f"{len(self.similar_pairs)} similar pairs"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_payload_summary.py tests/test_payloads.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/stages/payloads.py tests/test_payload_summary.py
git commit -m "feat: one-line summary() on every payload

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 2: Summary sidecars in the cache

**Files:**
- Modify: `src/karak/flow/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Produces: `store_summary(recipe: str, port: str, text: str, cache_dir) -> None` and `load_summary(recipe: str, port: str, cache_dir) -> str | None` in `karak.flow.cache`. Files are `<cache_dir>/<recipe>__<port>.summary.txt`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache.py`:

```python
def test_summary_sidecar_roundtrip(tmp_path):
    from karak.flow.cache import load_summary, store_summary

    store_summary("deadbeef", "cube", "ElementCube 2×2×1 float32 16 B space=raw", tmp_path)
    assert load_summary("deadbeef", "cube", tmp_path) == (
        "ElementCube 2×2×1 float32 16 B space=raw"
    )
    assert (tmp_path / "deadbeef__cube.summary.txt").exists()


def test_missing_summary_returns_none(tmp_path):
    from karak.flow.cache import load_summary

    assert load_summary("nope", "cube", tmp_path) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cache.py -v -k summary`
Expected: FAIL with `ImportError: cannot import name 'load_summary'`.

- [ ] **Step 3: Implement**

Append to `src/karak/flow/cache.py`:

```python
def _summary_path(recipe: str, port: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{recipe}__{port}.summary.txt"


def store_summary(
    recipe: str, port: str, text: str, cache_dir: str | Path
) -> None:
    """Store a payload's one-line summary next to its cached payload.

    A cached step whose output nobody consumes is never loaded, so the
    run dashboard reads this text instead of the payload.
    """
    path = _summary_path(recipe, port, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_summary(recipe: str, port: str, cache_dir: str | Path) -> str | None:
    """The stored summary text, or None for entries cached without one."""
    path = _summary_path(recipe, port, cache_dir)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/cache.py tests/test_cache.py
git commit -m "feat: cache stores a summary sidecar per payload

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 3: Reporter events from the executor

**Files:**
- Modify: `src/karak/flow/events.py`, `src/karak/flow/executor.py`, `src/karak/stages/base.py`
- Test: `tests/test_executor.py`

**Interfaces:**
- Consumes: `summary()` (Task 1), `store_summary`/`load_summary` (Task 2).
- Produces (in `karak.flow.events`):
  - `RunInfo(flow: str, input_path: str, out_base: str, work_dir: str, cache: bool, workers: int | None, device: str, version: str, nodes: tuple[tuple[str, str], ...])`, frozen dataclass; `nodes` is `(node_id, stage_type)` pairs in execution order, skipped sinks excluded.
  - `ParamValue(name: str, value: object, is_default: bool)`, frozen dataclass.
  - Reporter methods: `run_started(info: RunInfo)`, `node_params(node_id: str, params: list[ParamValue])`, `node_cache(node_id: str, recipe_hash: str, cached: bool, cache_dir: str)`, `node_outputs(node_id: str, summaries: dict[str, str])`, `node_failed(node_id: str, message: str)`, `run_finished(summary: dict, seconds: float)`.
  - Per-node event order: `node_params`, `node_cache`, `node_started` (only if it runs), `progress`*, `node_outputs` (only if the node has outputs), `node_finished`. On error: `node_failed`, then `FlowError` is raised.
  - `Stage.node_id: str` set by the executor before `run()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_executor.py`:

```python
class FullRecorder:
    """Records every reporter event, new and old."""

    def __init__(self):
        self.events = []

    def run_started(self, info):
        self.events.append(("run_started", info))

    def node_params(self, node_id, params):
        self.events.append(("params", node_id, params))

    def node_cache(self, node_id, recipe_hash, cached, cache_dir):
        self.events.append(("cache", node_id, recipe_hash, cached, cache_dir))

    def node_started(self, node_id, label):
        self.events.append(("start", node_id))

    def progress(self, node_id, done, total, msg=""):
        self.events.append(("progress", node_id, done, total, msg))

    def log(self, level, msg):
        pass

    def node_outputs(self, node_id, summaries):
        self.events.append(("outputs", node_id, summaries))

    def node_finished(self, node_id, seconds, cached):
        self.events.append(("finish", node_id, cached))

    def node_failed(self, node_id, message):
        self.events.append(("failed", node_id, message))

    def run_finished(self, summary, seconds):
        self.events.append(("run_finished", summary, seconds))

    def of(self, kind, node_id=None):
        return [e for e in self.events
                if e[0] == kind and (node_id is None or e[1] == node_id)]


def test_full_event_sequence(tmp_path):
    rec = FullRecorder()
    run(_chain_graph(), input_path="/in", out_base="o", work_dir=str(tmp_path),
        reporter=rec)
    kinds = [e[:1] if e[0] in ("run_started", "run_finished") else e[:2]
             for e in rec.events]
    assert kinds == [
        ("run_started",),
        ("params", "src"), ("cache", "src"), ("start", "src"),
        ("outputs", "src"), ("finish", "src"),
        ("params", "plus"), ("cache", "plus"), ("start", "plus"),
        ("outputs", "plus"), ("finish", "plus"),
        ("params", "out"), ("cache", "out"), ("start", "out"),
        ("finish", "out"),
        ("run_finished",),
    ]
    info = rec.events[0][1]
    assert info.flow == "chain"
    assert info.input_path == "/in"
    assert info.out_base == "o"
    assert info.work_dir == str(tmp_path)
    assert info.cache is True
    assert info.workers is None
    assert info.device == "cpu"
    assert info.version
    assert info.nodes == (
        ("src", "fake_source"), ("plus", "fake_add"), ("out", "fake_sink"),
    )
    assert rec.of("outputs", "plus")[0][2] == {"num": "ClusterStats 1 entries"}


def test_node_params_resolve_tokens_and_flag_defaults(tmp_path):
    rec = FullRecorder()
    run(_chain_graph(), input_path="/in", out_base="o", work_dir=str(tmp_path),
        reporter=rec)
    src = {p.name: p for p in rec.of("params", "src")[0][2]}
    assert (src["value"].value, src["value"].is_default) == (10, False)
    assert (src["path"].value, src["path"].is_default) == ("/in", False)
    out = {p.name: p for p in rec.of("params", "out")[0][2]}
    assert (out["out"].value, out["out"].is_default) == ("o", True)


def test_cached_leaf_still_reports_summaries(tmp_path):
    leaf = Graph(name="leaf", nodes=(Node("src", "fake_source", {"value": 3}),))
    first = FullRecorder()
    run(leaf, work_dir=str(tmp_path), reporter=first)
    second = FullRecorder()
    run(leaf, work_dir=str(tmp_path), reporter=second)

    hash1, cached1 = first.of("cache", "src")[0][2:4]
    hash2, cached2 = second.of("cache", "src")[0][2:4]
    assert hash1 == hash2
    assert (cached1, cached2) == (False, True)
    assert second.of("start", "src") == []
    assert second.of("outputs", "src")[0][2] == {"num": "ClusterStats 1 entries"}
    assert second.of("finish", "src")[0][2] is True


def test_cached_entry_without_sidecar_shows_placeholder(tmp_path):
    leaf = Graph(name="leaf", nodes=(Node("src", "fake_source", {"value": 3}),))
    run(leaf, work_dir=str(tmp_path))
    for sidecar in (tmp_path / "cache").glob("*.summary.txt"):
        sidecar.unlink()
    rec = FullRecorder()
    run(leaf, work_dir=str(tmp_path), reporter=rec)
    assert rec.of("outputs", "src")[0][2] == {"num": "(no summary recorded)"}


def test_old_style_reporter_still_works(tmp_path):
    class OldReporter:
        def __init__(self):
            self.finished = []

        def node_started(self, node_id, label):
            pass

        def node_finished(self, node_id, seconds, cached):
            self.finished.append(node_id)

        def progress(self, node_id, done, total, msg=""):
            pass

        def log(self, level, msg):
            pass

    old = OldReporter()
    run(_chain_graph(), input_path="/in", out_base="o", work_dir=str(tmp_path),
        reporter=old)
    assert old.finished == ["src", "plus", "out"]


class FakeFail(Stage):
    id = "fake_fail"
    label = "Fake fail"
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        raise ValueError("boom")


def test_failing_node_reports_failure(tmp_path):
    registry.register(FakeFail)
    try:
        rec = FullRecorder()
        graph = Graph(name="fail", nodes=(Node("bad", "fake_fail"),))
        with pytest.raises(FlowError, match="boom"):
            run(graph, work_dir=str(tmp_path), reporter=rec)
        assert rec.of("failed", "bad") == [("failed", "bad", "boom")]
        assert rec.of("run_finished") == []
    finally:
        registry._REGISTRY.pop("fake_fail", None)


class FakeNodeIdProbe(Stage):
    id = "fake_node_id_probe"
    label = "Node id probe"
    OUTPUTS: list = []

    def apply(self, inputs, params):
        RECORD.append(("node_id", self.node_id))
        return {}


def test_executor_sets_node_id(tmp_path):
    registry.register(FakeNodeIdProbe)
    try:
        graph = Graph(name="probe", nodes=(Node("p1", "fake_node_id_probe"),))
        run(graph, work_dir=str(tmp_path))
        assert ("node_id", "p1") in RECORD
    finally:
        registry._REGISTRY.pop("fake_node_id_probe", None)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_executor.py -v`
Expected: the new tests FAIL (no `run_started` events; `node_id` attribute missing); the existing tests still PASS.

- [ ] **Step 3: Extend `flow/events.py`**

Replace the whole file with:

```python
"""Progress/event seam between the executor and any UI.

Stages and the executor emit through a Reporter; Rich lives only in the CLI
front-end. The protocol is duck-typed — any object with these methods works.
The executor calls every method added after ``node_started``,
``node_finished``, ``progress`` and ``log`` only if the reporter defines it,
so older reporters keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunInfo:
    """Run context, sent once before the first node."""

    flow: str
    input_path: str
    out_base: str
    work_dir: str
    cache: bool
    workers: int | None
    device: str
    version: str
    nodes: tuple  # ((node_id, stage_type), ...) in execution order


@dataclass(frozen=True)
class ParamValue:
    """One resolved node parameter; ``is_default`` compares to the stage default."""

    name: str
    value: object
    is_default: bool


class Reporter(Protocol):
    def run_started(self, info: RunInfo) -> None: ...

    def node_params(self, node_id: str, params: list[ParamValue]) -> None: ...

    def node_cache(
        self, node_id: str, recipe_hash: str, cached: bool, cache_dir: str
    ) -> None: ...

    def node_started(self, node_id: str, label: str) -> None: ...

    def progress(
        self, node_id: str, done: int, total: int, msg: str = ""
    ) -> None: ...

    def log(self, level: str, msg: str) -> None: ...

    def node_outputs(self, node_id: str, summaries: dict[str, str]) -> None: ...

    def node_finished(
        self, node_id: str, seconds: float, cached: bool
    ) -> None: ...

    def node_failed(self, node_id: str, message: str) -> None: ...

    def run_finished(self, summary: dict, seconds: float) -> None: ...


class NullReporter:
    def run_started(self, info: RunInfo) -> None:
        pass

    def node_params(self, node_id: str, params: list[ParamValue]) -> None:
        pass

    def node_cache(
        self, node_id: str, recipe_hash: str, cached: bool, cache_dir: str
    ) -> None:
        pass

    def node_started(self, node_id: str, label: str) -> None:
        pass

    def progress(
        self, node_id: str, done: int, total: int, msg: str = ""
    ) -> None:
        pass

    def log(self, level: str, msg: str) -> None:
        pass

    def node_outputs(self, node_id: str, summaries: dict[str, str]) -> None:
        pass

    def node_finished(
        self, node_id: str, seconds: float, cached: bool
    ) -> None:
        pass

    def node_failed(self, node_id: str, message: str) -> None:
        pass

    def run_finished(self, summary: dict, seconds: float) -> None:
        pass
```

- [ ] **Step 4: Add `node_id` to `Stage`**

In `src/karak/stages/base.py`, below `workers: int | None = None       # injected by the flow executor; None = serial`, add:

```python
    node_id: str = ""               # injected by the flow executor
```

- [ ] **Step 5: Emit the events in `flow/executor.py`**

(a) Change the imports:

```python
from karak.flow.cache import (
    has_payload,
    load_payload,
    load_summary,
    recipe_hash,
    store_payload,
    store_summary,
)
from karak.flow.events import NullReporter, ParamValue, RunInfo
```

(b) Add these helpers above `def run(`:

```python
def _emit(reporter, event: str, *args) -> None:
    """Call an optional reporter method; older reporters may lack it."""
    method = getattr(reporter, event, None)
    if method is not None:
        method(*args)


def _karak_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("karak")
    except PackageNotFoundError:
        return "unknown"


def _summarize(payload) -> str:
    summary = getattr(payload, "summary", None)
    return summary() if callable(summary) else type(payload).__name__
```

(c) In `run()`, right after `summary: dict = {}`, add:

```python
    run_started_at = time.monotonic()
    order = [n for n in _topo_order(graph) if n not in skipped]
    devices = set()
    for node_id in order:
        node = graph.node(node_id)
        coerced = registry.get(node.type).coerce_params(node.params)
        if "device" in coerced:
            devices.add(str(coerced["device"]))
    _emit(reporter, "run_started", RunInfo(
        flow=graph.name,
        input_path=input_path,
        out_base=out_base,
        work_dir=str(work_dir),
        cache=cache,
        workers=workers,
        device=",".join(sorted(devices)) or "cpu",
        version=_karak_version(),
        nodes=tuple((n, graph.node(n).type) for n in order),
    ))
```

(d) Inside the loop, right after the line `hashes[(node_id, port.name)] = node_hash` loop (after `for port in cls.OUTPUTS: ...`), add:

```python
        defaults = _resolve_tokens(cls.coerce_params({}), tokens)
        _emit(reporter, "node_params", node_id, [
            ParamValue(p.name, params[p.name], params[p.name] == defaults[p.name])
            for p in cls.PARAMS
        ])
```

(e) Right after the `cached_hit = (...)` expression, add:

```python
        _emit(reporter, "node_cache", node_id, node_hash, cached_hit,
              str(cache_dir))
```

(f) In the cached branch, after the `outputs = {...}` dict, add:

```python
            summaries = {
                port.name: load_summary(node_hash, port.name, cache_dir)
                or (_summarize(outputs[port.name]) if port.name in outputs
                    else "(no summary recorded)")
                for port in cls.OUTPUTS
            }
```

(g) Replace the run branch from `stage = cls()` to the end of the `if cache and not is_sink:` block with:

```python
            stage = cls()
            stage.reporter = reporter
            stage.workers = workers
            stage.node_id = node_id
            reporter.node_started(node_id, cls.label or cls.id)
            try:
                outputs = stage.run(inputs, params)
            except Exception as exc:
                _emit(reporter, "node_failed", node_id, str(exc))
                raise FlowError(f"node {node_id!r} ({node.type}): {exc}") from exc
            summaries = {name: _summarize(p) for name, p in outputs.items()}
            if cache and not is_sink:
                for port_name, payload in outputs.items():
                    store_payload(node_hash, port_name, payload, cache_dir)
                    store_summary(node_hash, port_name, summaries[port_name],
                                  cache_dir)
```

(h) Replace the tail of the loop (from `elapsed = time.monotonic() - started`) and the final `return summary` with:

```python
        elapsed = time.monotonic() - started
        if summaries:
            _emit(reporter, "node_outputs", node_id, summaries)
        reporter.node_finished(node_id, elapsed, cached_hit)
        summary[node_id] = {"cached": cached_hit, "seconds": elapsed}

    _emit(reporter, "run_finished", summary, time.monotonic() - run_started_at)
    return summary
```

The loop header `for node_id in _topo_order(graph):` and the skipped-node branch stay as they are.

- [ ] **Step 6: Run the executor tests**

Run: `uv run pytest tests/test_executor.py -v`
Expected: all PASS, including the old `test_reporter_receives_events`.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -p no:warnings`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/karak/flow/events.py src/karak/flow/executor.py src/karak/stages/base.py tests/test_executor.py
git commit -m "feat: executor emits run, params, cache, outputs and failure events

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 4: Per-file progress from the load stage

**Files:**
- Modify: `src/karak/io/loaders.py` (`load_element_maps`), `src/karak/stages/load.py` (`apply`)
- Test: `tests/test_loader_workers.py`

**Interfaces:**
- Consumes: `Stage.node_id`, `Stage.reporter` (Task 3).
- Produces: `load_element_maps(..., workers: int = 1, on_file: Callable[[int, int, str], None] | None = None)`; `on_file(done, total, element)` fires once per loaded file (elements and BSE), `done` counts 1..total in completion order. Log line `"Found %d matching files: %d to load (BSE channel %r), excluded: %s"`. The load stage calls `reporter.progress(node_id, done, total, element)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_loader_workers.py`:

```python
@pytest.mark.parametrize("workers", [1, 2])
def test_on_file_fires_once_per_file(scene_dir, workers):
    from karak.config import LoaderConfig

    data_dir, colormap = scene_dir
    calls = []
    load_element_maps(
        str(data_dir),
        DownsampleConfig(header_trim_px=0, downsample_factor=1),
        exclude_elements=[],
        loader_config=LoaderConfig(colormap=colormap),
        workers=workers,
        on_file=lambda done, total, element: calls.append((done, total, element)),
    )
    assert [c[0] for c in calls] == [1, 2, 3, 4]
    assert {c[1] for c in calls} == {4}
    assert sorted(c[2] for c in calls) == ["A", "B", "C", "SEM"]


def test_loader_logs_discovery_line(scene_dir, caplog):
    import logging

    from karak.config import LoaderConfig

    data_dir, colormap = scene_dir
    with caplog.at_level(logging.INFO, logger="karak.io.loaders"):
        load_element_maps(
            str(data_dir),
            DownsampleConfig(header_trim_px=0, downsample_factor=1),
            exclude_elements=["C"],
            loader_config=LoaderConfig(colormap=colormap),
        )
    assert (
        "Found 4 matching files: 3 to load (BSE channel 'SEM'), excluded: C"
        in caplog.text
    )


def test_load_stage_reports_progress(scene_dir):
    from karak.stages.load import LoadElementsStage

    data_dir, colormap = scene_dir
    events = []

    class Recorder:
        def progress(self, node_id, done, total, msg=""):
            events.append((node_id, done, total, msg))

    stage = LoadElementsStage()
    stage.reporter = Recorder()
    stage.node_id = "src"
    stage.run({}, {"input_dir": str(data_dir), "colormap": colormap,
                   "exclude_elements": ""})
    assert [e[1] for e in events] == [1, 2, 3, 4]
    assert all(e[0] == "src" and e[2] == 4 for e in events)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_loader_workers.py -v`
Expected: the three new tests FAIL (`unexpected keyword argument 'on_file'`, missing log line, no progress events).

- [ ] **Step 3: Implement in `io/loaders.py`**

(a) Add `from collections.abc import Callable` to the imports at the top.

(b) Add the parameter after `workers: int = 1,`:

```python
    on_file: Callable[[int, int, str], None] | None = None,
```

and document it in the docstring after the `workers` entry:

```
    on_file : callable, optional
        Called as ``on_file(done, total, element)`` after each file is
        loaded (elements and BSE alike), in completion order. Lets callers
        show progress without the loader knowing about any UI.
```

(c) In the decide loop, record excluded names. Before `jobs: list[tuple[str, str, str]] = []` add `excluded: list[str] = []`, and change the excluded branch to:

```python
        elif element in skip:
            logger.info("Skipping excluded channel: %s", element)
            excluded.append(element)
```

(d) Directly after the decide loop (before the `# Execute:` comment) add:

```python
    logger.info(
        "Found %d matching files: %d to load (BSE channel %r), excluded: %s",
        len(files), len(jobs), bse_channel, ", ".join(excluded) or "none",
    )
    total = len(jobs)
```

(e) Replace the execute block (from `if workers > 1 and jobs:` through the end of the serial `else:` branch) with:

```python
    if workers > 1 and jobs:
        # Warm the LUT cache once so pool workers load it from disk
        # instead of each building a fresh one.
        get_full_lut(loader.colormap, base_dir=input_dir)
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        mp_context = multiprocessing.get_context("forkserver")
        arrays: list = [None] * len(jobs)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp_context,
        ) as pool:
            futures = {}
            for index, (kind, key, fpath) in enumerate(jobs):
                if kind == "bse":
                    future = pool.submit(_process_bse_file, fpath, factor, trims)
                else:
                    future = pool.submit(
                        _process_element_file, fpath, loader.colormap,
                        input_dir, factor, trims,
                    )
                futures[future] = index
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                arrays[index] = future.result()
                if on_file is not None:
                    on_file(done, total, jobs[index][1])
    else:
        arrays = []
        for kind, key, fpath in jobs:
            if kind == "bse":
                arrays.append(_process_bse_file(fpath, factor, trims))
            else:
                arrays.append(_process_element_file(
                    fpath, loader.colormap, input_dir, factor, trims,
                ))
            if on_file is not None:
                on_file(len(arrays), total, key)
```

- [ ] **Step 4: Forward progress in `stages/load.py`**

In `LoadElementsStage.apply`, before the `elements, bse, names = load_element_maps(` call, add:

```python
        reporter, node_id = self.reporter, self.node_id

        def on_file(done: int, total: int, element: str) -> None:
            if reporter is not None:
                reporter.progress(node_id, done, total, element)
```

and pass `on_file=on_file,` as the last argument of the `load_element_maps(...)` call.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_loader_workers.py tests/test_stage_parity.py tests/test_jet_inversion.py -v`
Expected: all PASS; `test_parallel_load_is_byte_identical` still passes.

- [ ] **Step 6: Commit**

```bash
git add src/karak/io/loaders.py src/karak/stages/load.py tests/test_loader_workers.py
git commit -m "feat: load stage reports per-file progress and a discovery line

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 5: The `stepwise` flow

**Files:**
- Modify: `src/karak/flow/builtins.py`, `tests/test_builtins.py`
- Create: `src/karak/flow/flows/stepwise.json`

**Interfaces:**
- Produces: `builtin_flow("stepwise")` → `Graph(name="stepwise", nodes=(Node("src", "load_elements", {"input_dir": "{input}"}),))`; `karak run --builtin stepwise` (the `--builtin` choices come from `builtin_names()`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_builtins.py`, change both parametrize lists from `["global", "tiled", "tiled-rare"]` to `["global", "tiled", "tiled-rare", "stepwise"]`, and append:

```python
def test_stepwise_starts_with_the_load_step():
    graph = builtin_flow("stepwise")
    assert [(n.id, n.type) for n in graph.nodes] == [("src", "load_elements")]
    assert graph.node("src").params == {"input_dir": "{input}"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_builtins.py -v`
Expected: the `stepwise` cases FAIL with `KeyError: 'stepwise'`.

- [ ] **Step 3: Implement**

In `src/karak/flow/builtins.py`, add after `tiled_rare_flow`:

```python
def stepwise_flow() -> Graph:
    """The flow the run dashboard is built against, one step at a time.

    It gains a node each time a pipeline step joins the dashboard work;
    today it holds only the load step.
    """
    return Graph(
        name="stepwise",
        nodes=(Node("src", "load_elements", {"input_dir": "{input}"}),),
    )
```

(`Node` is already imported in `builtins.py`.) Add `"stepwise": stepwise_flow,` to `_BUILTINS`.

Generate the shipped JSON:

```bash
uv run python -c "import json; from karak.flow.builtins import builtin_flow; print(json.dumps(builtin_flow('stepwise').to_json(), indent=2))" > src/karak/flow/flows/stepwise.json
```

In `src/karak/cli/main.py` and `src/karak/flow/__main__.py`, update the module docstring line `karak run (FLOW.json | --builtin global|tiled|tiled-rare) ...` to `... --builtin global|tiled|tiled-rare|stepwise ...`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_builtins.py tests/test_flow_cli.py -v && uv run karak validate --builtin stepwise`
Expected: tests PASS; the CLI prints `0 errors, ...` and exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/builtins.py src/karak/flow/flows/stepwise.json src/karak/cli/main.py src/karak/flow/__main__.py tests/test_builtins.py
git commit -m "feat: stepwise builtin flow, starting with the load step

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 6: Memory sampler and log capture

**Files:**
- Create: `src/karak/cli/memory.py`, `src/karak/cli/logs.py`
- Test: `tests/test_cli_memory.py`, `tests/test_cli_logs.py` (create)

**Interfaces:**
- Produces:
  - `karak.cli.memory.current_rss() -> int | None`, `peak_rss() -> int`, `MemorySampler(interval: float = 1.0)` with attributes `current: int | None`, `peak: int` and methods `start()`, `stop()` (idempotent), `sample()`.
  - `karak.cli.logs.capture_logs(reporter, logger_name: str = "karak")` context manager: forwards records at INFO and above to `reporter.log(levelname.lower(), message)`; restores handlers, level and `propagate` on exit, also after an exception.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_memory.py`:

```python
"""Process memory sampling for the dashboard."""

from __future__ import annotations

import sys

import pytest

from karak.cli.memory import MemorySampler, current_rss, peak_rss


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc is Linux-only")
def test_current_rss_is_positive_on_linux():
    assert current_rss() > 0


def test_peak_is_at_least_current():
    current = current_rss() or 0
    assert peak_rss() >= current > -1


def test_sampler_start_stop_is_idempotent():
    sampler = MemorySampler(interval=0.01)
    sampler.start()
    sampler.start()
    sampler.stop()
    sampler.stop()
    assert sampler.peak > 0
    assert sampler._thread is None
```

Create `tests/test_cli_logs.py`:

```python
"""karak log records reach the reporter only while a run is captured."""

from __future__ import annotations

import logging

import pytest

from karak.cli.logs import capture_logs


class Recorder:
    def __init__(self):
        self.lines = []

    def log(self, level, msg):
        self.lines.append((level, msg))


def test_capture_forwards_info_and_above():
    rec = Recorder()
    logger = logging.getLogger("karak.io.loaders")
    with capture_logs(rec):
        logger.info("hello %s", "x")
        logger.warning("careful")
        logger.debug("hidden")
    logger.info("after the run")
    assert rec.lines == [("info", "hello x"), ("warning", "careful")]


def test_capture_restores_logger_state_after_error():
    karak_logger = logging.getLogger("karak")
    before = (list(karak_logger.handlers), karak_logger.level,
              karak_logger.propagate)
    with pytest.raises(RuntimeError):
        with capture_logs(Recorder()):
            raise RuntimeError("stage blew up")
    after = (list(karak_logger.handlers), karak_logger.level,
             karak_logger.propagate)
    assert after == before
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_memory.py tests/test_cli_logs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'karak.cli.memory'`.

- [ ] **Step 3: Implement `cli/memory.py`**

```python
"""Process memory sampling for the run dashboard (standard library only).

With --workers, pool processes are not counted; the dashboard labels the
figure "main process" in that case.
"""

from __future__ import annotations

import os
import sys
import threading


def current_rss() -> int | None:
    """Resident set size of this process in bytes, or None without /proc."""
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def peak_rss() -> int:
    """Peak resident set size of this process in bytes (0 if unknown)."""
    try:
        import resource
    except ImportError:
        return current_rss() or 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class MemorySampler:
    """Samples current and peak RSS on a daemon thread."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.current: int | None = current_rss()
        self.peak: int = peak_rss()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        self.current = current_rss()
        self.peak = peak_rss()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="karak-memory", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.sample()
```

- [ ] **Step 4: Implement `cli/logs.py`**

```python
"""Route karak log records to a reporter for the duration of a run."""

from __future__ import annotations

import logging
from contextlib import contextmanager


class ReporterLogHandler(logging.Handler):
    def __init__(self, reporter):
        super().__init__(level=logging.INFO)
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.reporter.log(record.levelname.lower(), record.getMessage())
        except Exception:
            self.handleError(record)


@contextmanager
def capture_logs(reporter, logger_name: str = "karak"):
    """Send ``karak`` INFO+ records to ``reporter.log`` while the block runs.

    Propagation is switched off meanwhile so records are not printed a
    second time underneath a live display; everything is restored on exit.
    """
    logger = logging.getLogger(logger_name)
    handler = ReporterLogHandler(reporter)
    saved = (logger.level, logger.propagate)
    logger.addHandler(handler)
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_cli_memory.py tests/test_cli_logs.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/karak/cli/memory.py src/karak/cli/logs.py tests/test_cli_memory.py tests/test_cli_logs.py
git commit -m "feat: memory sampler and log capture for the run display

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 7: Plain reporter for the new events

**Files:**
- Modify: `src/karak/cli/reporter.py`
- Test: `tests/test_cli_reporter.py` (create)

**Interfaces:**
- Consumes: `RunInfo`, `ParamValue` (Task 3).
- Produces: `RichReporter` implements every event in the protocol plus `close(status: str | None = None) -> None` (idempotent; `status="interrupted"` prints `interrupted` once). `karak.cli.reporter.short(value, width: int = 60) -> str` truncates with `…` (reused by Task 8).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_reporter.py`:

```python
"""Line-based output of `karak run --plain`."""

from __future__ import annotations

from rich.console import Console

from karak.cli.reporter import RichReporter, short
from karak.flow.events import ParamValue, RunInfo

INFO = RunInfo(
    flow="stepwise", input_path="/data/NWA", out_base="out/nwa",
    work_dir="out/work", cache=True, workers=None, device="cpu",
    version="0.2.0", nodes=(("src", "load_elements"),),
)


def _reporter():
    console = Console(record=True, width=120, color_system=None)
    return RichReporter(console), console


def test_plain_run_prints_context_params_outputs_and_summary():
    reporter, console = _reporter()
    reporter.run_started(INFO)
    reporter.node_params("src", [
        ParamValue("header_trim_px", 100, False),
        ParamValue("colormap", "tima:jet", True),
    ])
    reporter.node_cache("src", "abcdef1234567890", False, "out/work/cache")
    reporter.node_started("src", "Load elements")
    reporter.progress("src", 3, 20, "Si")
    reporter.node_outputs("src", {"cube": "ElementCube 2×2×1 float32 16 B space=raw"})
    reporter.node_finished("src", 1.5, False)
    reporter.run_finished({"src": {"cached": False, "seconds": 1.5}}, 1.6)
    text = console.export_text()
    for expected in [
        "karak run", "flow=stepwise", "karak 0.2.0", "/data/NWA", "out/nwa",
        "workers serial", "device cpu", "cache on",
        "src.header_trim_px = 100", "src: 1 params at defaults",
        "src: 3/20 - Si", "src.cube -> ElementCube 2×2×1",
        "src: done in 1.5s", "run finished in 1.6s (1 ran, 0 cached)",
    ]:
        assert expected in text
    assert "colormap" not in text


def test_plain_cached_step_shows_hash():
    reporter, console = _reporter()
    reporter.run_started(INFO)
    reporter.node_cache("src", "abcdef1234567890", True, "out/work/cache")
    reporter.node_finished("src", 0.1, True)
    assert "src: cached (abcdef12)" in console.export_text()


def test_plain_failure_escapes_markup():
    reporter, console = _reporter()
    reporter.node_failed("src", "bad value [red]x[/red]")
    assert "src: failed: bad value [red]x[/red]" in console.export_text()


def test_long_param_values_are_truncated():
    assert short("x" * 100, width=10) == "xxxxxxxxx…"
    assert short("abc", width=10) == "abc"
    reporter, console = _reporter()
    reporter.node_params("exp", [ParamValue("flow_json", "{" + "a" * 500 + "}", False)])
    line = [l for l in console.export_text().splitlines() if "flow_json" in l][0]
    assert len(line) < 100


def test_close_is_idempotent():
    reporter, console = _reporter()
    reporter.close("interrupted")
    reporter.close("interrupted")
    reporter.close()
    assert console.export_text().count("interrupted") == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_reporter.py -v`
Expected: FAIL with `ImportError: cannot import name 'short'`.

- [ ] **Step 3: Implement**

In `src/karak/cli/reporter.py`, add `from rich.markup import escape` to the imports, add this function above `class RichReporter`:

```python
def short(value, width: int = 60) -> str:
    """``str(value)`` cut to ``width`` characters with a trailing ellipsis."""
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"
```

and replace `class RichReporter` (keep `render_bench_table` below it unchanged) with:

```python
class RichReporter:
    """Line-based output: `karak run --plain`, or any non-terminal stdout."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._hashes: dict[str, str] = {}
        self._closed = False

    def run_started(self, info) -> None:
        workers = "serial" if info.workers is None else str(info.workers)
        self.console.print(
            f"[bold]karak run[/bold]  flow={escape(info.flow)}  karak {info.version}"
        )
        self.console.print(f"  input   {escape(info.input_path or '-')}")
        self.console.print(
            f"  out     {escape(info.out_base)}   work {escape(info.work_dir)}"
        )
        self.console.print(
            f"  workers {workers}   device {info.device}   "
            f"cache {'on' if info.cache else 'off'}"
        )

    def node_params(self, node_id: str, params) -> None:
        changed = [p for p in params if not p.is_default]
        for p in changed:
            self.console.print(
                f"  {node_id}.{p.name} = {escape(short(p.value))}"
            )
        n_default = len(params) - len(changed)
        if n_default:
            self.console.print(
                f"  {node_id}: {n_default} params at defaults", style="dim"
            )

    def node_cache(self, node_id: str, recipe_hash: str, cached: bool,
                   cache_dir: str) -> None:
        self._hashes[node_id] = recipe_hash

    def node_started(self, node_id: str, label: str) -> None:
        self.console.print(f"[cyan]>[/cyan] {node_id}: {escape(label)}")

    def progress(self, node_id: str, done: int, total: int, msg: str = "") -> None:
        detail = f" - {msg}" if msg else ""
        self.console.print(f"  {node_id}: {done}/{total}{escape(detail)}")

    def log(self, level: str, msg: str) -> None:
        style = {"warning": "yellow", "error": "red"}.get(level, "dim")
        self.console.print(escape(msg), style=style)

    def node_outputs(self, node_id: str, summaries: dict) -> None:
        for port, text in summaries.items():
            self.console.print(f"  {node_id}.{port} -> {escape(text)}")

    def node_finished(self, node_id: str, seconds: float, cached: bool) -> None:
        if cached:
            short_hash = self._hashes.get(node_id, "")[:8]
            self.console.print(f"[green]#[/green] {node_id}: cached ({short_hash})")
        else:
            self.console.print(
                f"[green]#[/green] {node_id}: done in {seconds:.1f}s"
            )

    def node_failed(self, node_id: str, message: str) -> None:
        self.console.print(
            f"[red]x {node_id}: failed: {escape(message)}[/red]"
        )

    def run_finished(self, summary: dict, seconds: float) -> None:
        ran = [n for n, e in summary.items() if not e.get("skipped")]
        cached = sum(1 for n in ran if summary[n].get("cached"))
        self.console.print(
            f"run finished in {seconds:.1f}s "
            f"({len(ran) - cached} ran, {cached} cached)"
        )

    def close(self, status: str | None = None) -> None:
        if self._closed:
            return
        if status == "interrupted":
            self.console.print("interrupted", style="yellow")
        self._closed = status is not None or self._closed
```

Note the last line: a plain `close()` does not mark the reporter closed, so a later `close("interrupted")` still prints; once a status is printed, further calls are no-ops.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cli_reporter.py tests/test_cli.py tests/test_bench.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/cli/reporter.py tests/test_cli_reporter.py
git commit -m "feat: plain run output shows context, params, outputs and cache

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 8: The live dashboard

**Files:**
- Create: `src/karak/cli/dashboard.py`
- Test: `tests/test_cli_dashboard.py` (create)

**Interfaces:**
- Consumes: `RunInfo`, `ParamValue` (Task 3); `MemorySampler` (Task 6); `short` (Task 7).
- Produces: `DashboardReporter(console: Console | None = None, *, live: bool = True, sampler=None)` implementing every protocol event plus `close(status: str | None = None)` (idempotent) and `render() -> rich Panel`. `live=False` never starts `rich.live.Live` (tests call `render()` directly). The sampler needs only `start()`, `stop()`, `current`, `peak`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_dashboard.py`:

```python
"""The live `karak run` dashboard, rendered from recorded events."""

from __future__ import annotations

import io

from rich.console import Console

from karak.cli.dashboard import DashboardReporter
from karak.flow.events import ParamValue, RunInfo


class FakeSampler:
    current = 2_400_000_000
    peak = 4_100_000_000

    def start(self):
        pass

    def stop(self):
        pass


def _info(workers=None):
    return RunInfo(
        flow="stepwise", input_path="/data/NWA_4587_data", out_base="out/nwa",
        work_dir="out/work", cache=True, workers=workers, device="cpu",
        version="0.2.0", nodes=(("src", "load_elements"),),
    )


def _dashboard():
    return DashboardReporter(Console(width=120, color_system=None),
                             live=False, sampler=FakeSampler())


def _text(reporter):
    console = Console(record=True, width=120, color_system=None)
    console.print(reporter.render())
    return console.export_text()


def _params():
    return (
        [ParamValue("colormap", "tima:jet", True),
         ParamValue("header_trim_px", 100, False)]
        + [ParamValue(f"p{i}", i, True) for i in range(10)]
    )


def test_running_view_shows_context_params_progress_and_log():
    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", _params())
    d.node_cache("src", "abcdef1234567890", False, "out/work/cache")
    d.node_started("src", "Load elements")
    d.progress("src", 14, 20, "Si")
    d.log("info", "Loaded element 'Si'")
    text = _text(d)
    for expected in [
        "stepwise", "karak 0.2.0", "/data/NWA_4587_data", "out/nwa",
        "device cpu", "cache on", "mem 2.4 GB (peak 4.1 GB)",
        "load_elements", "14/20", "header_trim_px", "Si",
        "Loaded element 'Si'", "(4 more at defaults)",
    ]:
        assert expected in text, expected
    assert text.index("header_trim_px") < text.index("colormap")


def test_memory_label_with_workers():
    d = _dashboard()
    d.run_started(_info(workers=16))
    assert "mem (main process)" in _text(d)


def test_finished_view_shows_outputs_and_summary():
    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", _params())
    d.node_cache("src", "abcdef1234567890", False, "out/work/cache")
    d.node_started("src", "Load elements")
    d.node_outputs("src", {"cube": "ElementCube 6525×3990×19 float32 1.98 GB space=raw"})
    d.node_finished("src", 58.2, False)
    d.run_finished({"src": {"cached": False, "seconds": 58.2}}, 58.4)
    text = _text(d)
    for expected in [
        "src.cube", "ElementCube 6525×3990×19", "58.2s",
        "finished in 58.4s", "1 ran, 0 cached", "peak memory 4.1 GB",
        "out/work/cache",
    ]:
        assert expected in text, expected
    assert "params:" not in text


def test_cached_step_shows_short_hash():
    d = _dashboard()
    d.run_started(_info())
    d.node_cache("src", "abcdef1234567890", True, "out/work/cache")
    d.node_outputs("src", {"cube": "ElementCube 2×2×1 float32 16 B space=raw"})
    d.node_finished("src", 0.1, True)
    d.run_finished({"src": {"cached": True, "seconds": 0.1}}, 0.2)
    assert "cached abcdef12" in _text(d)


def test_failure_view():
    d = _dashboard()
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.node_failed("src", "No files matching '*.png' in /nowhere")
    text = _text(d)
    assert "failed" in text
    assert "No files matching '*.png' in /nowhere" in text


def test_interrupt_marks_running_step():
    d = _dashboard()
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.close("interrupted")
    assert "interrupted" in _text(d)


def test_long_param_values_are_truncated():
    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", [ParamValue("flow", "{" + "a" * 500 + "}", False)])
    d.node_started("src", "Load elements")
    assert "a" * 100 not in _text(d)


def test_close_is_idempotent():
    d = _dashboard()
    d.run_started(_info())
    d.close("interrupted")
    d.close()
    d.close("interrupted")
    assert d.status == "interrupted"


def test_live_mode_leaves_final_summary_on_screen():
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=True, color_system=None)
    d = DashboardReporter(console, sampler=FakeSampler())
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.node_outputs("src", {"cube": "ElementCube 2×2×1 float32 16 B space=raw"})
    d.node_finished("src", 0.5, False)
    d.run_finished({"src": {"cached": False, "seconds": 0.5}}, 0.6)
    assert "finished in 0.6s" in buf.getvalue()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_dashboard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'karak.cli.dashboard'`.

- [ ] **Step 3: Implement `cli/dashboard.py`**

```python
"""Live Rich dashboard for `karak run` (the default on a terminal).

The executor and stages send events through the Reporter protocol; this
class keeps a small model of the run and renders it with Rich ``Live``.
When the run ends the last frame stays on screen as the run summary.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from karak.cli.memory import MemorySampler
from karak.cli.reporter import short

LOG_LINES = 5
PARAM_ROWS = 8


@dataclass
class _Step:
    node_id: str
    stage: str
    status: str = "pending"  # pending|running|done|cached|failed|interrupted
    started: float | None = None
    seconds: float | None = None
    recipe_hash: str = ""
    progress: tuple | None = None  # (done, total, msg)
    params: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    error: str = ""


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _gb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.1f} GB"


class DashboardReporter:
    def __init__(self, console: Console | None = None, *, live: bool = True,
                 sampler=None):
        self.console = console or Console()
        self.sampler = sampler or MemorySampler()
        self._use_live = live
        self._live: Live | None = None
        self._stopped = False
        self.info = None
        self.steps: dict[str, _Step] = {}
        self.logs: deque = deque(maxlen=LOG_LINES)
        self.current: str | None = None
        self.status = "running"  # running|done|failed|interrupted
        self.total_seconds: float | None = None

    # -- reporter protocol --------------------------------------------------

    def run_started(self, info) -> None:
        self.info = info
        self.steps = {nid: _Step(nid, stage) for nid, stage in info.nodes}
        self.sampler.start()
        if self._use_live:
            self._live = Live(
                console=self.console, get_renderable=self.render,
                refresh_per_second=4, transient=False,
            )
            self._live.start()

    def node_params(self, node_id: str, params) -> None:
        self._step(node_id).params = list(params)
        self.current = node_id

    def node_cache(self, node_id: str, recipe_hash: str, cached: bool,
                   cache_dir: str) -> None:
        step = self._step(node_id)
        step.recipe_hash = recipe_hash
        if cached:
            step.status = "cached"

    def node_started(self, node_id: str, label: str) -> None:
        step = self._step(node_id)
        step.status = "running"
        step.started = time.monotonic()
        self.current = node_id

    def progress(self, node_id: str, done: int, total: int, msg: str = "") -> None:
        self._step(node_id).progress = (done, total, msg)

    def log(self, level: str, msg: str) -> None:
        self.logs.append((level, msg))

    def node_outputs(self, node_id: str, summaries: dict) -> None:
        self._step(node_id).outputs = dict(summaries)

    def node_finished(self, node_id: str, seconds: float, cached: bool) -> None:
        step = self._step(node_id)
        step.seconds = seconds
        step.status = "cached" if cached else "done"
        step.progress = None
        if self.current == node_id:
            self.current = None

    def node_failed(self, node_id: str, message: str) -> None:
        step = self._step(node_id)
        step.status = "failed"
        step.error = message
        self.status = "failed"
        self._stop()

    def run_finished(self, summary: dict, seconds: float) -> None:
        self.total_seconds = seconds
        self.status = "done"
        self._stop()

    def close(self, status: str | None = None) -> None:
        if status == "interrupted" and self.status == "running":
            self.status = "interrupted"
            for step in self.steps.values():
                if step.status == "running":
                    step.status = "interrupted"
        self._stop()

    # -- internals ----------------------------------------------------------

    def _step(self, node_id: str) -> _Step:
        return self.steps.setdefault(node_id, _Step(node_id, "?"))

    def _stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.sampler.stop()
        if self._live is not None:
            self._live.update(self.render(), refresh=True)
            self._live.stop()
            self._live = None

    # -- rendering ----------------------------------------------------------

    def render(self) -> Panel:
        parts = [self._header(), self._steps_table()]
        outputs = self._outputs()
        if outputs is not None:
            parts.append(outputs)
        if self.status == "running":
            step = self.steps.get(self.current) if self.current else None
            if step is not None and step.params:
                parts.append(self._params_panel(step))
            if step is not None and step.progress:
                parts.append(self._progress_line(step))
            if self.logs:
                parts.append(self._log_panel())
        else:
            parts.append(self._footer())
        title = "karak run"
        if self.info is not None:
            title += f"  {self.info.flow}  karak {self.info.version}"
        return Panel(Group(*parts), title=title, title_align="left")

    def _memory_text(self) -> str:
        info = self.info
        label = "mem" if info is None or info.workers in (None, 1) else "mem (main process)"
        if self.sampler.current is None:
            return f"{label} peak {_gb(self.sampler.peak)}"
        return f"{label} {_gb(self.sampler.current)} (peak {_gb(self.sampler.peak)})"

    def _header(self) -> Table:
        grid = Table.grid(padding=(0, 3))
        grid.add_column()
        grid.add_column()
        if self.info is not None:
            info = self.info
            workers = "serial" if info.workers is None else str(info.workers)
            grid.add_row(f"input {short(info.input_path or '-', 70)}",
                         f"out {short(info.out_base, 40)}")
            grid.add_row(
                f"workers {workers}   device {info.device}   "
                f"cache {'on' if info.cache else 'off'}",
                self._memory_text(),
            )
        return grid

    def _status_text(self, step: _Step) -> Text:
        if step.status == "running":
            label = "▶ running"
            if step.progress:
                label = f"▶ {step.progress[0]}/{step.progress[1]}"
            return Text(label, style="cyan")
        if step.status == "cached":
            return Text(f"cached {step.recipe_hash[:8]}", style="green")
        style = {"done": "green", "failed": "bold red",
                 "interrupted": "yellow", "pending": "dim"}.get(step.status, "")
        return Text(step.status, style=style)

    def _time_text(self, step: _Step) -> str:
        if step.status == "running" and step.started is not None:
            return _clock(time.monotonic() - step.started)
        if step.seconds is not None:
            return f"{step.seconds:.1f}s"
        return ""

    def _steps_table(self) -> Table:
        table = Table(box=None, pad_edge=False, header_style="bold")
        for name in ("step", "stage", "status"):
            table.add_column(name)
        table.add_column("time", justify="right")
        for step in self.steps.values():
            table.add_row(step.node_id, step.stage, self._status_text(step),
                          self._time_text(step))
        return table

    def _outputs(self) -> Table | None:
        rows = [(f"{s.node_id}.{port}", text)
                for s in self.steps.values() for port, text in s.outputs.items()]
        if not rows:
            return None
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        for name, text in rows:
            grid.add_row(name, f"→ {text}")
        return grid

    def _params_panel(self, step: _Step) -> Panel:
        changed = [p for p in step.params if not p.is_default]
        defaults = [p for p in step.params if p.is_default]
        shown_defaults = defaults[: max(PARAM_ROWS - len(changed), 0)]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        for p in changed:
            grid.add_row(p.name, Text(short(p.value)))
        for p in shown_defaults:
            grid.add_row(p.name, Text(short(p.value), style="dim"))
        hidden = len(defaults) - len(shown_defaults)
        if hidden:
            grid.add_row("", Text(f"({hidden} more at defaults)", style="dim"))
        return Panel(grid, title=f"params: {step.node_id}", title_align="left")

    def _progress_line(self, step: _Step) -> Table:
        done, total, msg = step.progress
        grid = Table.grid(padding=(0, 2))
        grid.add_row(
            ProgressBar(total=max(total, 1), completed=done, width=50),
            f"{done}/{total}",
            short(msg, 30),
        )
        return grid

    def _log_panel(self) -> Panel:
        styles = {"warning": "yellow", "error": "red"}
        lines = [
            Text(msg, style=styles.get(level, "dim"), no_wrap=True,
                 overflow="ellipsis")
            for level, msg in self.logs
        ]
        return Panel(Group(*lines), title="log", title_align="left")

    def _footer(self) -> Group:
        lines = []
        if self.status == "done":
            ran = [s for s in self.steps.values() if s.status in ("done", "cached")]
            cached = sum(1 for s in ran if s.status == "cached")
            lines.append(Text(
                f"finished in {self.total_seconds:.1f}s · "
                f"{len(ran) - cached} ran, {cached} cached · "
                f"peak memory {_gb(self.sampler.peak)}",
                style="green",
            ))
        elif self.status == "failed":
            for step in self.steps.values():
                if step.status == "failed":
                    lines.append(Text(f"failed at {step.node_id}: {step.error}",
                                      style="bold red"))
        elif self.status == "interrupted":
            lines.append(Text("interrupted", style="yellow"))
        if self.info is not None:
            lines.append(Text(f"cache {self.info.work_dir}/cache", style="dim"))
        return Group(*lines)
```


- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_cli_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/karak/cli/dashboard.py tests/test_cli_dashboard.py
git commit -m "feat: live Rich dashboard reporter for karak run

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 9: Wire the dashboard into `karak run`

**Files:**
- Modify: `src/karak/flow/__main__.py`, `src/karak/cli/main.py`
- Test: `tests/test_cli_run.py` (create)

**Interfaces:**
- Consumes: `DashboardReporter` (Task 8), `RichReporter.close` (Task 7), `capture_logs` (Task 6), `stepwise` (Task 5).
- Produces: `karak.cli.main._run_reporter(plain: bool, isatty: bool)`; `karak run ... --plain`; exit codes 0/1/130. `python -m karak.flow run` without a reporter keeps printing its `N nodes: ...` line; with a reporter it does not.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_run.py`:

```python
"""`karak run`: reporter choice, log capture, exit codes, real stepwise run."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.cli.dashboard import DashboardReporter
from karak.cli.main import _run_reporter, main
from karak.cli.reporter import RichReporter
from karak.stages import registry
from karak.stages.base import Port, Stage


def test_reporter_is_plain_when_stdout_is_not_a_terminal():
    assert isinstance(_run_reporter(plain=False, isatty=False), RichReporter)


def test_reporter_is_plain_with_flag():
    assert isinstance(_run_reporter(plain=True, isatty=True), RichReporter)


def test_reporter_is_dashboard_on_a_terminal():
    assert isinstance(_run_reporter(plain=False, isatty=True), DashboardReporter)


class FailingStage(Stage):
    id = "cli_test_fail"
    label = "Always fails"
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        raise ValueError("boom from stage")


@pytest.fixture
def failing_stage():
    registry.register(FailingStage)
    yield
    registry._REGISTRY.pop("cli_test_fail", None)


def _karak_logger_state():
    logger = logging.getLogger("karak")
    return list(logger.handlers), logger.level, logger.propagate


def test_run_failure_exits_1_and_restores_logging(tmp_path, capsys, failing_stage):
    flow = tmp_path / "flow.json"
    flow.write_text(json.dumps({
        "version": 1, "name": "fail",
        "nodes": [{"id": "bad", "type": "cli_test_fail", "params": {}}],
        "edges": [],
    }))
    before = _karak_logger_state()
    rc = main(["run", str(flow), "--out", str(tmp_path / "o"), "--plain"])
    assert rc == 1
    assert "boom from stage" in capsys.readouterr().err
    assert _karak_logger_state() == before


def test_run_validation_failure_exits_1(tmp_path, capsys):
    flow = tmp_path / "flow.json"
    flow.write_text(json.dumps({
        "version": 1, "name": "broken",
        "nodes": [{"id": "x", "type": "no_such_stage", "params": {}}],
        "edges": [],
    }))
    before = _karak_logger_state()
    rc = main(["run", str(flow), "--out", str(tmp_path / "o"), "--plain"])
    assert rc == 1
    assert "failed validation" in capsys.readouterr().err
    assert _karak_logger_state() == before


def test_keyboard_interrupt_exits_130(tmp_path, monkeypatch, capsys):
    import karak.flow.executor as executor

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(executor, "run", interrupted)
    rc = main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"),
               "--plain"])
    assert rc == 130
    assert "interrupted" in capsys.readouterr().out


@pytest.fixture
def scene(tmp_path):
    import imageio.v2 as imageio
    import matplotlib

    import karak.io.loaders as loaders

    orig_dir, orig_mem = loaders._CACHE_DIR, loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}
    palette = (
        np.array(
            [matplotlib.colormaps["jet"](s)[:3] for s in np.linspace(0, 1, 64)]
        ) * 255
    ).astype(np.uint8)
    lut_path = tmp_path / "palette.npy"
    np.save(lut_path, palette)
    cube = make_synthetic_scene()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, element in enumerate(("A", "B", "C")):
        indices = np.round(cube[:, :, i] * 63).astype(np.uint8)
        imageio.imwrite(data_dir / f"s-01-{element}.png", palette[indices])
    imageio.imwrite(data_dir / "s-01-SEM.png", np.zeros((64, 64), dtype=np.uint8))
    yield data_dir, f"lut:{lut_path}"
    loaders._CACHE_DIR, loaders._FULL_LUT_CACHE = orig_dir, orig_mem


def test_stepwise_plain_run_end_to_end(tmp_path, capsys, scene):
    data_dir, colormap = scene
    argv = ["run", "--builtin", "stepwise", "--input", str(data_dir),
            "--out", str(tmp_path / "out" / "run"), "--plain",
            "--set", f"src.colormap={colormap}"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    for expected in [
        "flow=stepwise", "src.colormap = lut:", "src: 4/4",
        "src.cube -> ElementCube 32×32×3 float32", "src.bse -> BseImage",
        "run finished in", "(1 ran, 0 cached)", "Found 4 matching files",
    ]:
        assert expected in out, expected
    assert "nodes:" not in out

    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "src: cached (" in out
    assert "src.cube -> ElementCube 32×32×3" in out
    assert "(0 ran, 1 cached)" in out
```

`make_synthetic_scene()` returns a 64×64 scene and the stage default `downsample_factor` is 2, so the cube is 32×32×3.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_cli_run.py -v`
Expected: FAIL with `ImportError: cannot import name '_run_reporter'`.

- [ ] **Step 3: Add `--plain` and quiet the summary line in `flow/__main__.py`**

In `build_parser()`, after the `--device` argument of `run_parser`, add:

```python
    run_parser.add_argument(
        "--plain", action="store_true",
        help="Line-based output instead of the live dashboard "
             "(automatic when stdout is not a terminal).",
    )
```

At the end of `main()`, replace

```python
    cached = sum(1 for entry in summary.values() if entry.get("cached"))
    print(f"{len(summary)} nodes: {cached} cached, "
          f"{len(summary) - cached} executed")
    return 0
```

with

```python
    if reporter is None:
        cached = sum(1 for entry in summary.values() if entry.get("cached"))
        print(f"{len(summary)} nodes: {cached} cached, "
              f"{len(summary) - cached} executed")
    return 0
```

- [ ] **Step 4: Wire up `cli/main.py`**

Add these functions above `def main(`:

```python
def _run_reporter(plain: bool, isatty: bool):
    """The live dashboard on a terminal; line output with --plain or a pipe."""
    if plain or not isatty:
        from karak.cli.reporter import RichReporter

        return RichReporter()
    from karak.cli.dashboard import DashboardReporter

    return DashboardReporter()


def _run_command(argv: list[str]) -> int:
    from karak.cli.logs import capture_logs
    from karak.flow.__main__ import main as flow_main
    from karak.flow.executor import FlowError

    reporter = _run_reporter("--plain" in argv, sys.stdout.isatty())
    try:
        with capture_logs(reporter):
            return flow_main(argv, reporter=reporter)
    except FlowError as exc:
        reporter.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        reporter.close("interrupted")
        return 130
    finally:
        reporter.close()
```

and in `main()`, replace

```python
    if argv and argv[0] in _FLOW_COMMANDS:
        from karak.cli.reporter import RichReporter
        from karak.flow.__main__ import main as flow_main

        reporter = RichReporter() if argv[0] == "run" else None
        return flow_main(argv, reporter=reporter)
```

with

```python
    if argv and argv[0] == "run":
        return _run_command(argv)
    if argv and argv[0] in _FLOW_COMMANDS:
        from karak.flow.__main__ import main as flow_main

        return flow_main(argv)
```

`test_keyboard_interrupt_exits_130` patches `karak.flow.executor.run`; `flow/__main__.py` imports it inside `main()` (`from karak.flow.executor import run as run_flow`), so the patch takes effect. If it was moved to module level, keep the import inside the function.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_cli_run.py tests/test_cli.py tests/test_flow_cli.py -v`
Expected: all PASS.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -p no:warnings`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/karak/flow/__main__.py src/karak/cli/main.py tests/test_cli_run.py
git commit -m "feat: karak run shows the live dashboard, --plain for line output

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 10: Docs and the real-data run

**Files:**
- Modify: `docs/user_guide.md` (section `## CLI reference`, and `## Flows`), `CHANGELOG.md` (`## [Unreleased]`), `CLAUDE.md` (`## Commands`)

- [ ] **Step 1: Update the docs**

In `CLAUDE.md` `## Commands`, after the line `karak run ... --no-qc ...`, add:

```
karak run --builtin stepwise --input DIR --out BASE  # the growing step-by-step flow (load only, for now)
karak run ... --plain                      # line output instead of the live dashboard
```

In `CHANGELOG.md` under `## [Unreleased]` → `### Added`, add:

```markdown
- `karak run` shows a live dashboard on a terminal: run context, the
  running step's parameters (changed ones first), per-file progress for
  the load step, output summaries, cache status with recipe hashes,
  memory use, and the latest log lines. The final frame stays on screen
  as the run summary. `--plain` (automatic when stdout is not a terminal)
  prints the same information as lines.
- `stepwise` builtin flow, which grows one step at a time; it currently
  runs the load step only.
- Payloads have `summary()`; the cache stores it next to each payload.
- Exit codes for `karak run`: 1 when a step fails, 130 on Ctrl-C.
```

In `docs/user_guide.md` `## CLI reference`, add a paragraph and an example after the `karak run` usage:

```markdown
On a terminal, `karak run` shows a live dashboard: the run context
(input, output, workers, device, cache, memory), a table of steps with
their status and time, the running step's parameters with changed values
first, a progress bar where the stage reports progress (the load step
reports one tick per file), output summaries such as
`ElementCube 6525×3990×19 float32 1.98 GB space=raw`, and the last five
log lines. A cached step shows the first 8 characters of its recipe hash.
When the run ends, the last frame stays on screen as the summary.

With `--plain`, or when stdout is not a terminal (a pipe, `nohup`, a CI
log), the same information prints as plain lines.

`karak run` exits with 1 when a step fails and with 130 on Ctrl-C;
completed steps stay cached.
```

In `## Flows`, add `stepwise` to the list of built-in flows with the sentence: "`stepwise` grows one step at a time as steps join the dashboard work; today it runs only the load step (`src`)."

- [ ] **Step 2: Run the full suite and the stage-reference drift test**

Run: `uv run pytest -p no:warnings`
Expected: all PASS.

- [ ] **Step 3: Real-data run in a terminal**

Run in a pseudo-terminal so the dashboard is active, writing to the session scratchpad (not the repo):

```bash
OUT=/tmp/claude-1000/-home-brendon-Projects-science-karak/72767705-9b7b-4ce5-bba8-e682b902629c/scratchpad/stepwise/run
script -q -c "uv run karak run --builtin stepwise --input /home/brendon/Dropbox/Projects/izawa/NWA_4587_data --out $OUT" /dev/null | tail -40
```

Expected: exit 0; the final frame shows `src load_elements done <≈60s>`, `src.cube → ElementCube 6525×3990×19 float32 1.98 GB space=raw`, `src.bse → BseImage 6525×3990 float32 104 MB`, `finished in … · 1 ran, 0 cached · peak memory …`, and the cache directory.

Run it a second time: the step shows `cached <hash>` with the same output summaries, `0 ran, 1 cached`.

Run once with `--plain` and confirm the line output: header, `src: 1/20 - Al` … `src: 20/20 - Zn`, output lines, `run finished`.

- [ ] **Step 4: Show the user the screen output**

Paste the final dashboard frame of the first run and the cached rerun into the conversation for review before calling the work done.

- [ ] **Step 5: Commit**

```bash
git add docs/user_guide.md CHANGELOG.md CLAUDE.md
git commit -m "docs: run dashboard, --plain, and the stepwise flow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```
