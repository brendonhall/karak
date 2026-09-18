# CPU/GPU Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opt-in CPU multicore (`workers`) and CUDA (`device`) execution for karak stages, plus a `karak bench` subcommand that measures per-node wall time across configurations.

**Architecture:** `workers` is a runtime executor setting injected onto stages like `reporter` (never hashed, results byte-identical). `device` is a declared `Param` on GPU-capable stages (hashed, CPU/GPU outputs cache separately). All GPU imports are lazy behind a new `karak/accel.py`; core functions stay numpy-in/numpy-out.

**Tech Stack:** Python 3.12, `concurrent.futures.ProcessPoolExecutor`, optional `cupy-cuda12x` + `cuml-cu12` + `cucim-cu12` extra, pytest, Rich (CLI only).

**Spec:** `docs/superpowers/specs/2026-09-18-acceleration-design.md`

## Global Constraints

- Defaults reproduce current behavior exactly: `workers=None` (serial), `device="cpu"`.
- `workers` must never change numeric results; parity tests assert byte-identical output.
- `device` enters the recipe hash (it is an ordinary `Param`); `workers` never does.
- GPU imports only inside functions; a plain install never imports cupy/cuml/cucim.
- GPU tests are marked `@pytest.mark.skipif(not cuda_available(), reason="no CUDA")`; this machine has no GPU, so they must skip cleanly.
- Rich stays in `cli/`; `flow/` may import `cli` only lazily inside a subcommand branch.
- Run tests with `uv run pytest`. Commit after each task and push to `main` (repo rule).
- Exact-tier GPU parity tolerance: `np.allclose(cpu, gpu, atol=1e-5)`.
- No em dashes in prose or help strings.

---

### Task 1: `karak/accel.py`

**Files:**
- Create: `src/karak/accel.py`
- Test: `tests/test_accel.py`

**Interfaces:**
- Produces: `resolve_workers(workers: int | None) -> int`, `cuda_available() -> bool`, `gpu_name() -> str | None`, `get_array_module(device: str)`, `to_device(arr, device: str)`, `to_numpy(arr) -> np.ndarray`. All later tasks import these from `karak.accel`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_accel.py
"""Tests for karak.accel: worker resolution and CUDA access helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from karak import accel
from karak.stages.base import StageError


def test_resolve_workers_none_is_serial():
    assert accel.resolve_workers(None) == 1


def test_resolve_workers_zero_is_all_cores():
    assert accel.resolve_workers(0) == (os.cpu_count() or 1)


def test_resolve_workers_positive_passthrough():
    assert accel.resolve_workers(3) == 3


def test_resolve_workers_negative_raises():
    with pytest.raises(ValueError):
        accel.resolve_workers(-1)


def test_get_array_module_cpu_is_numpy():
    assert accel.get_array_module("cpu") is np


def test_get_array_module_unknown_device_raises():
    with pytest.raises(StageError):
        accel.get_array_module("tpu")


def test_get_array_module_cuda_without_gpu_raises(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    with pytest.raises(StageError, match="karak\\[cuda\\]"):
        accel.get_array_module("cuda")


def test_to_device_cpu_is_identity():
    arr = np.arange(4)
    assert accel.to_device(arr, "cpu") is arr


def test_to_numpy_passthrough():
    arr = np.arange(4)
    assert accel.to_numpy(arr) is arr


def test_cuda_available_is_bool():
    assert isinstance(accel.cuda_available(), bool)


def test_gpu_name_none_without_gpu(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    assert accel.gpu_name() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_accel.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'karak.accel'`

- [ ] **Step 3: Write the implementation**

```python
# src/karak/accel.py
"""Optional acceleration helpers: CPU worker resolution and CUDA access.

The whole GPU surface of karak lives here. CuPy is imported lazily so a
plain install never touches it.
"""

from __future__ import annotations

import os

import numpy as np

from karak.stages.base import StageError

_INSTALL_HINT = (
    "device='cuda' requested but no usable GPU stack was found. "
    "Install the extra with: pip install 'karak[cuda]' "
    "(needs an NVIDIA driver and CUDA 12)."
)


def resolve_workers(workers: int | None) -> int:
    """None -> 1 (serial), 0 -> all cores, N -> N."""
    if workers is None:
        return 1
    if workers < 0:
        raise ValueError(f"workers must be >= 0, got {workers}")
    if workers == 0:
        return os.cpu_count() or 1
    return workers


def cuda_available() -> bool:
    """True when CuPy imports and sees at least one device."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_name() -> str | None:
    """Name of GPU 0, or None without a usable CUDA stack."""
    if not cuda_available():
        return None
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def get_array_module(device: str):
    """numpy for 'cpu', cupy for 'cuda'; StageError otherwise."""
    if device == "cpu":
        return np
    if device == "cuda":
        if not cuda_available():
            raise StageError(_INSTALL_HINT)
        import cupy

        return cupy
    raise StageError(f"unknown device {device!r}; expected 'cpu' or 'cuda'")


def to_device(arr, device: str):
    """Move an array to the device. No-op on CPU."""
    if device == "cpu":
        return arr
    return get_array_module(device).asarray(arr)


def to_numpy(arr) -> np.ndarray:
    """Bring an array back to host memory. No-op for numpy arrays."""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "get"):  # cupy
        return arr.get()
    return np.asarray(arr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_accel.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/karak/accel.py tests/test_accel.py
git commit -m "feat: karak.accel with worker resolution and lazy CUDA access"
git push origin main
```

---

### Task 2: `workers` through the executor and CLI

**Files:**
- Modify: `src/karak/stages/base.py:105` (add class attribute next to `reporter`)
- Modify: `src/karak/flow/executor.py:120-130` (signature) and `:216-217` (injection)
- Modify: `src/karak/flow/__main__.py:59-75` (run parser) and `:109-117` (pass through)
- Test: `tests/test_executor.py` (extend), `tests/test_flow_cli.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `executor.run(..., workers: int | None = None)`; `Stage.workers: int | None = None` set on instances by the executor; CLI flag `--workers N` on `karak run` / `python -m karak.flow run`. Stages read `self.workers` and pass `resolve_workers(self.workers)` to core functions.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_executor.py` (it already registers `FakeSource`/`FakeAdd`/`FakeSink` via the autouse `_fake_stages` fixture; `RECORD` is a module-level list):

```python
class FakeWorkerProbe(Stage):
    id = "fake_worker_probe"
    label = "Fake worker probe"
    OUTPUTS = [Port("num")]
    PARAMS = [Param("value", "int", 1)]

    def apply(self, inputs, params):
        RECORD.append(("workers", self.workers))
        return {"num": ClusterStats(stats={"value": params["value"]})}


def _probe_graph():
    return Graph(
        name="probe",
        nodes=(Node("p", "fake_worker_probe"),),
        edges=(),
    )


def test_executor_injects_workers(tmp_path):
    registry.register(FakeWorkerProbe)
    try:
        run(_probe_graph(), work_dir=str(tmp_path), cache=False, workers=5)
        assert ("workers", 5) in RECORD
    finally:
        registry._REGISTRY.pop(FakeWorkerProbe.id, None)


def test_executor_workers_default_none(tmp_path):
    registry.register(FakeWorkerProbe)
    try:
        run(_probe_graph(), work_dir=str(tmp_path), cache=False)
        assert ("workers", None) in RECORD
    finally:
        registry._REGISTRY.pop(FakeWorkerProbe.id, None)
```

Append to `tests/test_flow_cli.py` (follow its existing style for invoking `karak.flow.__main__.main`; if it uses a different helper, adapt the invocation, not the assertion):

```python
def test_run_parser_accepts_workers():
    from karak.flow.__main__ import build_parser

    args = build_parser().parse_args(
        ["run", "--builtin", "global", "--workers", "4"]
    )
    assert args.workers == 4


def test_run_parser_workers_default_none():
    from karak.flow.__main__ import build_parser

    args = build_parser().parse_args(["run", "--builtin", "global"])
    assert args.workers is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_executor.py tests/test_flow_cli.py -v -k workers`
Expected: FAIL (`self.workers` attribute missing / unrecognized `--workers` / unexpected keyword `workers`)

- [ ] **Step 3: Implement**

In `src/karak/stages/base.py`, directly under `reporter: Any = None` (line 105):

```python
    workers: int | None = None      # injected by the flow executor; None = serial
```

In `src/karak/flow/executor.py`, add to the `run()` signature after `reporter=None`:

```python
    workers: int | None = None,
```

and in the non-cached branch, next to `stage.reporter = reporter`:

```python
            stage.workers = workers
```

In `src/karak/flow/__main__.py`, add to the run parser after `--no-qc`:

```python
    run_parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="CPU worker processes for stages that parallelize "
             "(0 = all cores; default: serial). Never changes results.",
    )
```

and pass `workers=args.workers` in the `run_flow(...)` call.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest`
Expected: all PASS (defaults unchanged, so nothing else moves)

- [ ] **Step 5: Commit**

```bash
git add src/karak/stages/base.py src/karak/flow/executor.py src/karak/flow/__main__.py tests/test_executor.py tests/test_flow_cli.py
git commit -m "feat: workers runtime setting through executor and CLI"
git push origin main
```

---

### Task 3: `apply_device`, `--device`, and the CUDA pre-flight

**Files:**
- Modify: `src/karak/flow/builtins.py` (add `apply_device` under `override_params`, line 182)
- Modify: `src/karak/flow/__main__.py` (run parser + run branch)
- Test: `tests/test_builtins.py` (extend), `tests/test_flow_cli.py` (extend)

**Interfaces:**
- Consumes: `override_params(graph, overrides)` (`flow/builtins.py:182`), `karak.accel.cuda_available`.
- Produces: `apply_device(graph: Graph, device: str) -> Graph` (sets `device` only on nodes whose stage declares that param); CLI flag `--device {cpu,cuda}`; pre-flight `SystemExit` when any node has `device == "cuda"` and `cuda_available()` is false.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_builtins.py`:

```python
def test_apply_device_sets_only_declaring_nodes():
    from karak.flow.builtins import apply_device, builtin_flow
    from karak.stages import registry
    from karak.stages.base import Param, Port, Stage
    from karak.flow.graph import Graph, Node

    class FakeCudaStage(Stage):
        id = "fake_cuda_stage"
        label = "Fake CUDA stage"
        OUTPUTS = [Port("x")]
        PARAMS = [Param("device", "str", "cpu", choices=("cpu", "cuda"))]

        def apply(self, inputs, params):
            return {}

    registry.register(FakeCudaStage)
    try:
        graph = Graph(
            name="g",
            nodes=(
                Node("a", "fake_cuda_stage"),
                Node("b", "load_elements"),
            ),
            edges=(),
        )
        out = apply_device(graph, "cuda")
        assert out.node("a").params["device"] == "cuda"
        assert "device" not in out.node("b").params
    finally:
        registry._REGISTRY.pop(FakeCudaStage.id, None)


def test_apply_device_no_declaring_nodes_is_identity():
    from karak.flow.builtins import apply_device, builtin_flow

    graph = builtin_flow("global")
    assert apply_device(graph, "cuda") is graph
```

Append to `tests/test_flow_cli.py`:

```python
def test_run_parser_accepts_device():
    from karak.flow.__main__ import build_parser

    args = build_parser().parse_args(
        ["run", "--builtin", "global", "--device", "cuda"]
    )
    assert args.device == "cuda"


def test_cuda_preflight_rejects_without_gpu(monkeypatch, tmp_path):
    import karak.accel as accel
    from karak.flow.__main__ import main as flow_main
    from karak.stages import registry
    from karak.stages.base import Param, Port, Stage

    class FakeCudaStage(Stage):
        id = "fake_cuda_stage"
        label = "Fake CUDA stage"
        OUTPUTS = [Port("x")]
        PARAMS = [Param("device", "str", "cpu", choices=("cpu", "cuda"))]

        def apply(self, inputs, params):
            return {}

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    registry.register(FakeCudaStage)
    try:
        flow_path = tmp_path / "flow.json"
        flow_path.write_text(
            '{"version": 1, "name": "g", '
            '"nodes": [{"id": "a", "type": "fake_cuda_stage", "params": {}}], '
            '"edges": []}'
        )
        with pytest.raises(SystemExit, match="cuda"):
            flow_main(["run", str(flow_path), "--device", "cuda",
                       "--out", str(tmp_path / "o")])
    finally:
        registry._REGISTRY.pop(FakeCudaStage.id, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_builtins.py tests/test_flow_cli.py -v -k device`
Expected: FAIL with `ImportError: cannot import name 'apply_device'` / unrecognized `--device`

- [ ] **Step 3: Implement**

In `src/karak/flow/builtins.py`, below `override_params`:

```python
def apply_device(graph: Graph, device: str) -> Graph:
    """Set ``device`` on every node whose stage declares that param.

    Nodes without a declared ``device`` param are left untouched, so the
    override is safe on any flow.
    """
    from karak.stages import registry

    overrides = {
        f"{node.id}.device": device
        for node in graph.nodes
        if any(p.name == "device" for p in registry.get(node.type).PARAMS)
    }
    return override_params(graph, overrides) if overrides else graph
```

In `src/karak/flow/__main__.py`, add to the run parser:

```python
    run_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Apply this device to every node that declares a device "
             "param (cuda needs the karak[cuda] extra).",
    )
```

In the run branch, after the `--set` overrides are applied:

```python
    from karak.flow.builtins import apply_device

    if args.device:
        graph = apply_device(graph, args.device)
    if any(n.params.get("device") == "cuda" for n in graph.nodes):
        import karak.accel as accel

        if not accel.cuda_available():
            raise SystemExit(
                "error: device='cuda' requested but no usable GPU stack "
                "was found. Install with: pip install 'karak[cuda]'"
            )
```

(`apply_device` joins the existing `override_params` import at the top of the file instead of a local import if that reads better; either is fine.)

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/karak/flow/builtins.py src/karak/flow/__main__.py tests/test_builtins.py tests/test_flow_cli.py
git commit -m "feat: --device flag with apply_device and CUDA pre-flight"
git push origin main
```

---

### Task 4: `karak bench`

**Files:**
- Create: `src/karak/cli/bench.py`
- Modify: `src/karak/cli/main.py:25` (`_FLOW_COMMANDS`) and `main()` routing
- Modify: `src/karak/cli/reporter.py` (add `render_bench_table`)
- Modify: `src/karak/flow/__main__.py` (mirror subcommand, lazy import)
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `executor.run(...) -> {node_id: {"cached": bool, "seconds": float}}`, `apply_device` (Task 3), `workers=` kwarg (Task 2), `QC_STAGE_TYPES` and `_load_graph`/`_parse_set` from `flow/__main__.py`.
- Produces: `parse_config(spec: str) -> BenchConfig` (`BenchConfig(label, workers, device)` frozen dataclass), `run_bench(graph, *, input_path, out_base, work_dir, configs, repeats, include_qc) -> dict` (the JSON-shaped result), `bench_main(argv) -> int`, `render_bench_table(result: dict) -> None` in `cli/reporter.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench.py
"""Tests for the karak bench subcommand."""

from __future__ import annotations

import json

import pytest

from karak.cli.bench import BenchConfig, parse_config, run_bench
from karak.flow.graph import Graph, Node
from karak.stages import registry
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import ClusterStats


class BenchProbe(Stage):
    id = "bench_probe"
    label = "Bench probe"
    OUTPUTS = [Port("num")]
    PARAMS = [Param("value", "int", 1)]

    def apply(self, inputs, params):
        return {"num": ClusterStats(stats={"value": params["value"]})}


@pytest.fixture(autouse=True)
def _probe_stage():
    registry.register(BenchProbe)
    yield
    registry._REGISTRY.pop(BenchProbe.id, None)


def _graph():
    return Graph(name="g", nodes=(Node("p", "bench_probe"),), edges=())


def test_parse_config_baseline():
    assert parse_config("baseline") == BenchConfig("baseline", None, None)


def test_parse_config_workers_and_device():
    assert parse_config("workers=0,device=cuda") == BenchConfig(
        "workers=0,device=cuda", 0, "cuda"
    )


def test_parse_config_unknown_key_exits():
    with pytest.raises(SystemExit):
        parse_config("threads=4")


def test_run_bench_result_shape(tmp_path):
    result = run_bench(
        _graph(),
        input_path="",
        out_base=str(tmp_path / "out"),
        work_dir=str(tmp_path / "work"),
        configs=[parse_config("baseline"), parse_config("workers=2")],
        repeats=2,
        include_qc=False,
    )
    assert {"meta", "configs"} <= set(result)
    assert result["meta"]["cpus"] >= 1
    assert len(result["configs"]) == 2
    for cfg in result["configs"]:
        assert cfg["nodes"]["p"] >= 0.0
        assert cfg["total"] >= 0.0


def test_bench_json_round_trips(tmp_path):
    result = run_bench(
        _graph(),
        input_path="",
        out_base=str(tmp_path / "out"),
        work_dir=str(tmp_path / "work"),
        configs=[parse_config("baseline")],
        repeats=1,
        include_qc=False,
    )
    text = json.dumps(result)
    assert json.loads(text) == result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bench.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'karak.cli.bench'`

- [ ] **Step 3: Implement `cli/bench.py`**

```python
# src/karak/cli/bench.py
"""karak bench: time flow nodes across worker/device configurations.

Runs the flow once per --config with the cache disabled, keeps the
per-node minimum over --repeats, and writes <out>_bench.json plus a
terminal table. --compare renders a table from two result files.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchConfig:
    label: str
    workers: int | None
    device: str | None


def parse_config(spec: str) -> BenchConfig:
    """'baseline' | comma list of workers=N and device=NAME."""
    if spec == "baseline":
        return BenchConfig("baseline", None, None)
    workers: int | None = None
    device: str | None = None
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise SystemExit(f"error: bench config needs key=value, got {part!r}")
        if key == "workers":
            workers = int(value)
        elif key == "device":
            device = value
        else:
            raise SystemExit(
                f"error: unknown bench config key {key!r} in {spec!r} "
                "(expected workers=N and/or device=NAME)"
            )
    return BenchConfig(spec, workers, device)


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _meta(input_path: str) -> dict:
    from karak import accel

    return {
        "hostname": socket.gethostname(),
        "cpus": os.cpu_count() or 1,
        "gpu": accel.gpu_name(),
        "git": _git_revision(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": input_path,
    }


def run_bench(
    graph,
    *,
    input_path: str,
    out_base: str,
    work_dir: str,
    configs: list[BenchConfig],
    repeats: int,
    include_qc: bool,
) -> dict:
    from karak.flow.__main__ import QC_STAGE_TYPES
    from karak.flow.builtins import apply_device
    from karak.flow.executor import run as run_flow

    skip = frozenset() if include_qc else QC_STAGE_TYPES
    results = []
    for cfg in configs:
        run_graph = apply_device(graph, cfg.device) if cfg.device else graph
        best: dict[str, float] = {}
        for _ in range(repeats):
            summary = run_flow(
                run_graph,
                input_path=input_path,
                out_base=out_base,
                work_dir=work_dir,
                cache=False,
                workers=cfg.workers,
                skip_types=skip,
            )
            for node_id, entry in summary.items():
                if entry.get("skipped"):
                    continue
                seconds = entry["seconds"]
                if node_id not in best or seconds < best[node_id]:
                    best[node_id] = seconds
        results.append({
            "label": cfg.label,
            "workers": cfg.workers,
            "device": cfg.device,
            "nodes": best,
            "total": sum(best.values()),
        })
    return {"meta": _meta(input_path), "configs": results}


def _build_parser() -> argparse.ArgumentParser:
    from karak.flow.__main__ import _add_flow_args

    parser = argparse.ArgumentParser(
        prog="karak bench",
        description="Time each flow node across configurations.",
    )
    _add_flow_args(parser)
    parser.add_argument("--input", default="", help="Input data directory")
    parser.add_argument("--out", default="output/bench",
                        help="Output basename ({out} token)")
    parser.add_argument("--work", default=None,
                        help="Work directory (default: <out dir>/work)")
    parser.add_argument("--config", action="append", default=[],
                        metavar="SPEC",
                        help="'baseline' or comma list of workers=N,"
                             "device=NAME (repeatable)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per config; per-node minimum is kept")
    parser.add_argument("--qc", action="store_true",
                        help="Include QC figure sinks (skipped by default)")
    parser.add_argument("--set", action="append", default=[],
                        metavar="NODE.PARAM=VALUE",
                        help="Override a node parameter (repeatable)")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Render a table from two bench JSON files")
    return parser


def bench_main(argv: list[str] | None = None) -> int:
    from karak.cli.reporter import render_bench_table

    args = _build_parser().parse_args(argv)

    if args.compare:
        merged = {"meta": None, "configs": []}
        for path in args.compare:
            data = json.loads(Path(path).read_text())
            stem = Path(path).stem
            merged["meta"] = merged["meta"] or data["meta"]
            for cfg in data["configs"]:
                merged["configs"].append({**cfg, "label": f"{stem}:{cfg['label']}"})
        render_bench_table(merged)
        return 0

    from karak.flow.__main__ import _load_graph, _parse_set
    from karak.flow.builtins import override_params

    graph = _load_graph(args)
    if args.set:
        graph = override_params(graph, _parse_set(args.set))
    configs = [parse_config(s) for s in (args.config or ["baseline"])]

    if any(c.device == "cuda" for c in configs):
        from karak import accel

        if not accel.cuda_available():
            raise SystemExit(
                "error: a config requests device=cuda but no usable GPU "
                "stack was found. Install with: pip install 'karak[cuda]'"
            )

    work_dir = args.work or str(Path(args.out).parent / "work")
    result = run_bench(
        graph,
        input_path=args.input,
        out_base=args.out,
        work_dir=work_dir,
        configs=configs,
        repeats=args.repeats,
        include_qc=args.qc,
    )
    json_path = Path(f"{args.out}_bench.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {json_path}")
    render_bench_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(bench_main())
```

Add to `src/karak/cli/reporter.py`:

```python
def render_bench_table(result: dict) -> None:
    """Rich table: nodes as rows, configs as columns, speedup vs first."""
    from rich.console import Console
    from rich.table import Table

    configs = result["configs"]
    meta = result.get("meta") or {}
    title = f"karak bench  ({meta.get('hostname', '?')}, {meta.get('cpus', '?')} cpus"
    if meta.get("gpu"):
        title += f", {meta['gpu']}"
    title += ")"

    table = Table(title=title)
    table.add_column("node")
    for cfg in configs:
        table.add_column(cfg["label"], justify="right")
    if len(configs) > 1:
        table.add_column("speedup", justify="right")

    node_ids = list(configs[0]["nodes"])
    baseline = configs[0]["nodes"]
    for node_id in node_ids:
        row = [node_id]
        for cfg in configs:
            row.append(f"{cfg['nodes'].get(node_id, float('nan')):.2f}s")
        if len(configs) > 1:
            last = configs[-1]["nodes"].get(node_id)
            row.append(f"{baseline[node_id] / last:.1f}x" if last else "-")
        table.add_row(*row)

    totals = ["total"] + [f"{cfg['total']:.2f}s" for cfg in configs]
    if len(configs) > 1 and configs[-1]["total"]:
        totals.append(f"{configs[0]['total'] / configs[-1]['total']:.1f}x")
    table.add_section()
    table.add_row(*totals)
    Console().print(table)
```

Route in `src/karak/cli/main.py` `main()`, before the `_FLOW_COMMANDS` check:

```python
    if argv and argv[0] == "bench":
        from karak.cli.bench import bench_main

        return bench_main(argv[1:])
```

Mirror in `src/karak/flow/__main__.py`: `build_parser()` gains `sub.add_parser("bench", add_help=False)` and `main()` gains, before `_load_graph`:

```python
    if args.command == "bench":
        from karak.cli.bench import bench_main  # lazy: Rich stays in cli/

        return bench_main(sys.argv[2:] if argv is None else argv[1:])
```

(Use `parse_known_args` or route on `argv[0] == "bench"` before full parsing if the stub subparser fights the bench flags; the test only exercises `bench_main` directly, so keep the mirror minimal.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bench.py -v && uv run pytest`
Expected: all PASS

- [ ] **Step 5: Smoke the CLI by hand**

Run: `uv run karak bench --builtin global --input /nonexistent --out /tmp/claude-1000/bench_smoke --config baseline 2>&1 | head -5`
Expected: fails inside the load stage (no input data), which proves routing works; a `FlowError` mentioning `src` is a pass here.

- [ ] **Step 6: Commit**

```bash
git add src/karak/cli/bench.py src/karak/cli/reporter.py src/karak/cli/main.py src/karak/flow/__main__.py tests/test_bench.py
git commit -m "feat: karak bench subcommand with per-node timing and compare"
git push origin main
```

---

### Task 5: Loader `workers`

**Files:**
- Modify: `src/karak/io/loaders.py:254-438` (`load_element_maps`)
- Modify: `src/karak/stages/load.py:95-102` (pass workers)
- Test: `tests/test_loader_workers.py`

**Interfaces:**
- Consumes: `resolve_workers` (Task 1), `stage.workers` (Task 2).
- Produces: `load_element_maps(..., workers: int = 1)`; module-level helpers `_process_element_file(fpath, colormap, base_dir, factor, trims) -> np.ndarray` and `_process_bse_file(fpath, factor, trims) -> np.ndarray` (must be module-level so `ProcessPoolExecutor` can pickle them).

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_loader_workers.py
"""Parity: parallel loading is byte-identical to serial loading."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.config import DownsampleConfig
from karak.io.loaders import load_element_maps


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    """Palette PNGs + grayscale SEM (same recipe as test_flow_end_to_end)."""
    import imageio.v2 as imageio
    import matplotlib

    import karak.io.loaders as loaders

    tmp_path = tmp_path_factory.mktemp("loader_scene")
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


def _load(scene_dir, workers):
    from karak.config import LoaderConfig

    data_dir, colormap = scene_dir
    return load_element_maps(
        str(data_dir),
        DownsampleConfig(header_trim_px=0, downsample_factor=1),
        exclude_elements=[],
        loader_config=LoaderConfig(colormap=colormap),
        workers=workers,
    )


def test_parallel_load_is_byte_identical(scene_dir):
    e1, b1, n1 = _load(scene_dir, workers=1)
    e4, b4, n4 = _load(scene_dir, workers=4)
    assert n1 == n4
    assert np.array_equal(b1, b4)
    for name in n1:
        assert np.array_equal(e1[name], e4[name]), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loader_workers.py -v`
Expected: FAIL with `TypeError: load_element_maps() got an unexpected keyword argument 'workers'`

- [ ] **Step 3: Restructure `load_element_maps`**

Add two module-level helpers above `load_element_maps` (they reuse the existing `_apply_trim`):

```python
def _process_element_file(fpath, colormap, base_dir, factor, trims):
    """Read one false-color map, invert, downsample, trim. Pool-safe."""
    raw = imageio.imread(fpath)
    scalar = invert_colormap(raw, colormap_spec=colormap, base_dir=base_dir)
    ds = downscale_local_mean(scalar, (factor, factor))
    return _apply_trim(ds, **trims).astype(np.float32)


def _process_bse_file(fpath, factor, trims):
    """Read the BSE/SEM image as grayscale, downsample, trim. Pool-safe."""
    raw = imageio.imread(fpath)
    gray = rgb2gray(raw) if raw.ndim == 3 else raw.astype(np.float32) / 255.0
    ds = downscale_local_mean(gray, (factor, factor))
    return _apply_trim(ds, **trims).astype(np.float32)
```

Then restructure the body of `load_element_maps` (signature gains `workers: int = 1` after `loader_config`). Keep discovery, skip/include logic, and all logging exactly as they are, but split decide-from-execute:

1. Build `trims = dict(trim_top=trim_top, trim_bottom=trim_bottom, trim_left=trim_left, trim_right=trim_right)`.
2. Walk the sorted file list exactly as today, but instead of loading, append jobs: `("bse", bse_channel, fpath)` for the BSE branches, `("element", element, fpath)` for kept elements. The skip/exclude/include `logger.info` lines stay in this loop unchanged.
3. Execute jobs. Serial when `workers <= 1` (call the helpers directly, in job order, and keep the existing per-element `logger.info("Loaded element ...")` lines with the same fields). Parallel otherwise:

```python
    if workers > 1 and jobs:
        # Warm the LUT cache once so pool workers load it from disk
        # instead of each building a fresh one.
        get_full_lut(loader.colormap, base_dir=input_dir)
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = []
            for kind, key, fpath in jobs:
                if kind == "bse":
                    futures.append(pool.submit(_process_bse_file, fpath, factor, trims))
                else:
                    futures.append(pool.submit(
                        _process_element_file, fpath, loader.colormap,
                        input_dir, factor, trims,
                    ))
            arrays = [f.result() for f in futures]
    else:
        arrays = [
            _process_bse_file(fpath, factor, trims) if kind == "bse"
            else _process_element_file(fpath, loader.colormap, input_dir,
                                       factor, trims)
            for kind, key, fpath in jobs
        ]
```

4. Assemble: walk `jobs` and `arrays` together; `bse_array = arr` for the bse job, `elements_dict[key] = arr` otherwise; emit the "Loaded element" / "Loaded BSE channel" log lines here so serial and parallel log identically. The trailing shape check and the missing-BSE error stay unchanged.

In `src/karak/stages/load.py`, thread it through:

```python
        from karak.accel import resolve_workers

        elements, bse, names = load_element_maps(
            params["input_dir"],
            downsample,
            exclude_elements=_split_csv(params["exclude_elements"]),
            bse_channel=params["bse_channel"],
            include_elements=include,
            loader_config=loader,
            workers=resolve_workers(self.workers),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loader_workers.py -v && uv run pytest`
Expected: all PASS (the end-to-end flow test exercises the serial path unchanged)

- [ ] **Step 5: Measure on the publication sample**

Run:
```bash
uv run python - <<'EOF'
import time
from karak.config import DownsampleConfig
from karak.io.loaders import load_element_maps
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

DATA = "/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/data"
for workers in (1, 12):
    t0 = time.perf_counter()
    load_element_maps(DATA, DownsampleConfig(), exclude_elements=["Fe-L"], workers=workers)
    print(f"workers={workers}: {time.perf_counter() - t0:.1f} s")
EOF
```
Expected: workers=12 several times faster than workers=1 (baseline about 60 s). Record both numbers in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/karak/io/loaders.py src/karak/stages/load.py tests/test_loader_workers.py
git commit -m "feat: parallel element-map loading (workers), byte-identical"
git push origin main
```

---

### Task 6: Denoise `workers` (CPU)

**Files:**
- Modify: `src/karak/preprocessing/denoise.py:34-171` (both cube denoisers), `denoise_cube` (line 364)
- Modify: `src/karak/stages/denoise.py` (pass workers into `denoise_cube`)
- Test: `tests/test_denoise_workers.py`

**Interfaces:**
- Consumes: `resolve_workers`, `stage.workers`.
- Produces: `bilateral_denoise_cube(..., workers: int = 1)`, `anisotropic_denoise_cube(..., workers: int = 1)`, `denoise_cube(..., workers: int = 1)`; module-level `_bilateral_channel(channel, mask, sigma_color, sigma_spatial)` and `_anisotropic_channel(channel, mask, niter, kappa, gamma, option)`.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_denoise_workers.py
"""Parity: parallel denoising is byte-identical to serial."""

from __future__ import annotations

import numpy as np

from conftest import make_synthetic_scene
from karak.preprocessing.denoise import (
    anisotropic_denoise_cube,
    bilateral_denoise_cube,
)


def _cube_and_mask():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    return cube, mask


def test_bilateral_parallel_parity():
    cube, mask = _cube_and_mask()
    serial = bilateral_denoise_cube(cube, mask, workers=1)
    parallel = bilateral_denoise_cube(cube, mask, workers=3)
    assert np.array_equal(serial, parallel)


def test_anisotropic_parallel_parity():
    cube, mask = _cube_and_mask()
    serial = anisotropic_denoise_cube(cube, mask, workers=1)
    parallel = anisotropic_denoise_cube(cube, mask, workers=3)
    assert np.array_equal(serial, parallel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_denoise_workers.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'workers'`

- [ ] **Step 3: Implement**

Extract the existing per-channel bodies into module-level functions (the code inside each `for i in range(C)` loop, verbatim):

```python
def _bilateral_channel(channel, mask, sigma_color, sigma_spatial):
    """One channel of bilateral_denoise_cube. Pool-safe."""
    channel = channel.copy()
    channel[~mask] = np.nanmean(channel[mask])
    return denoise_bilateral(
        channel, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
    )


def _anisotropic_channel(channel, mask, niter, kappa, gamma, option):
    """One channel of anisotropic_denoise_cube. Pool-safe."""
    channel = channel.copy()
    channel[~mask] = np.nanmean(channel[mask])
    cmin = channel[mask].min()
    cmax = channel[mask].max()
    if cmax > cmin:
        channel_scaled = (channel - cmin) / (cmax - cmin)
    else:
        return channel
    diffused = anisotropic_diffusion(
        channel_scaled, niter=niter, kappa=kappa, gamma=gamma, option=option,
    )
    return diffused * (cmax - cmin) + cmin
```

(Note the `cmax <= cmin` branch returns the mean-filled `channel`, which is what the current loop assigns in that case.)

Both cube functions gain `workers: int = 1` and replace their loops with:

```python
    args = [(cube[:, :, i].copy(), mask, sigma_color, sigma_spatial)
            for i in range(C)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            channels = list(pool.map(_bilateral_channel, *zip(*args)))
    else:
        channels = [_bilateral_channel(*a) for a in args]
    for i, ch in enumerate(channels):
        denoised[:, :, i] = ch
        logger.info("Bilateral denoise: channel %d/%d done", i + 1, C)
```

(the anisotropic version mirrors this with `_anisotropic_channel` and its arg tuple). The mask re-apply and completion log lines stay as they are.

`denoise_cube` (line 364) gains `workers: int = 1` and forwards it to whichever method it dispatches to. In `src/karak/stages/denoise.py`, the stage's `apply` passes `workers=resolve_workers(self.workers)` into `denoise_cube` (import `resolve_workers` from `karak.accel`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_denoise_workers.py -v && uv run pytest`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/karak/preprocessing/denoise.py src/karak/stages/denoise.py tests/test_denoise_workers.py
git commit -m "feat: parallel per-channel denoising (workers), byte-identical"
git push origin main
```

---

### Task 7: `karak[cuda]` extra and denoise `device` (GPU)

**Files:**
- Modify: `pyproject.toml` (optional-dependencies)
- Modify: `src/karak/preprocessing/denoise.py` (`bilateral_denoise_cube`, `denoise_cube`)
- Modify: `src/karak/stages/denoise.py` (declare `device` param)
- Modify: `docs/stage_reference.md` (regenerate)
- Test: `tests/test_denoise_gpu.py`

**Interfaces:**
- Consumes: `get_array_module`, `to_device`, `to_numpy`, `cuda_available` (Task 1).
- Produces: `bilateral_denoise_cube(..., device: str = "cpu")`, `denoise_cube(..., device: str = "cpu")`; `device` Param on the denoise stage.

- [ ] **Step 1: Add the extra to `pyproject.toml`**

Under `[project.optional-dependencies]` (create the table if absent; check whether `dev` lives there or in `[dependency-groups]` and leave it where it is):

```toml
[project.optional-dependencies]
cuda = [
    "cupy-cuda12x>=13",
    "cuml-cu12>=25.2",
    "cucim-cu12>=25.2",
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_denoise_gpu.py
"""GPU bilateral denoise: exact-tier parity with the CPU path."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.accel import cuda_available
from karak.preprocessing.denoise import bilateral_denoise_cube, denoise_cube
from karak.stages.base import StageError


def test_cuda_without_gpu_raises(monkeypatch):
    import karak.accel as accel

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    with pytest.raises(StageError):
        bilateral_denoise_cube(cube, mask, device="cuda")


def test_anisotropic_cuda_unsupported():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    with pytest.raises(StageError, match="anisotropic"):
        denoise_cube(cube, mask, method="anisotropic", device="cuda")


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_bilateral_gpu_matches_cpu():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cpu = bilateral_denoise_cube(cube, mask)
    gpu = bilateral_denoise_cube(cube, mask, device="cuda")
    assert gpu.dtype == cpu.dtype
    assert np.allclose(cpu, gpu, atol=1e-5)
```

Adjust the `denoise_cube` call signature in the second test to match the real dispatcher (read `denoise.py:364` first; the method argument may be named differently, keep the assertion the same).

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_denoise_gpu.py -v`
Expected: the two CPU-runnable tests FAIL (`unexpected keyword argument 'device'`); the GPU test SKIPs.

- [ ] **Step 4: Implement**

`bilateral_denoise_cube` gains `device: str = "cpu"` and branches before the channel loop:

```python
    if device == "cuda":
        from karak.accel import get_array_module, to_numpy

        cp = get_array_module(device)  # raises StageError without CUDA
        from cucim.skimage.restoration import denoise_bilateral as gpu_bilateral

        gpu_cube = cp.asarray(cube)
        gpu_mask = cp.asarray(mask)
        out = cp.zeros_like(gpu_cube)
        for i in range(C):
            channel = gpu_cube[:, :, i].copy()
            channel[~gpu_mask] = cp.nanmean(channel[gpu_mask])
            out[:, :, i] = gpu_bilateral(
                channel, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
            )
        out[~gpu_mask] = 0.0
        return to_numpy(out).astype(cube.dtype)
```

`denoise_cube` gains `device: str = "cpu"`, forwards it to the bilateral path, and raises `StageError("device='cuda' supports only method='bilateral'; anisotropic diffusion has no GPU path")` for the anisotropic method.

In `src/karak/stages/denoise.py`, append to `PARAMS`:

```python
        Param("device", "str", "cpu", "Device",
              "cpu or cuda (GPU bilateral via cuCIM; needs karak[cuda]). "
              "Results match cpu within float tolerance.",
              choices=("cpu", "cuda")),
```

and pass `device=params["device"]` into `denoise_cube`.

- [ ] **Step 5: Regenerate the stage reference and run the suite**

Run: `uv run python -m karak.stages.reference && uv run pytest`
Expected: all PASS, GPU test skipped; `docs/stage_reference.md` shows the new param (the drift test pins this).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/karak/preprocessing/denoise.py src/karak/stages/denoise.py docs/stage_reference.md tests/test_denoise_gpu.py
git commit -m "feat: karak[cuda] extra + GPU bilateral denoise (device param)"
git push origin main
```

---

### Task 8: Tiled clustering `workers`

**Files:**
- Modify: `src/karak/clustering/tiling.py:471-540` (`run_tiled_hdbscan`)
- Modify: `src/karak/stages/cluster.py:97-120` (`HdbscanTiledStage.apply`)
- Test: `tests/test_tiling_workers.py`

**Interfaces:**
- Consumes: `resolve_workers`, `stage.workers`, `run_hdbscan(tile_features, hdb_cfg)`.
- Produces: `run_tiled_hdbscan(..., workers: int = 1)`.

**Design note:** the phase registry (fingerprint running means, registry matching) is order-dependent, so only the per-tile `run_hdbscan` call parallelizes. The merge loop stays sequential in tile order, which keeps results byte-identical.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/test_tiling_workers.py
"""Parity: parallel per-tile HDBSCAN is byte-identical to serial."""

from __future__ import annotations

import numpy as np
import pytest

from karak.config import ClusterConfig, HDBSCANConfig, TiledConfig
from karak.clustering.tiling import run_tiled_hdbscan


@pytest.fixture(scope="module")
def tiled_inputs():
    """A 64x64 scene with two blobs, split into 4 tiles of 32px."""
    rng = np.random.default_rng(0)
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    mineral = np.ones((H, W), dtype=bool)
    mineral_indices = np.column_stack(np.nonzero(mineral)).astype(np.int32)

    blob = (xx >= 32).astype(np.float32)
    features = np.column_stack([
        blob.ravel() + rng.normal(0, 0.05, H * W),
        (1 - blob).ravel() + rng.normal(0, 0.05, H * W),
    ]).astype(np.float32)

    cube = np.stack([blob, 1 - blob, np.ones((H, W), np.float32)], axis=-1)
    config = ClusterConfig(
        strategy="tiled",
        hdbscan=HDBSCANConfig(min_cluster_size=50, random_state=0),
        tiled=TiledConfig(tile_size=32, merge_threshold=0.9,
                          min_tile_pixels=100, min_clusters_per_tile=1),
    )
    return features, mineral_indices, (H, W), cube, config


def _run(tiled_inputs, workers):
    features, indices, shape, cube, config = tiled_inputs
    return run_tiled_hdbscan(
        features, indices, shape, cube, config,
        skip_knn=True, workers=workers,
    )


def test_parallel_tiles_byte_identical(tiled_inputs):
    raw1, clean1, prob1, tiles1, reg1 = _run(tiled_inputs, workers=1)
    raw2, clean2, prob2, tiles2, reg2 = _run(tiled_inputs, workers=3)
    assert np.array_equal(raw1, raw2)
    assert np.array_equal(prob1, prob2)
    assert len(reg1) == len(reg2)
    for e1, e2 in zip(reg1, reg2):
        assert e1.global_id == e2.global_id
        assert e1.n_pixels == e2.n_pixels
        assert np.allclose(e1.mean_fingerprint, e2.mean_fingerprint)
```

If `ClusterConfig`/`TiledConfig` field names differ, align with `src/karak/config.py:325` before running; the stage code in `stages/cluster.py:101-110` shows the exact constructor shapes.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tiling_workers.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'workers'`

- [ ] **Step 3: Implement**

`run_tiled_hdbscan` gains `workers: int = 1`. After `tiles = compute_tile_grid(...)` and before the merge loop, precompute the per-tile HDBSCAN results:

```python
    if workers > 1 and len(tiles) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                tile.tile_id: pool.submit(
                    run_hdbscan, pca_features[tile.pixel_indices], hdb_cfg,
                )
                for tile in tiles
            }
            tile_hdbscan = {tid: f.result() for tid, f in futures.items()}
    else:
        tile_hdbscan = None
```

In the merge loop, replace the direct call at line 540:

```python
        if tile_hdbscan is not None:
            tile_labels, tile_probs, _ = tile_hdbscan[tile.tile_id]
        else:
            tile_labels, tile_probs, _ = run_hdbscan(tile_features, hdb_cfg)
```

Everything after (fingerprints, registry matching, merge maps) is untouched and stays in tile order.

In `src/karak/stages/cluster.py`, `HdbscanTiledStage.apply` passes it through:

```python
        from karak.accel import resolve_workers

        raw_labels, _, probabilities, tile_results, phase_registry = (
            run_tiled_hdbscan(
                features.features,
                features.mineral_indices,
                features.image_shape,
                cube.pixels,
                config,
                skip_knn=True,
                workers=resolve_workers(self.workers),
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tiling_workers.py -v && uv run pytest`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/karak/clustering/tiling.py src/karak/stages/cluster.py tests/test_tiling_workers.py
git commit -m "feat: parallel per-tile HDBSCAN (workers), byte-identical merge"
git push origin main
```

---

### Task 9: cuML HDBSCAN `device` (loose tier)

**Files:**
- Modify: `src/karak/clustering/hdbscan_cluster.py:21-105` (`run_hdbscan`)
- Modify: `src/karak/clustering/tiling.py` (thread `device` into per-tile calls)
- Modify: `src/karak/stages/cluster.py:17-26` (`_HDBSCAN_PARAMS`) and both `apply` methods
- Modify: `docs/stage_reference.md` (regenerate)
- Test: `tests/test_hdbscan_gpu.py`

**Interfaces:**
- Consumes: `get_array_module`, `to_numpy`, `cuda_available`.
- Produces: `run_hdbscan(pca_features, config, device: str = "cpu")`; `run_tiled_hdbscan(..., device: str = "cpu")`; `device` Param on `hdbscan_global` and `hdbscan_tiled`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hdbscan_gpu.py
"""cuML HDBSCAN device path: loose tier (shape and sanity only)."""

from __future__ import annotations

import numpy as np
import pytest

from karak.accel import cuda_available
from karak.clustering.hdbscan_cluster import run_hdbscan
from karak.config import HDBSCANConfig
from karak.stages.base import StageError


def _features():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (500, 3))
    b = rng.normal(5, 0.1, (500, 3))
    return np.vstack([a, b]).astype(np.float32)


def test_cuda_without_gpu_raises(monkeypatch):
    import karak.accel as accel

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    with pytest.raises(StageError):
        run_hdbscan(_features(), HDBSCANConfig(min_cluster_size=100),
                    device="cuda")


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_hdbscan_shapes_and_sanity():
    features = _features()
    labels, probs, _ = run_hdbscan(
        features, HDBSCANConfig(min_cluster_size=100), device="cuda",
    )
    assert labels.shape == (1000,)
    assert labels.dtype == np.int32
    assert probs.shape == (1000,)
    assert probs.dtype == np.float32
    assert len(set(labels.tolist()) - {-1}) == 2
```

If `HDBSCANConfig(min_cluster_size=100)` needs more required fields, copy the construction from `stages/cluster.py:29-35`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hdbscan_gpu.py -v`
Expected: the no-GPU test FAILS (`unexpected keyword argument 'device'`); the CUDA test SKIPs.

- [ ] **Step 3: Implement**

`run_hdbscan` gains `device: str = "cpu"` and branches at the top:

```python
    if device == "cuda":
        from karak.accel import get_array_module, to_numpy

        cp = get_array_module(device)  # raises StageError without CUDA
        from cuml.cluster import HDBSCAN as CumlHDBSCAN

        min_samples = (config.min_samples if config.min_samples is not None
                       else config.min_cluster_size)
        model = CumlHDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
        )
        model.fit(cp.asarray(pca_features, dtype=cp.float32))
        labels = to_numpy(model.labels_).astype(np.int32)
        probabilities = to_numpy(model.probabilities_).astype(np.float32)
        return labels, probabilities, model
```

(The cuda branch fits all pixels and ignores `subsample_n`; note that in the param help below.)

`run_tiled_hdbscan` gains `device: str = "cpu"` and passes `device=device` to every `run_hdbscan` call, including the ones submitted to the process pool. GPU work inside a process pool would contend for the device, so when `device == "cuda"`, force the serial tile path regardless of `workers` (add `and device == "cpu"` to the pool condition from Task 8).

In `src/karak/stages/cluster.py`, append to `_HDBSCAN_PARAMS`:

```python
    Param("device", "str", "cpu", "Device",
          "cpu (hdbscan package, exact baseline) or cuda (cuML; needs "
          "karak[cuda]). cuda results differ from cpu and ignore "
          "subsample_n.",
          choices=("cpu", "cuda")),
```

Both `apply` methods pass `device=params["device"]` (global into `run_hdbscan`, tiled into `run_tiled_hdbscan`).

- [ ] **Step 4: Regenerate reference, run the suite**

Run: `uv run python -m karak.stages.reference && uv run pytest`
Expected: all PASS, GPU tests skipped.

- [ ] **Step 5: Commit**

```bash
git add src/karak/clustering/hdbscan_cluster.py src/karak/clustering/tiling.py src/karak/stages/cluster.py docs/stage_reference.md tests/test_hdbscan_gpu.py
git commit -m "feat: cuML HDBSCAN device param on cluster stages (loose tier)"
git push origin main
```

---

### Task 10: Docs and the first real benchmark

**Files:**
- Modify: `README.md` (commands section), `CLAUDE.md` (commands section)
- Create: `benchmarks/2026-09-18-cpu-12core.json` (bench output, committed)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Document the new flags**

Add to the command blocks in `README.md` and `CLAUDE.md`:

```bash
karak run ... --workers 0                  # parallel stages on all cores (same results)
karak run ... --device cuda                # GPU paths where stages support it
karak bench --builtin tiled --input DIR --out BASE \
    --config baseline --config workers=0   # per-node timing comparison
karak bench --compare a.json b.json        # cross-machine table
```

- [ ] **Step 2: Run the CPU benchmark on the publication sample**

```bash
mkdir -p benchmarks
uv run karak bench --builtin tiled \
    --input "/home/brendon/Dropbox/Projects/izawa/NWA4587_LPSC26/data" \
    --out output/nwa4587_bench \
    --config baseline --config workers=0
cp output/nwa4587_bench_bench.json benchmarks/2026-09-18-cpu-12core.json
```

Expected: the table shows `src`, `dn`, and `hdb` speedups; runtime is two full flow runs, so expect roughly an hour. Run it in the background and report the table when done.

- [ ] **Step 3: Run the full suite one last time**

Run: `uv run pytest`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md benchmarks/2026-09-18-cpu-12core.json
git commit -m "docs: acceleration flags + first CPU benchmark on NWA 4587"
git push origin main
```

---

## Deferred (spec-gated, no tasks)

- PCA, normalize, and KNN acceleration: only if `benchmarks/2026-09-18-cpu-12core.json` shows them as material costs.
- QC figure parallelism: same gate.
- GPU validation and `--compare` across machines: waits for SSH access to the 4090 host; the GPU tests in Tasks 7 and 9 then run there with `uv run pytest tests/test_denoise_gpu.py tests/test_hdbscan_gpu.py`.
