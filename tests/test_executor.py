"""Tests for the flow executor: topo order, tokens, caching, eviction."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from conftest import complete

from karak.flow.executor import FlowError, PayloadStore, run
from karak.flow.graph import Edge, Endpoint, Graph, Node
from karak.stages import registry
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import BseImage, ClusterStats


RECORD: list = []


class FakeSource(Stage):
    id = "fake_source"
    label = "Fake source"
    OUTPUTS = [Port("num")]
    PARAMS = [
        Param("value", "int", 1),
        Param("path", "str", "unset"),
    ]

    def apply(self, inputs, params):
        RECORD.append(("fake_source", params["path"]))
        return {"num": ClusterStats(stats={"value": params["value"]})}


class FakeAdd(Stage):
    id = "fake_add"
    label = "Fake add"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]
    PARAMS = [Param("add", "int", 0)]

    def apply(self, inputs, params):
        RECORD.append(("fake_add", params["add"]))
        value = inputs["num"].stats["value"] + params["add"]
        return {"num": ClusterStats(stats={"value": value})}


class FakeSink(Stage):
    id = "fake_sink"
    label = "Fake sink"
    INPUTS = [Port("num")]
    OUTPUTS: list = []
    PARAMS = [Param("out", "str", "{out}")]

    def apply(self, inputs, params):
        RECORD.append(("fake_sink", inputs["num"].stats["value"], params["out"]))
        return {}


@pytest.fixture(autouse=True)
def _fake_stages():
    for cls in (FakeSource, FakeAdd, FakeSink):
        registry.register(cls)
    RECORD.clear()
    yield
    for cls in (FakeSource, FakeAdd, FakeSink):
        registry._REGISTRY.pop(cls.id, None)


def _chain_graph(add=2):
    return complete(Graph(
        name="chain",
        nodes=(
            Node("src", "fake_source", {"value": 10, "path": "{input}"}),
            Node("plus", "fake_add", {"add": add}),
            Node("out", "fake_sink"),
        ),
        edges=(
            Edge("e1", Endpoint("src", "num"), Endpoint("plus", "num")),
            Edge("e2", Endpoint("plus", "num"), Endpoint("out", "num")),
        ),
    ))


def test_chain_executes_in_topo_order(tmp_path):
    run(_chain_graph(), input_path="/in", out_base="result",
        work_dir=str(tmp_path))
    assert [r[0] for r in RECORD] == ["fake_source", "fake_add", "fake_sink"]
    assert RECORD[2][1] == 12  # 10 + 2 flowed through


def test_tokens_resolved(tmp_path):
    run(_chain_graph(), input_path="/data/in", out_base="output/x",
        work_dir=str(tmp_path))
    assert RECORD[0] == ("fake_source", "/data/in")
    assert RECORD[2][2] == "output/x"


def test_warm_rerun_skips_producers_but_runs_sinks(tmp_path):
    graph = _chain_graph()
    run(graph, input_path="/in", out_base="o", work_dir=str(tmp_path))
    RECORD.clear()
    summary = run(graph, input_path="/in", out_base="o", work_dir=str(tmp_path))
    names = [r[0] for r in RECORD]
    assert "fake_source" not in names
    assert "fake_add" not in names
    assert names == ["fake_sink"]  # sinks are uncached, always run
    assert summary["src"]["cached"] is True
    assert summary["plus"]["cached"] is True
    assert summary["out"]["cached"] is False


def test_param_change_invalidates_only_downstream(tmp_path):
    run(_chain_graph(add=2), input_path="/in", out_base="o",
        work_dir=str(tmp_path))
    RECORD.clear()
    summary = run(_chain_graph(add=3), input_path="/in", out_base="o",
                  work_dir=str(tmp_path))
    names = [r[0] for r in RECORD]
    assert "fake_source" not in names        # upstream reused
    assert "fake_add" in names               # changed node re-runs
    assert summary["src"]["cached"] is True
    assert summary["plus"]["cached"] is False
    assert RECORD[-1][1] == 13


def test_no_cache_forces_full_run(tmp_path):
    graph = _chain_graph()
    run(graph, input_path="/in", out_base="o", work_dir=str(tmp_path))
    RECORD.clear()
    run(graph, input_path="/in", out_base="o", work_dir=str(tmp_path),
        cache=False)
    assert [r[0] for r in RECORD] == ["fake_source", "fake_add", "fake_sink"]


def test_skip_types_drops_matching_sinks(tmp_path):
    run(_chain_graph(), input_path="/in", out_base="o", work_dir=str(tmp_path),
        skip_types={"fake_sink"})
    assert [r[0] for r in RECORD] == ["fake_source", "fake_add"]


def test_invalid_graph_raises_before_running(tmp_path):
    bad = Graph(nodes=(Node("a", "no_such"),))
    with pytest.raises(FlowError):
        run(bad, work_dir=str(tmp_path))
    assert RECORD == []


def test_reporter_receives_events(tmp_path):
    events: list = []

    class Recorder:
        def node_started(self, node_id, label):
            events.append(("start", node_id))

        def node_finished(self, node_id, seconds, cached):
            events.append(("finish", node_id, cached))

        def progress(self, node_id, done, total, msg=""):
            pass

        def log(self, level, msg):
            pass

    run(_chain_graph(), input_path="/in", out_base="o", work_dir=str(tmp_path),
        reporter=Recorder())
    assert ("start", "src") in events
    assert ("finish", "src", False) in events


def test_payload_store_refcount_eviction():
    store = PayloadStore({("a", "num"): 2})
    payload = ClusterStats(stats={"value": 1})
    store.put("a", "num", payload)
    assert store.get("a", "num") is payload
    assert store.get("a", "num") is payload
    assert ("a", "num") not in store._in_ram  # dropped after last consumer


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


def test_branching_graph_both_consumers_get_payload(tmp_path):
    graph = complete(Graph(
        name="branch",
        nodes=(
            Node("src", "fake_source", {"value": 5}),
            Node("p1", "fake_add", {"add": 1}),
            Node("p2", "fake_add", {"add": 2}),
            Node("s1", "fake_sink"),
            Node("s2", "fake_sink"),
        ),
        edges=(
            Edge("e1", Endpoint("src", "num"), Endpoint("p1", "num")),
            Edge("e2", Endpoint("src", "num"), Endpoint("p2", "num")),
            Edge("e3", Endpoint("p1", "num"), Endpoint("s1", "num")),
            Edge("e4", Endpoint("p2", "num"), Endpoint("s2", "num")),
        ),
    ))
    run(graph, input_path="/in", out_base="o", work_dir=str(tmp_path))
    sink_values = sorted(r[1] for r in RECORD if r[0] == "fake_sink")
    assert sink_values == [6, 7]


class FakeWorkerProbe(Stage):
    id = "fake_worker_probe"
    label = "Fake worker probe"
    OUTPUTS = [Port("num")]
    PARAMS = [Param("value", "int", 1)]

    def apply(self, inputs, params):
        RECORD.append(("workers", self.workers))
        return {"num": ClusterStats(stats={"value": params["value"]})}


def _probe_graph():
    return complete(Graph(
        name="probe",
        nodes=(Node("p", "fake_worker_probe"),),
        edges=(),
    ))


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
    leaf = complete(Graph(name="leaf", nodes=(Node("src", "fake_source", {"value": 3}),)))
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
    leaf = complete(Graph(name="leaf", nodes=(Node("src", "fake_source", {"value": 3}),)))
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
        graph = complete(Graph(name="fail", nodes=(Node("bad", "fake_fail"),)))
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
        graph = complete(Graph(name="probe", nodes=(Node("p1", "fake_node_id_probe"),)))
        run(graph, work_dir=str(tmp_path))
        assert ("node_id", "p1") in RECORD
    finally:
        registry._REGISTRY.pop("fake_node_id_probe", None)


def test_executor_rejects_a_flow_with_missing_params(tmp_path):
    graph = Graph(name="partial", nodes=(Node("src", "fake_source", {"value": 1}),))
    with pytest.raises(FlowError, match="missing param"):
        run(graph, work_dir=str(tmp_path))
    assert RECORD == []


# --- the background cache writer ----------------------------------------


def _chain(n=3):
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
    """A reporter that records node starts, log lines and run_finished."""

    def __init__(self):
        self.events = []

    def node_started(self, node_id, label):
        self.events.append(("start", node_id, time.monotonic()))

    def log(self, level, msg):
        self.events.append(("log", msg, time.monotonic(), level))

    def run_finished(self, summary, seconds):
        self.events.append(("finished", {k: dict(v) for k, v in summary.items()},
                            seconds))

    def __getattr__(self, name):
        return lambda *a, **k: None


def _slow_store(monkeypatch, delay, finished=None):
    from karak.flow import cache

    real = cache.store_payload

    def slow_store(recipe, port, *args, **kwargs):
        time.sleep(delay)
        path = real(recipe, port, *args, **kwargs)
        if finished is not None:
            finished[(recipe, port)] = time.monotonic()
        return path

    monkeypatch.setattr(cache, "store_payload", slow_store)


def test_next_node_starts_before_the_previous_output_is_written(tmp_path, monkeypatch):
    finished = {}
    _slow_store(monkeypatch, 0.3, finished)
    reporter = _Timeline()
    summary = run(_chain(), input_path="x", out_base=str(tmp_path / "o"),
                  work_dir=str(tmp_path / "w"), reporter=reporter)
    starts = {e[1]: e[2] for e in reporter.events if e[0] == "start"}
    src_written = min(finished.values())
    assert starts["add1"] < src_written          # add1 ran while src's file was pending
    assert all(v["write_seconds"] >= 0 for v in summary.values())
    assert summary["src"]["write_seconds"] >= 0.3
    assert summary["out"]["write_seconds"] == 0.0
    logs = [e[1] for e in reporter.events if e[0] == "log"]
    assert any(line.startswith("cache: src.num written in") for line in logs)
    # and every file exists once run() returns
    assert len(list((tmp_path / "w" / "cache").glob("*__num.h5"))) == 3


def test_run_finished_comes_after_the_writer_drains(tmp_path, monkeypatch):
    _slow_store(monkeypatch, 0.3)
    reporter = _Timeline()
    run(_chain(), input_path="x", out_base=str(tmp_path / "o"),
        work_dir=str(tmp_path / "w"), reporter=reporter)
    assert reporter.events[-1][0] == "finished"
    _, final, seconds = reporter.events[-1]
    assert final["src"]["write_seconds"] >= 0.3
    assert seconds >= 0.9                        # three serial 0.3 s writes
    logs = [e for e in reporter.events if e[0] == "log"]
    assert len(logs) == 3                        # every line drained before the end


def test_cache_hits_and_skipped_nodes_have_zero_write_seconds(tmp_path):
    graph = _chain()
    run(graph, input_path="x", out_base="o", work_dir=str(tmp_path))
    summary = run(graph, input_path="x", out_base="o", work_dir=str(tmp_path),
                  skip_types={"fake_sink"})
    assert summary["src"]["cached"] is True
    assert summary["out"]["skipped"] is True
    assert all(v["write_seconds"] == 0.0 for v in summary.values())


class FakeImageSource(Stage):
    id = "fake_image_source"
    label = "Fake image source"
    OUTPUTS = [Port("bse")]
    PARAMS = [Param("value", "float", 1.0)]

    def apply(self, inputs, params):
        return {"bse": BseImage(pixels=np.full((8, 8), params["value"],
                                               dtype=np.float32))}


class FakeImageAdd(Stage):
    id = "fake_image_add"
    label = "Fake image add"
    INPUTS = [Port("bse")]
    OUTPUTS = [Port("bse")]
    PARAMS = [Param("add", "float", 0.0)]

    def apply(self, inputs, params):
        return {"bse": BseImage(pixels=inputs["bse"].pixels + params["add"])}


class FakeImageSink(Stage):
    id = "fake_image_sink"
    label = "Fake image sink"
    INPUTS = [Port("bse")]
    OUTPUTS: list = []

    def apply(self, inputs, params):
        RECORD.append(("fake_image_sink", inputs["bse"].pixels.copy()))
        return {}


@pytest.fixture
def _image_stages():
    classes = (FakeImageSource, FakeImageAdd, FakeImageSink)
    for cls in classes:
        registry.register(cls)
    yield
    for cls in classes:
        registry._REGISTRY.pop(cls.id, None)


def test_spilled_reload_waits_for_the_pending_write(tmp_path, monkeypatch,
                                                    _image_stages):
    _slow_store(monkeypatch, 0.3)
    graph = complete(Graph(name="images", nodes=(
        Node("src", "fake_image_source", {"value": 1.0}),
        Node("add1", "fake_image_add", {"add": 1.0}),
        Node("add2", "fake_image_add", {"add": 2.0}),
        Node("out", "fake_image_sink"),
    ), edges=(
        Edge("e1", Endpoint("src", "bse"), Endpoint("add1", "bse")),
        Edge("e2", Endpoint("add1", "bse"), Endpoint("add2", "bse")),
        Edge("e3", Endpoint("add2", "bse"), Endpoint("out", "bse")),
    )))
    # a zero budget spills every payload, so each consumer reloads its
    # input from a file the writer may not have finished yet
    summary = run(graph, input_path="x", out_base=str(tmp_path / "o"),
                  work_dir=str(tmp_path / "w"), ram_budget=0)
    assert summary["add2"]["cached"] is False
    name, pixels = RECORD[-1]
    assert name == "fake_image_sink"
    np.testing.assert_array_equal(pixels, np.full((8, 8), 4.0, np.float32))


def test_run_passes_the_ram_budget_and_logs_spills(tmp_path, _image_stages):
    class Logger:
        def __init__(self):
            self.lines = []

        def log(self, level, msg):
            self.lines.append(msg)

        def __getattr__(self, name):
            return lambda *a, **k: None

    graph = complete(Graph(name="images", nodes=(
        Node("src", "fake_image_source", {"value": 1.0}),
        Node("add", "fake_image_add", {"add": 1.0}),
        Node("out", "fake_image_sink"),
    ), edges=(
        Edge("e1", Endpoint("src", "bse"), Endpoint("add", "bse")),
        Edge("e2", Endpoint("add", "bse"), Endpoint("out", "bse")),
    )))
    reporter = Logger()
    run(graph, input_path="x", out_base=str(tmp_path / "o"),
        work_dir=str(tmp_path / "w"), reporter=reporter, ram_budget=0)
    assert any(line.startswith("store: src.bse (") and "spilled to cache" in line
               for line in reporter.lines)


def test_writer_failure_fails_the_run(tmp_path, monkeypatch):
    from karak.flow import cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    reporter = _Timeline()
    with pytest.raises(FlowError, match="cache writer: disk full"):
        run(_chain(), input_path="x", out_base=str(tmp_path / "o"),
            work_dir=str(tmp_path / "w"), reporter=reporter)
    # the FlowError carries the message; the executor does not log it too
    assert not [e for e in reporter.events if e[0] == "log" and e[3] == "error"]


_WRITE_FAILED = threading.Event()


class WaitAdd(Stage):
    """Adds like FakeAdd, but only once the writer has failed."""

    id = "fake_wait_add"
    label = "Wait add"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]
    PARAMS: list = []

    def apply(self, inputs, params):
        assert _WRITE_FAILED.wait(5)
        RECORD.append(("fake_wait_add",))
        return {"num": ClusterStats(stats={"value": inputs["num"].stats["value"]})}


def test_writer_error_stops_the_run_at_the_next_node(tmp_path, monkeypatch):
    from karak.flow import cache

    _WRITE_FAILED.clear()

    def failing_store(*args, **kwargs):
        try:
            raise OSError("disk full")
        finally:
            _WRITE_FAILED.set()

    monkeypatch.setattr(cache, "store_payload", failing_store)
    registry.register(WaitAdd)
    try:
        graph = complete(Graph(name="stop", nodes=(
            Node("src", "fake_source", {"value": 1, "path": "x"}),
            Node("mid", "fake_wait_add", {}),
            Node("out", "fake_sink"),
        ), edges=(
            Edge("e1", Endpoint("src", "num"), Endpoint("mid", "num")),
            Edge("e2", Endpoint("mid", "num"), Endpoint("out", "num")),
        )))
        with pytest.raises(FlowError, match="cache writer: disk full"):
            run(graph, input_path="x", out_base=str(tmp_path / "o"),
                work_dir=str(tmp_path / "w"))
    finally:
        registry._REGISTRY.pop(WaitAdd.id, None)
    assert [r[0] for r in RECORD] == ["fake_source", "fake_wait_add"]


def test_run_start_sweeps_stale_tmp_files(tmp_path):
    import os
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    cache_dir = tmp_path / "w" / "cache"
    cache_dir.mkdir(parents=True)
    dead = cache_dir / f"abc__num.h5.{proc.pid}.tmp"
    live = cache_dir / f"def__num.h5.{os.getpid()}.tmp"
    dead.write_bytes(b"x")
    live.write_bytes(b"x")
    run(_chain(), input_path="x", out_base=str(tmp_path / "o"),
        work_dir=str(tmp_path / "w"))
    assert not dead.exists()
    assert live.exists()


class Boom(Stage):
    id = "fake_boom"
    label = "Boom"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]
    PARAMS: list = []

    def apply(self, inputs, params):
        raise RuntimeError("boom")


class Interrupt(Stage):
    id = "fake_interrupt"
    label = "Interrupt"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]
    PARAMS: list = []

    def apply(self, inputs, params):
        raise KeyboardInterrupt


@pytest.fixture
def _failing_stages():
    for cls in (Boom, Interrupt):
        registry.register(cls)
    yield
    for cls in (Boom, Interrupt):
        registry._REGISTRY.pop(cls.id, None)


def _src_then(node_type):
    return complete(Graph(name="fail", nodes=(
        Node(id="src", type="fake_source", params={"value": 1, "path": "x"}),
        Node(id="b", type=node_type, params={}),
    ), edges=(Edge(id="e", src=Endpoint("src", "num"), dst=Endpoint("b", "num")),)))


def test_failing_node_still_gets_earlier_outputs_written(tmp_path, monkeypatch,
                                                        _failing_stages):
    _slow_store(monkeypatch, 0.2)
    with pytest.raises(FlowError, match="boom"):
        run(_src_then("fake_boom"), input_path="x", out_base=str(tmp_path / "o"),
            work_dir=str(tmp_path / "w"))
    assert list((tmp_path / "w" / "cache").glob("*__num.h5"))   # src landed


def test_interrupt_drains_the_writer(tmp_path, monkeypatch, _failing_stages):
    _slow_store(monkeypatch, 0.2)
    with pytest.raises(KeyboardInterrupt):
        run(_src_then("fake_interrupt"), input_path="x",
            out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"))
    assert list((tmp_path / "w" / "cache").glob("*__num.h5"))


def test_writer_error_during_a_failure_is_logged_not_raised(
        tmp_path, monkeypatch, _failing_stages):
    from karak.flow import cache

    def failing_store(*args, **kwargs):
        time.sleep(0.1)
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    reporter = _Timeline()
    with pytest.raises(FlowError, match="boom"):     # the original error wins
        run(_src_then("fake_boom"), input_path="x",
            out_base=str(tmp_path / "o"), work_dir=str(tmp_path / "w"),
            reporter=reporter)
    errors = [e[1] for e in reporter.events if e[0] == "log" and e[3] == "error"]
    assert errors == ["cache writer: disk full"]


def test_no_cache_creates_no_writer_and_writes_nothing(tmp_path, monkeypatch):
    from karak.flow import cache

    def forbidden(*args, **kwargs):
        raise AssertionError("store_payload must not be called with cache=False")

    monkeypatch.setattr(cache, "store_payload", forbidden)
    summary = run(_chain(), input_path="x", out_base=str(tmp_path / "o"),
                  work_dir=str(tmp_path / "w"), cache=False)
    assert not (tmp_path / "w" / "cache").exists()
    assert all(v["write_seconds"] == 0.0 for v in summary.values())


def test_writer_queue_is_bounded_by_the_ram_budget(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}
    real = executor.CacheWriter

    class Spy(real):
        def __init__(self, *args, **kwargs):
            seen.update(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(executor, "CacheWriter", Spy)
    run(_chain(), input_path="x", out_base=str(tmp_path / "o"),
        work_dir=str(tmp_path / "w"), ram_budget=12345)
    assert seen["max_pending_bytes"] == 12345
