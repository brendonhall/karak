"""Tests for the flow executor: topo order, tokens, caching, eviction."""

from __future__ import annotations

import pytest

from karak.flow.executor import FlowError, PayloadStore, run
from karak.flow.graph import Edge, Endpoint, Graph, Node
from karak.stages import registry
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import ClusterStats


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
    return Graph(
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
    )


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


def test_payload_store_spills_large_payloads():
    import numpy as np

    from karak.stages.payloads import BseImage

    big = BseImage(pixels=np.zeros((64, 64), dtype=np.float32))
    reloads = []

    def reload(node, port):
        reloads.append((node, port))
        return big

    store = PayloadStore({("a", "bse"): 2}, reload=reload, spill_threshold=1)
    store.put("a", "bse", big)
    assert ("a", "bse") not in store._in_ram  # never held in RAM
    assert store.get("a", "bse") is big
    assert store.get("a", "bse") is big
    assert len(reloads) == 2  # reloaded once per consumer


def test_branching_graph_both_consumers_get_payload(tmp_path):
    graph = Graph(
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
    )
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
