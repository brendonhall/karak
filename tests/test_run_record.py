"""Run records: every run keeps the complete flow it executed, and what happened."""

from __future__ import annotations

import json

import pytest

from conftest import complete
from karak.flow.executor import FlowError, plan_recipes, run
from karak.flow.graph import Edge, Endpoint, Graph, Node
from karak.flow.record import RunRecord, latest_run_dir
from karak.flow.validate import validate
from karak.stages import registry
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import ClusterStats


class RecSource(Stage):
    id = "rec_source"
    label = "Record source"
    OUTPUTS = [Port("num")]
    PARAMS = [Param("value", "int", 1), Param("path", "str", "{input}")]

    def apply(self, inputs, params):
        return {"num": ClusterStats(stats={"value": params["value"]})}


class RecSink(Stage):
    id = "rec_sink"
    label = "Record sink"
    INPUTS = [Port("num")]
    PARAMS = [Param("out", "str", "{out}"), Param("flow_json", "str", "{flow}")]

    def apply(self, inputs, params):
        return {}


class RecBoom(Stage):
    id = "rec_boom"
    label = "Record boom"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        raise ValueError("boom in stage")


class RecStop(Stage):
    id = "rec_stop"
    label = "Record stop"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        raise KeyboardInterrupt


@pytest.fixture(autouse=True)
def _stages():
    for cls in (RecSource, RecSink, RecBoom, RecStop):
        registry.register(cls)
    yield
    for cls in (RecSource, RecSink, RecBoom, RecStop):
        registry._REGISTRY.pop(cls.id, None)


def _graph(middle=None):
    nodes = [Node("src", "rec_source", {"value": 7})]
    edges = []
    last = "src"
    if middle:
        nodes.append(Node("mid", middle))
        edges.append(Edge("e0", Endpoint("src", "num"), Endpoint("mid", "num")))
        last = "mid"
    nodes.append(Node("out", "rec_sink"))
    edges.append(Edge("e1", Endpoint(last, "num"), Endpoint("out", "num")))
    return complete(Graph(name="rec", nodes=tuple(nodes), edges=tuple(edges)))


def _run(tmp_path, graph, **kwargs):
    out = tmp_path / "out" / "run"
    record = RunRecord(
        str(out), graph,
        argv=["run", "flow.json"], source="flow.json",
        tokens={"input": "/data/in", "out": str(out), "work": str(tmp_path / "work")},
        settings={"workers": None, "cache": True, "no_qc": False},
        overrides={"src.value": 7},
    )
    run(graph, input_path="/data/in", out_base=str(out),
        work_dir=str(tmp_path / "work"), record=record, **kwargs)
    return record


def _load(record):
    return json.loads((record.path / "run.json").read_text())


def test_ok_run_records_complete_flow_params_and_outputs(tmp_path):
    graph = _graph()
    record = _run(tmp_path, graph)
    data = _load(record)

    assert record.path.parent == tmp_path / "out" / "run" / "runs"
    assert json.loads((record.path / "flow.json").read_text()) == graph.to_json()
    assert data["status"] == "ok" and data["error"] is None
    assert data["finished"] is not None and data["seconds"] >= 0
    assert data["flow"] == {"name": "rec", "source": "flow.json", "file": "flow.json"}
    assert data["argv"] == ["run", "flow.json"]
    assert data["tokens"]["input"] == "/data/in"
    assert data["overrides"] == {"src.value": 7}
    assert data["settings"]["cache"] is True
    assert data["karak"]["version"]
    assert "commit" in (data["karak"]["git"] or {"commit": None})
    assert data["libraries"]["numpy"]

    src = data["nodes"]["src"]
    assert src["type"] == "rec_source"
    assert src["status"] == "ran"
    assert src["params"] == {"value": 7, "path": "/data/in"}   # resolved
    tokens = {"{input}": "/data/in", "{out}": str(tmp_path / "out" / "run"),
              "{work}": str(tmp_path / "work"),
              "{flow}": json.dumps(graph.to_json())}
    assert src["recipe"] == plan_recipes(graph, tokens)["src"]
    output = src["outputs"]["num"]
    assert output["summary"] == "ClusterStats 1 entries"
    assert (tmp_path / "work" / "cache" / output["file"].split("/")[-1]).exists()

    sink = data["nodes"]["out"]
    assert sink["status"] == "ran"
    assert sink["params"]["flow_json"] == "{flow}"   # not the whole JSON again
    assert latest_run_dir(str(tmp_path / "out" / "run")) == record.path


def test_cached_rerun_is_a_new_record_with_the_same_recipes(tmp_path):
    graph = _graph()
    first = _run(tmp_path, graph)
    second = _run(tmp_path, graph)
    assert first.path != second.path
    a, b = _load(first), _load(second)
    assert b["nodes"]["src"]["status"] == "cached"
    assert b["nodes"]["src"]["recipe"] == a["nodes"]["src"]["recipe"]
    assert b["nodes"]["src"]["outputs"]["num"]["summary"] == "ClusterStats 1 entries"
    assert latest_run_dir(str(tmp_path / "out" / "run")) == second.path


def test_failed_stage_is_recorded(tmp_path):
    with pytest.raises(FlowError):
        _run(tmp_path, _graph("rec_boom"))
    record_dir = latest_run_dir(str(tmp_path / "out" / "run"))
    data = json.loads((record_dir / "run.json").read_text())
    assert data["status"] == "failed"
    assert "boom in stage" in data["error"]
    assert data["nodes"]["mid"]["status"] == "failed"
    assert data["nodes"]["src"]["status"] == "ran"


def test_interrupted_run_is_recorded(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, _graph("rec_stop"))
    record_dir = latest_run_dir(str(tmp_path / "out" / "run"))
    data = json.loads((record_dir / "run.json").read_text())
    assert data["status"] == "interrupted"
    assert data["nodes"]["mid"]["status"] == "interrupted"


def test_validation_failure_is_recorded(tmp_path):
    graph = Graph(name="rec", nodes=(Node("src", "rec_source", {"value": 1}),))
    with pytest.raises(FlowError, match="missing param"):
        _run(tmp_path, graph)
    record_dir = latest_run_dir(str(tmp_path / "out" / "run"))
    data = json.loads((record_dir / "run.json").read_text())
    assert data["status"] == "failed"
    assert "missing param" in data["error"]


def test_recorded_flow_reruns_to_the_same_recipes(tmp_path):
    graph = _graph()
    record = _run(tmp_path, graph)
    rerun = Graph.from_json(json.loads((record.path / "flow.json").read_text()))
    assert [i for i in validate(rerun) if i.level == "error"] == []
    data = _load(record)
    tokens = {"{input}": data["tokens"]["input"], "{out}": data["tokens"]["out"],
              "{work}": data["tokens"]["work"],
              "{flow}": json.dumps(rerun.to_json())}
    recipes = plan_recipes(rerun, tokens)
    assert recipes["src"] == data["nodes"]["src"]["recipe"]


def test_same_second_runs_get_distinct_directories(tmp_path):
    graph = _graph()
    dirs = {_run(tmp_path, graph).path for _ in range(3)}
    assert len(dirs) == 3


def test_cli_run_writes_a_record_with_overrides(tmp_path, capsys):
    from karak.flow.__main__ import main

    flow = tmp_path / "flow.json"
    flow.write_text(json.dumps(_graph().to_json()))
    out = tmp_path / "out" / "run"
    assert main(["run", str(flow), "--out", str(out), "--input", "/data/in",
                 "--set", "src.value=3"]) == 0
    data = json.loads((latest_run_dir(str(out)) / "run.json").read_text())
    assert data["status"] == "ok"
    assert data["overrides"] == {"src.value": 3}
    assert data["nodes"]["src"]["params"]["value"] == 3
    assert data["flow"]["source"] == str(flow)
    recorded = json.loads((latest_run_dir(str(out)) / "flow.json").read_text())
    assert recorded["nodes"][0]["params"]["value"] == 3
