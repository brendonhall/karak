"""Tests for the karak bench subcommand."""

from __future__ import annotations

import json

import pytest

from karak.cli.bench import BenchConfig, bench_main, parse_config, run_bench
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


def test_compare_nonexistent_file_exits(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"meta": {}, "configs": []}))
    missing = tmp_path / "missing.json"
    with pytest.raises(SystemExit):
        bench_main(["--compare", str(missing), str(good)])


def test_compare_malformed_json_exits(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"meta": {}, "configs": []}))
    with pytest.raises(SystemExit):
        bench_main(["--compare", str(bad), str(good)])
