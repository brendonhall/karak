"""Tests for builtin flows and param overrides."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from karak.flow.builtins import builtin_flow, override_params
from karak.flow.graph import Graph
from karak.flow.validate import validate


@pytest.mark.parametrize("name", ["global", "tiled", "tiled-rare"])
def test_builtin_flows_validate_clean(name):
    graph = builtin_flow(name)
    errors = [i for i in validate(graph) if i.level == "error"]
    assert errors == []


def test_unknown_builtin_raises():
    with pytest.raises(KeyError):
        builtin_flow("nope")


@pytest.mark.parametrize("name", ["global", "tiled", "tiled-rare"])
def test_shipped_json_matches_builtin(name):
    graph = builtin_flow(name)
    shipped = json.loads(
        resources.files("karak.flow").joinpath(f"flows/{name}.json").read_text()
    )
    assert Graph.from_json(shipped) == graph


def test_tiled_rare_contains_rare_phase_node():
    types = {n.type for n in builtin_flow("tiled-rare").nodes}
    assert "rare_phase" in types
    assert "hdbscan_tiled" in types


def test_global_has_no_tiled_nodes():
    types = {n.type for n in builtin_flow("global").nodes}
    assert "hdbscan_tiled" not in types
    assert "rare_phase" not in types
    assert "fingerprints" in types
    assert "export_h5" in types


def test_override_params():
    graph = builtin_flow("global")
    modified = override_params(
        graph, {"hdb.min_cluster_size": 123, "src.downsample_factor": 1}
    )
    assert modified.node("hdb").params["min_cluster_size"] == 123
    assert modified.node("src").params["downsample_factor"] == 1
    # original untouched
    assert graph.node("hdb").params.get("min_cluster_size") != 123


def test_override_unknown_node_raises():
    with pytest.raises(KeyError):
        override_params(builtin_flow("global"), {"ghost.param": 1})


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


def test_apply_device_sets_denoise_device():
    from karak.flow.builtins import apply_device, builtin_flow

    graph = builtin_flow("global")
    modified = apply_device(graph, "cuda")
    # denoise stage now declares device parameter
    assert modified.node("dn").params["device"] == "cuda"
    # other nodes should not have device param
    assert "device" not in modified.node("src").params
