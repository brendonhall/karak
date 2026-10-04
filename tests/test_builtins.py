"""Tests for builtin flows and param overrides."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from karak.flow.builtins import builtin_flow, override_params
from karak.flow.graph import Graph
from karak.flow.validate import validate


@pytest.mark.parametrize("name", ["global", "tiled", "tiled-rare", "stepwise"])
def test_builtin_flows_validate_clean(name):
    graph = builtin_flow(name)
    errors = [i for i in validate(graph) if i.level == "error"]
    assert errors == []


def test_unknown_builtin_raises():
    with pytest.raises(KeyError):
        builtin_flow("nope")


@pytest.mark.parametrize("name", ["global", "tiled", "tiled-rare", "stepwise"])
def test_shipped_flows_are_complete_version_2(name):
    from karak.flow.complete import FLOW_VERSION, complete_graph

    shipped = json.loads(
        resources.files("karak.flow").joinpath(f"flows/{name}.json").read_text()
    )
    graph = Graph.from_json(shipped)
    assert graph.version == FLOW_VERSION
    assert complete_graph(graph)[1] == {}
    assert builtin_flow(name) == graph


def test_builtin_names_are_the_shipped_files():
    from karak.flow.builtins import builtin_names

    shipped = sorted(
        f.name[:-5] for f in resources.files("karak.flow").joinpath("flows").iterdir()
        if f.name.endswith(".json")
    )
    assert builtin_names() == shipped


def test_override_rejects_an_undeclared_param():
    with pytest.raises(ValueError, match="has no param 'bogus'"):
        override_params(builtin_flow("stepwise"), {"src.bogus": 1})


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


def test_apply_device_no_declaring_nodes_is_identity():
    from karak.flow.builtins import apply_device
    from karak.flow.graph import Graph, Node

    # Synthetic graph with only load_elements, which declares no device param
    graph = Graph(
        name="no_device",
        nodes=(Node("src", "load_elements"),),
        edges=(),
    )
    result = apply_device(graph, "cuda")
    # When no nodes declare device, should return the same graph object
    assert result is graph


def test_stepwise_runs_load_through_noise_assign_then_stats_and_fingerprints():
    from karak.stages.denoise import DenoiseStage
    from karak.stages.fingerprints import FingerprintsStage
    from karak.stages.load import LoadElementsStage
    from karak.stages.mask import MaskStage
    from karak.stages.noise import NoiseAssignStage
    from karak.stages.normalize import NormalizeStage
    from karak.stages.cluster import HdbscanGlobalStage
    from karak.stages.pca import PCAStage
    from karak.stages.stats import ClusterStatsStage

    graph = builtin_flow("stepwise")
    assert [(n.id, n.type) for n in graph.nodes] == [
        ("src", "load_elements"), ("msk", "mask"), ("dn", "denoise"),
        ("nrm", "normalize"), ("pca", "pca"), ("hdb", "hdbscan_global"),
        ("knn", "noise_assign"), ("stats", "cluster_stats"),
        ("fp", "fingerprints"),
    ]
    assert graph.node("src").params == LoadElementsStage.template()
    assert graph.node("msk").params == MaskStage.template()
    assert graph.node("dn").params == DenoiseStage.template()
    assert graph.node("nrm").params == NormalizeStage.template()
    assert graph.node("pca").params == PCAStage.template()
    assert graph.node("hdb").params == HdbscanGlobalStage.template()
    assert graph.node("knn").params == NoiseAssignStage.template()
    assert graph.node("stats").params == ClusterStatsStage.template()
    assert graph.node("fp").params == FingerprintsStage.template()
    assert [(e.src.node, e.src.port, e.dst.node, e.dst.port)
            for e in graph.edges] == [
        ("src", "cube", "msk", "cube"),
        ("src", "cube", "dn", "cube"),
        ("msk", "masks", "dn", "masks"),
        ("dn", "cube", "nrm", "cube"),
        ("msk", "masks", "nrm", "masks"),
        ("nrm", "cube", "pca", "cube"),
        ("msk", "masks", "pca", "masks"),
        ("pca", "features", "hdb", "features"),
        ("hdb", "labels", "knn", "labels"),
        ("pca", "features", "knn", "features"),
        ("knn", "labels", "stats", "labels"),
        ("knn", "labels", "fp", "labels"),
        ("dn", "cube", "fp", "cube"),
    ]


def test_apply_overrides_applies_the_device_before_the_set_values():
    from karak.flow.builtins import apply_overrides

    graph = apply_overrides(builtin_flow("global"), device="cuda",
                            overrides={"hdb.device": "cpu"})
    assert graph.node("hdb").params["device"] == "cpu"
    assert graph.node("dn").params["device"] == "cuda"


def test_apply_overrides_without_overrides_is_identity():
    from karak.flow.builtins import apply_overrides

    graph = builtin_flow("global")
    assert apply_overrides(graph) is graph
