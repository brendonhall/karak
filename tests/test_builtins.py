"""Tests for builtin flows and param overrides."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from karak.flow.builtins import builtin_flow, override_params
from karak.flow.graph import Graph
from karak.flow.validate import validate


@pytest.mark.parametrize("name", ["global", "tiled", "tiled-rare", "stepwise", "paper"])
def test_builtin_flows_validate_clean(name):
    graph = builtin_flow(name)
    errors = [i for i in validate(graph) if i.level == "error"]
    assert errors == []


def test_unknown_builtin_raises():
    with pytest.raises(KeyError):
        builtin_flow("nope")


@pytest.mark.parametrize("name", ["global", "tiled", "tiled-rare", "stepwise", "paper"])
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


def test_stepwise_is_global_without_its_output_steps():
    """stepwise is the dashboard flow: every global processing step with
    the same params and wiring, without the HDF5 export and QC figures.
    `karak run --builtin global` after a stepwise run reuses every cached
    step and runs only the outputs."""
    def is_output(node_type):
        return node_type == "export_h5" or node_type.startswith("qc_")

    glob, step = builtin_flow("global"), builtin_flow("stepwise")
    processing = {n.id: (n.type, n.params) for n in glob.nodes
                  if not is_output(n.type)}
    assert {n.id: (n.type, n.params) for n in step.nodes} == processing
    kept = set(processing)

    def wiring(graph):
        return {(e.src.node, e.src.port, e.dst.node, e.dst.port)
                for e in graph.edges if e.dst.node in kept}

    assert wiring(step) == wiring(glob)
    assert {n.type for n in glob.nodes if is_output(n.type)} >= {"export_h5"}


# The settings of the published NWA 4587 run (eds_pipeline.h5,
# clusters.attrs["cluster_config"] and the load stage of that file).
PAPER_SETTINGS = {
    "src": {"colormap": "cmap:jet", "header_trim_px": 100},
    "msk": {"valid_mask_path": "{input}/mask/Valid_mask.csv"},
    "nrm": {"accumulate": "float32"},
    "hdb": {"min_cluster_size": 100, "min_samples": 25, "tile_size": 1024,
            "merge_threshold": 0.88, "accumulate": "float32"},
    "rare": {"merge_threshold": 0.88, "accumulate": "float32"},
    "fp": {"accumulate": "float32"},
    # the QC tile grid must keep the tiles the clustering keeps (2 x 100)
    "qc_tiled": {"min_tile_pixels": 200},
}


PAPER_EXTRA_NODES = ["src_hires", "names", "oliv", "weath", "pyx", "phos",
                     "qc_named_phase_map"]


def test_paper_is_tiled_rare_plus_the_published_hand_steps():
    paper, base = builtin_flow("paper"), builtin_flow("tiled-rare")
    base_ids = [n.id for n in base.nodes]
    assert [n.id for n in paper.nodes if n.id not in PAPER_EXTRA_NODES] == base_ids
    for node in base.nodes:
        expected = {**node.params, **PAPER_SETTINGS.get(node.id, {})}
        assert paper.node(node.id).params == expected, node.id
    assert all(paper.node(n).params["device"] == "cpu"
               for n in ("dn", "nrm", "pca", "hdb", "rare", "knn"))
    types = {n.id: n.type for n in paper.nodes}
    assert types["src_hires"] == "load_elements" and types["names"] == "name_phases"
    assert types["oliv"] == "split_threshold" and types["weath"] == "split_gmm"
    assert types["pyx"] == "split_hires" and types["phos"] == "split_gmm"
    hires = paper.node("src_hires").params
    assert hires["downsample_factor"] == 1 and hires["include_elements"] == "Ca,Mg"
    assert hires["header_trim_px"] == paper.node("src").params["header_trim_px"]
    assert hires["colormap"] == paper.node("src").params["colormap"]

    def source(node_id, port):
        return next((e.src.node, e.src.port) for e in paper.edges
                    if e.dst.node == node_id and e.dst.port == port)

    chain = [("names", "knn"), ("oliv", "names"), ("weath", "oliv"),
             ("pyx", "weath"), ("phos", "pyx")]
    for node_id, upstream in chain:
        assert source(node_id, "labels") == (upstream, "labels")
    assert source("pyx", "cube_hires") == ("src_hires", "cube")
    for consumer in ("stats", "fp", "exp", "qc_phase_map"):
        assert source(consumer, "labels") == ("phos", "labels")
    assert source("qc_named_phase_map", "labels") == ("phos", "labels")
    assert source("qc_named_phase_map", "bse") == ("src", "bse")
    assert source("exp", "labels_hires") == ("pyx", "labels_hires")
    assert source("weath", "bse") == ("src", "bse")
    assert paper.node("oliv").params["rule"] == "Fe-K > 0.6 & Ca < 0.10"
    assert paper.node("weath").params["keep_parent"] is True
    assert paper.node("weath").params["features"] == "Ca/(Ca+Mg),BSE,Fe-K"
    assert paper.node("phos").params["keep_parent"] is False
    assert paper.node("phos").params["order_by"] == "Cl"
    assert paper.node("pyx").params["feature"] == "Ca/(Ca+Mg)"


@pytest.mark.parametrize("name", ["tiled", "tiled-rare", "paper"])
def test_qc_tile_grid_keeps_the_tiles_the_clustering_keeps(name):
    """qc_tiled rebuilds the tile grid with its own min_tile_pixels and
    matches tile results by position, so it must equal the clustering's
    effective threshold (min_tile_pixels, or 2 x min_cluster_size for 0)."""
    graph = builtin_flow(name)
    hdb = graph.node("hdb").params
    clustering = hdb["min_tile_pixels"] or 2 * hdb["min_cluster_size"]
    assert graph.node("qc_tiled").params["min_tile_pixels"] == clustering


def test_paper_qc_grid_numbers_a_sparse_tile_like_the_clustering():
    """Review regression: tiles of 500 and 3,000 mineral pixels."""
    import numpy as np

    from karak.clustering.tiling import compute_tile_grid

    graph = builtin_flow("paper")
    hdb = graph.node("hdb").params
    rows_a, cols_a = np.divmod(np.arange(500), 100)
    rows_b, cols_b = np.divmod(np.arange(3000), 100)
    idx = np.concatenate([np.stack([rows_a, cols_a], 1),
                          np.stack([rows_b, cols_b + 1024], 1)]).astype(np.int32)
    shape = (1024, 2048)
    clustering = compute_tile_grid(idx, shape, hdb["tile_size"],
                                   hdb["min_tile_pixels"] or 2 * hdb["min_cluster_size"])
    qc = compute_tile_grid(idx, shape, hdb["tile_size"],
                           graph.node("qc_tiled").params["min_tile_pixels"])
    assert [(t.tile_id, t.col_start) for t in qc] == \
        [(t.tile_id, t.col_start) for t in clustering] == [(0, 0), (1, 1024)]
