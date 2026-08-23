"""Tests for the legacy YAML -> flow graph shim and Param-default drift."""

from __future__ import annotations

import pytest

from karak.config import (
    ClusterConfig,
    DenoiseConfig,
    HDBSCANConfig,
    MaskConfig,
    PCAConfig,
    PipelineConfig,
    RarePhaseConfig,
    RefinementConfig,
    TiledConfig,
)
from karak.flow.builtins import flow_from_config
from karak.flow.validate import validate
from karak.stages import get


def _errors(graph):
    return [i for i in validate(graph) if i.level == "error"]


def test_default_config_maps_to_global_flow():
    cfg = PipelineConfig()
    graph = flow_from_config(cfg)
    types = {n.type for n in graph.nodes}
    assert "hdbscan_global" in types
    assert "hdbscan_tiled" not in types
    assert "rare_phase" not in types
    assert "refine" not in types
    assert _errors(graph) == []
    # params mapped through
    assert graph.node("src").params["input_dir"] == cfg.input_dir
    assert graph.node("src").params["downsample_factor"] == 2
    assert graph.node("hdb").params["min_cluster_size"] == 1000
    assert graph.node("knn").params["k"] == 5
    assert graph.node("exp").params["path"] == cfg.hdf5_output


def test_tiled_config_with_rare_and_refine():
    cfg = PipelineConfig(
        cluster=ClusterConfig(
            strategy="tiled",
            tiled=TiledConfig(tile_size=256),
            rare_phase=RarePhaseConfig(enabled=True, min_cluster_size=30),
            refinement=RefinementConfig(enabled=True, target_phase=1),
        )
    )
    graph = flow_from_config(cfg)
    types = {n.type for n in graph.nodes}
    assert {"hdbscan_tiled", "rare_phase", "refine"} <= types
    assert _errors(graph) == []
    assert graph.node("hdb").params["tile_size"] == 256
    assert graph.node("rare").params["min_cluster_size"] == 30
    assert graph.node("ref").params["target_phase"] == 1
    # cleaned-label consumers rewired through refine
    phase_map_edges = [
        e for e in graph.edges
        if e.dst.node == "qc_phase_map" and e.dst.port == "labels"
    ]
    assert phase_map_edges[0].src.node == "ref"


def test_exclude_elements_joined():
    cfg = PipelineConfig(exclude_elements=["Fe-L", "SEM"])
    graph = flow_from_config(cfg)
    assert graph.node("src").params["exclude_elements"] == "Fe-L,SEM"


# ---------------------------------------------------------------------------
# Param-default drift guard: stage Param defaults == Pydantic field defaults
# ---------------------------------------------------------------------------

_DRIFT_MAP = [
    # (stage id, param name, pydantic default)
    ("mask", "min_object_size", MaskConfig().min_object_size),
    ("denoise", "method", DenoiseConfig().method),
    ("denoise", "sigma_spatial", DenoiseConfig().sigma_spatial),
    ("denoise", "niter", DenoiseConfig().niter),
    ("denoise", "kappa", DenoiseConfig().kappa),
    ("denoise", "gamma", DenoiseConfig().gamma),
    ("denoise", "option", DenoiseConfig().option),
    ("pca", "random_state", PCAConfig().random_state),
    ("hdbscan_global", "min_cluster_size", HDBSCANConfig().min_cluster_size),
    ("hdbscan_global", "random_state", HDBSCANConfig().random_state),
    ("hdbscan_tiled", "tile_size", TiledConfig().tile_size),
    ("hdbscan_tiled", "merge_threshold", TiledConfig().merge_threshold),
    ("hdbscan_tiled", "min_clusters_per_tile",
     TiledConfig().min_clusters_per_tile),
    ("rare_phase", "min_cluster_size", RarePhaseConfig().min_cluster_size),
    ("rare_phase", "subsample_n", RarePhaseConfig().subsample_n),
    ("noise_assign", "k", HDBSCANConfig().noise_reassign_k),
    ("refine", "target_phase", RefinementConfig().target_phase),
]


@pytest.mark.parametrize("stage_id,param,expected", _DRIFT_MAP)
def test_param_defaults_match_pydantic(stage_id, param, expected):
    declared = {p.name: p for p in get(stage_id).PARAMS}
    assert declared[param].default == expected
