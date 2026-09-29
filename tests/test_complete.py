"""Completing flows: every param spelled out, filled from stage templates."""

from __future__ import annotations

from karak.flow.complete import (
    FLOW_VERSION,
    canonicalize,
    complete_graph,
    missing_params,
)
from karak.flow.graph import Edge, Endpoint, Graph, Node
from karak.stages.load import LoadElementsStage


def _sparse():
    return Graph(
        name="sparse",
        version=1,
        nodes=(
            Node("src", "load_elements",
                 {"input_dir": "{input}", "downsample_factor": 4},
                 ui={"x": 10, "y": 20}),
            Node("msk", "mask"),
        ),
        edges=(
            Edge("e1", Endpoint("src", "cube"), Endpoint("msk", "cube")),
            Edge("e2", Endpoint("src", "bse"), Endpoint("msk", "bse")),
        ),
    )


def test_missing_params_lists_what_each_node_lacks():
    missing = missing_params(_sparse())
    assert "downsample_factor" not in missing["src"]
    assert "colormap" in missing["src"]
    assert missing["msk"] == ["min_object_size", "valid_mask_path"]


def test_complete_graph_fills_template_values_and_reports_them():
    graph, added = complete_graph(_sparse())
    src = graph.node("src")
    assert src.params["downsample_factor"] == 4          # kept as written
    assert src.params["colormap"] == LoadElementsStage.template()["colormap"]
    assert list(src.params) == list(LoadElementsStage.template())  # declared order
    assert added["msk"] == ["min_object_size", "valid_mask_path"]
    assert graph.version == FLOW_VERSION
    assert src.ui == {"x": 10, "y": 20}
    assert missing_params(graph) == {}


def test_completing_a_complete_graph_adds_nothing():
    graph, _ = complete_graph(_sparse())
    again, added = complete_graph(graph)
    assert added == {}
    assert again == graph


def test_complete_graph_leaves_unknown_types_and_params_for_validate():
    graph = Graph(nodes=(Node("a", "no_such_stage", {"x": 1}),
                         Node("b", "mask", {"bogus": 1})))
    completed, _ = complete_graph(graph)
    assert completed.node("a").params == {"x": 1}
    assert completed.node("b").params["bogus"] == 1


def test_canonicalize_coerces_values_without_resolving_tokens():
    graph, _ = complete_graph(_sparse())
    src = graph.node("src")
    edited = Graph(
        nodes=(Node("src", "load_elements",
                    {**src.params, "downsample_factor": "3"}),)
        + graph.nodes[1:],
        edges=graph.edges, name=graph.name, version=graph.version,
    )
    canon = canonicalize(edited)
    assert canon.node("src").params["downsample_factor"] == 3
    assert canon.node("src").params["input_dir"] == "{input}"
