"""One test per validation rule."""

from __future__ import annotations

import pytest

from karak.flow.graph import Edge, Endpoint, Graph, Node
from karak.flow.validate import Issue, validate


def _edge(eid, src_node, src_port, dst_node, dst_port):
    return Edge(eid, Endpoint(src_node, src_port), Endpoint(dst_node, dst_port))


def _load_mask_graph(**mask_params):
    return Graph(
        nodes=(
            Node("src", "load_elements", {"input_dir": "/data"}),
            Node("msk", "mask", dict(mask_params)),
        ),
        edges=(_edge("e1", "src", "cube", "msk", "cube"),),
        name="ok",
    )


def _errors(graph):
    return [i for i in validate(graph) if i.level == "error"]


def _warnings(graph):
    return [i for i in validate(graph) if i.level == "warning"]


def test_valid_graph_has_no_errors():
    assert _errors(_load_mask_graph()) == []


def test_duplicate_node_ids():
    graph = Graph(nodes=(
        Node("a", "load_elements"), Node("a", "mask"),
    ))
    assert any("duplicate" in i.message for i in _errors(graph))


def test_unknown_stage_type():
    graph = Graph(nodes=(Node("a", "no_such_stage"),))
    assert any("unknown stage" in i.message for i in _errors(graph))


def test_param_coercion_error():
    graph = _load_mask_graph(min_object_size="not_a_number")
    assert any(i.where == "msk" for i in _errors(graph))


def test_unknown_param_name():
    graph = _load_mask_graph(nope=1)
    assert any(i.where == "msk" for i in _errors(graph))


def test_dangling_edge():
    graph = Graph(
        nodes=(Node("src", "load_elements"),),
        edges=(_edge("e1", "src", "cube", "ghost", "cube"),),
    )
    assert any(i.where == "e1" for i in _errors(graph))


def test_edge_references_unknown_port():
    graph = Graph(
        nodes=(
            Node("src", "load_elements"),
            Node("msk", "mask"),
        ),
        edges=(_edge("e1", "src", "nope", "msk", "cube"),),
    )
    assert any(i.where == "e1" for i in _errors(graph))


def test_missing_required_input():
    graph = Graph(nodes=(Node("msk", "mask"),))
    assert any(
        "cube" in i.message and i.where == "msk" for i in _errors(graph)
    )


def test_duplicate_input_connection():
    graph = Graph(
        nodes=(
            Node("a", "load_elements"),
            Node("b", "load_elements"),
            Node("msk", "mask"),
        ),
        edges=(
            _edge("e1", "a", "cube", "msk", "cube"),
            _edge("e2", "b", "cube", "msk", "cube"),
        ),
    )
    assert any(
        "multiple" in i.message and i.where == "msk" for i in _errors(graph)
    )


def test_space_mismatch():
    # load produces RAW; normalize expects DENOISED
    graph = Graph(
        nodes=(
            Node("src", "load_elements"),
            Node("msk", "mask"),
            Node("nrm", "normalize"),
        ),
        edges=(
            _edge("e1", "src", "cube", "msk", "cube"),
            _edge("e2", "src", "cube", "nrm", "cube"),
            _edge("e3", "msk", "masks", "nrm", "masks"),
        ),
    )
    assert any(i.where == "e2" for i in _errors(graph))


def test_unconsumed_output_is_warning():
    graph = _load_mask_graph()  # bse and masks never consumed
    warnings = _warnings(graph)
    assert any("bse" in i.message for i in warnings)
    assert _errors(graph) == []


def test_cycle_detected():
    graph = Graph(
        nodes=(
            Node("a", "denoise"),
            Node("b", "denoise"),
        ),
        edges=(
            _edge("e1", "a", "cube", "b", "cube"),
            _edge("e2", "b", "cube", "a", "cube"),
        ),
    )
    assert any("cycle" in i.message.lower() for i in _errors(graph))


def test_issue_shape():
    issue = Issue("error", "node1", "boom")
    assert (issue.level, issue.where, issue.message) == ("error", "node1", "boom")
