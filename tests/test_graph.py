"""Tests for the flow graph model and its JSON round-trip."""

from __future__ import annotations

import json

from karak.flow.graph import Edge, Endpoint, Graph, Node


def _graph() -> Graph:
    nodes = (
        Node("src", "load_elements", {"input_dir": "{input}"}),
        Node("msk", "mask", {"min_object_size": 50}, ui={"x": 220, "y": 80}),
    )
    edges = (
        Edge("e1", Endpoint("src", "cube"), Endpoint("msk", "cube")),
    )
    return Graph(nodes=nodes, edges=edges, name="test", version=1)


def test_json_roundtrip_preserves_everything():
    graph = _graph()
    data = graph.to_json()
    json.dumps(data)  # pure JSON
    back = Graph.from_json(data)
    assert back == graph
    assert back.node("msk").ui == {"x": 220, "y": 80}


def test_json_format_shape():
    data = _graph().to_json()
    assert data["version"] == 1
    assert data["name"] == "test"
    assert data["nodes"][0] == {
        "id": "src", "type": "load_elements",
        "params": {"input_dir": "{input}"},
    }
    assert data["edges"][0] == {
        "id": "e1",
        "from": {"node": "src", "port": "cube"},
        "to": {"node": "msk", "port": "cube"},
    }


def test_node_lookup_and_in_edges():
    graph = _graph()
    assert graph.node("src").type == "load_elements"
    incoming = graph.in_edges("msk")
    assert len(incoming) == 1
    assert incoming[0].src == Endpoint("src", "cube")
    assert graph.in_edges("src") == ()
