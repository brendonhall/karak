"""Builtin flows: complete flow JSON files shipped in ``karak/flow/flows/``.

The files are the source of truth: every node lists every param, so a
builtin run takes no value from code. ``karak flow init`` copies one as the
starting point for a custom pipeline.
"""

from __future__ import annotations

import json
from importlib import resources

from karak.flow.graph import Graph, Node


def _flows_dir():
    return resources.files("karak.flow").joinpath("flows")


def builtin_names() -> list[str]:
    return sorted(
        f.name[: -len(".json")] for f in _flows_dir().iterdir()
        if f.name.endswith(".json")
    )


def builtin_flow(name: str) -> Graph:
    """Load a shipped flow by name; ``KeyError`` for an unknown name."""
    if name not in builtin_names():
        raise KeyError(name)
    return Graph.from_json(json.loads(_flows_dir().joinpath(f"{name}.json").read_text()))


def override_params(graph: Graph, overrides: dict) -> Graph:
    """Return a new Graph with ``{"node.param": value}`` overrides applied.

    ``KeyError`` for an unknown node, ``ValueError`` for a param the node's
    stage does not declare.
    """
    from karak.stages import registry

    updates: dict[str, dict] = {}
    for spec, value in overrides.items():
        node_id, param = spec.split(".", 1)
        node = graph.node(node_id)  # raises KeyError for unknown nodes
        declared = [p.name for p in registry.get(node.type).PARAMS]
        if param not in declared:
            raise ValueError(
                f"{spec}: stage {node.type!r} has no param {param!r}; "
                f"declared: {declared}"
            )
        updates.setdefault(node_id, {})[param] = value
    nodes = tuple(
        node if node.id not in updates
        else Node(
            id=node.id,
            type=node.type,
            params={**node.params, **updates[node.id]},
            ui=node.ui,
        )
        for node in graph.nodes
    )
    return Graph(
        nodes=nodes, edges=graph.edges, name=graph.name, version=graph.version
    )


def apply_device(graph: Graph, device: str) -> Graph:
    """Set ``device`` on every node whose stage declares that param.

    Nodes without a declared ``device`` param are left untouched, so the
    override is safe on any flow.
    """
    from karak.stages import registry

    overrides = {
        f"{node.id}.device": device
        for node in graph.nodes
        if any(p.name == "device" for p in registry.get(node.type).PARAMS)
    }
    return override_params(graph, overrides) if overrides else graph
