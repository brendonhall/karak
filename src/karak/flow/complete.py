"""Complete flows: every node spells out every one of its stage's params.

A flow run never takes a value from code. Stage ``Param`` defaults are
templates: they fill a new or older flow here, explicitly, and the result
is written back to the flow JSON that is run and recorded.
"""

from __future__ import annotations

import dataclasses

from karak.flow.graph import Graph, Node
from karak.stages import registry

FLOW_VERSION = 2
COMPLETE_HINT = "fill them in with `karak flow complete FLOW.json`"


def _stage(node: Node):
    try:
        return registry.get(node.type)
    except KeyError:
        return None


def missing_params(graph: Graph) -> dict[str, list[str]]:
    """``{node_id: [param, ...]}`` for nodes of known type that lack params."""
    missing = {}
    for node in graph.nodes:
        cls = _stage(node)
        if cls is None:
            continue
        names = [p.name for p in cls.PARAMS if p.name not in node.params]
        if names:
            missing[node.id] = names
    return missing


def complete_graph(graph: Graph) -> tuple[Graph, dict[str, list[str]]]:
    """Fill every missing param from its stage template.

    Returns the completed graph (format version 2, params in declared
    order) and ``{node_id: [added param, ...]}``. Values already given are
    kept as written; unknown stage types and undeclared params are left
    for ``validate`` to report.
    """
    nodes = []
    added: dict[str, list[str]] = {}
    for node in graph.nodes:
        cls = _stage(node)
        if cls is None:
            nodes.append(node)
            continue
        params = {}
        for p in cls.PARAMS:
            if p.name in node.params:
                params[p.name] = node.params[p.name]
            else:
                params[p.name] = p.default
                added.setdefault(node.id, []).append(p.name)
        params.update({k: v for k, v in node.params.items() if k not in params})
        nodes.append(dataclasses.replace(node, params=params))
    completed = dataclasses.replace(graph, nodes=tuple(nodes), version=FLOW_VERSION)
    return completed, added


def canonicalize(graph: Graph) -> Graph:
    """Coerce every param to its declared type, leaving tokens unresolved.

    Nodes whose params do not coerce (missing, unknown, out of bounds) are
    left as they are, so ``validate`` reports them.
    """
    nodes = []
    for node in graph.nodes:
        cls = _stage(node)
        try:
            params = cls.coerce_params(node.params, require_complete=True)
        except (AttributeError, ValueError):
            nodes.append(node)
            continue
        nodes.append(dataclasses.replace(node, params=params))
    return dataclasses.replace(graph, nodes=tuple(nodes))
