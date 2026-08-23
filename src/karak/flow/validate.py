"""Structural validation of a flow graph.

``validate(graph)`` is a pure function returning a list of Issues. It is the
headless gate before execution and the GUI's on-canvas error source — same
rules, two front-ends.
"""

from __future__ import annotations

from dataclasses import dataclass

from karak.flow.graph import Graph
from karak.stages import registry


@dataclass(frozen=True)
class Issue:
    level: str          # "error" | "warning"
    where: str          # node id or edge id
    message: str


def validate(graph: Graph) -> list[Issue]:
    issues: list[Issue] = []

    # Duplicate node ids
    seen: set[str] = set()
    for node in graph.nodes:
        if node.id in seen:
            issues.append(Issue("error", node.id, "duplicate node id"))
        seen.add(node.id)

    # Unknown stage types + param coercion
    stage_types: dict[str, type] = {}
    for node in graph.nodes:
        try:
            stage_types[node.id] = registry.get(node.type)
        except KeyError:
            issues.append(
                Issue("error", node.id, f"unknown stage type {node.type!r}")
            )
            continue
        try:
            stage_types[node.id].coerce_params(node.params)
        except ValueError as exc:
            issues.append(Issue("error", node.id, str(exc)))

    node_ids = {n.id for n in graph.nodes}

    # Edge endpoint checks
    for edge in graph.edges:
        for endpoint, direction in ((edge.src, "from"), (edge.dst, "to")):
            if endpoint.node not in node_ids:
                issues.append(Issue(
                    "error", edge.id,
                    f"{direction} references unknown node {endpoint.node!r}",
                ))
        src_cls = stage_types.get(edge.src.node)
        dst_cls = stage_types.get(edge.dst.node)
        src_port = dst_port = None
        if src_cls is not None:
            src_port = next(
                (p for p in src_cls.OUTPUTS if p.name == edge.src.port), None
            )
            if src_port is None:
                issues.append(Issue(
                    "error", edge.id,
                    f"{edge.src.node!r} has no output port {edge.src.port!r}",
                ))
        if dst_cls is not None:
            dst_port = next(
                (p for p in dst_cls.INPUTS if p.name == edge.dst.port), None
            )
            if dst_port is None:
                issues.append(Issue(
                    "error", edge.id,
                    f"{edge.dst.node!r} has no input port {edge.dst.port!r}",
                ))
        # Port-type (space/state) mismatch
        if (
            src_port is not None and dst_port is not None
            and src_port.space is not None and dst_port.space is not None
            and src_port.space != dst_port.space
        ):
            issues.append(Issue(
                "error", edge.id,
                f"port type mismatch: {edge.src.node}.{edge.src.port} "
                f"carries {src_port.space!r} but {edge.dst.node}."
                f"{edge.dst.port} expects {dst_port.space!r}",
            ))

    # Missing / duplicate input connections
    for node in graph.nodes:
        cls = stage_types.get(node.id)
        if cls is None:
            continue
        incoming: dict[str, int] = {}
        for edge in graph.in_edges(node.id):
            incoming[edge.dst.port] = incoming.get(edge.dst.port, 0) + 1
        for port in cls.INPUTS:
            count = incoming.get(port.name, 0)
            if count == 0 and port.required:
                issues.append(Issue(
                    "error", node.id,
                    f"required input {port.name!r} is not connected",
                ))
            elif count > 1:
                issues.append(Issue(
                    "error", node.id,
                    f"input {port.name!r} has multiple connections",
                ))

    # Unconsumed outputs (warning)
    consumed = {(e.src.node, e.src.port) for e in graph.edges}
    for node in graph.nodes:
        cls = stage_types.get(node.id)
        if cls is None:
            continue
        for port in cls.OUTPUTS:
            if (node.id, port.name) not in consumed:
                issues.append(Issue(
                    "warning", node.id,
                    f"output {port.name!r} is never consumed",
                ))

    # Cycles (Kahn)
    indegree = {n.id: 0 for n in graph.nodes}
    successors: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        if edge.src.node in indegree and edge.dst.node in indegree:
            indegree[edge.dst.node] += 1
            successors[edge.src.node].append(edge.dst.node)
    queue = [nid for nid, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited < len(indegree):
        remaining = sorted(nid for nid, deg in indegree.items() if deg > 0)
        issues.append(Issue(
            "error", ",".join(remaining),
            f"cycle detected involving nodes {remaining}",
        ))

    return issues
