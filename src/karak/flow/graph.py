"""The flow graph model: nodes, edges, and JSON (de)serialization.

A flow is a DAG of stage instances. ``node.type`` references a registered
stage id; ``node.params`` supplies values for that stage's declared Params;
``node.ui`` is GUI-only (position) and ignored by the executor. The JSON
form is the contract every front-end (CLI, cache, GUI) works against.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Endpoint:
    node: str
    port: str


@dataclass(frozen=True)
class Edge:
    id: str
    src: Endpoint     # an OUTPUT port
    dst: Endpoint     # an INPUT port


@dataclass(frozen=True)
class Node:
    id: str
    type: str
    params: dict = field(default_factory=dict)
    ui: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Graph:
    nodes: tuple = ()
    edges: tuple = ()
    name: str = ""
    version: int = 1

    def node(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def in_edges(self, node_id: str) -> tuple:
        return tuple(e for e in self.edges if e.dst.node == node_id)

    def to_json(self) -> dict:
        def node_json(node: Node) -> dict:
            data = {"id": node.id, "type": node.type, "params": dict(node.params)}
            if node.ui:
                data["ui"] = dict(node.ui)
            return data

        return {
            "version": self.version,
            "name": self.name,
            "nodes": [node_json(n) for n in self.nodes],
            "edges": [
                {
                    "id": e.id,
                    "from": {"node": e.src.node, "port": e.src.port},
                    "to": {"node": e.dst.node, "port": e.dst.port},
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Graph":
        nodes = tuple(
            Node(
                id=n["id"],
                type=n["type"],
                params=n.get("params", {}),
                ui=n.get("ui", {}),
            )
            for n in data.get("nodes", [])
        )
        edges = tuple(
            Edge(
                id=e["id"],
                src=Endpoint(e["from"]["node"], e["from"]["port"]),
                dst=Endpoint(e["to"]["node"], e["to"]["port"]),
            )
            for e in data.get("edges", [])
        )
        return cls(
            nodes=nodes,
            edges=edges,
            name=data.get("name", ""),
            version=data.get("version", 1),
        )
