"""Headless flow executor: validate -> topo-sort -> run with caching.

Memory model: every producing node's outputs are written through to the
cache, then held in RAM only while downstream consumers remain (refcount).
Payloads whose arrays exceed ``spill_threshold`` bytes are dropped from RAM
immediately after caching and reloaded per consumer — the generic form of
the old runner's pop-then-reread-HDF5 trick.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from pathlib import Path

import numpy as np

from karak.flow.cache import (
    has_payload,
    load_payload,
    recipe_hash,
    store_payload,
)
from karak.flow.events import NullReporter
from karak.flow.graph import Graph
from karak.flow.validate import validate
from karak.stages import registry


class FlowError(Exception):
    """Raised when a flow fails validation or execution."""


def _payload_nbytes(payload) -> int:
    total = 0
    for field in dataclasses.fields(payload):
        value = getattr(payload, field.name)
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


class PayloadStore:
    """Refcounted in-RAM payload store with optional spill-to-cache.

    ``consumers`` maps (node, port) -> number of downstream consumers.
    ``get`` decrements the count and evicts the payload once it reaches 0.
    Payloads larger than ``spill_threshold`` bytes are not held in RAM at
    all; ``reload`` fetches them from the cache per consumer.
    """

    def __init__(self, consumers: dict, reload=None,
                 spill_threshold: int | None = None):
        self._remaining = dict(consumers)
        self._in_ram: dict = {}
        self._spilled: set = set()
        self._reload = reload
        self._spill_threshold = spill_threshold

    def put(self, node: str, port: str, payload) -> None:
        key = (node, port)
        if self._remaining.get(key, 0) <= 0:
            return  # unconsumed output — drop immediately
        if (
            self._spill_threshold is not None
            and self._reload is not None
            and _payload_nbytes(payload) > self._spill_threshold
        ):
            self._spilled.add(key)
            return
        self._in_ram[key] = payload

    def get(self, node: str, port: str):
        key = (node, port)
        if key in self._spilled:
            payload = self._reload(node, port)
        else:
            payload = self._in_ram[key]
        self.release(node, port)
        return payload

    def release(self, node: str, port: str) -> None:
        """Decrement the consumer count without fetching (cache-hit path)."""
        key = (node, port)
        remaining = self._remaining.get(key, 0) - 1
        self._remaining[key] = remaining
        if remaining <= 0:
            self._in_ram.pop(key, None)
            self._spilled.discard(key)


def _topo_order(graph: Graph) -> list[str]:
    indegree = {n.id: len(graph.in_edges(n.id)) for n in graph.nodes}
    successors: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for edge in graph.edges:
        successors[edge.src.node].append(edge.dst.node)
    order: list[str] = []
    ready = [n.id for n in graph.nodes if indegree[n.id] == 0]
    while ready:
        current = ready.pop(0)
        order.append(current)
        for nxt in successors[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return order


def _resolve_tokens(params: dict, tokens: dict) -> dict:
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str):
            for token, replacement in tokens.items():
                value = value.replace(token, str(replacement))
        resolved[key] = value
    return resolved


def run(
    graph: Graph,
    *,
    input_path: str = "",
    out_base: str = "",
    work_dir: str = "work",
    cache: bool = True,
    reporter=None,
    skip_types: frozenset | set = frozenset(),
    spill_threshold: int = 256 * 1024 * 1024,
) -> dict:
    """Execute a flow. Returns {node_id: {"cached": bool, "seconds": float}}."""
    errors = [i for i in validate(graph) if i.level == "error"]
    if errors:
        detail = "; ".join(f"[{i.where}] {i.message}" for i in errors)
        raise FlowError(f"flow {graph.name!r} failed validation: {detail}")

    reporter = reporter or NullReporter()
    cache_dir = Path(work_dir) / "cache"
    # {flow} must come last so tokens inside the embedded flow JSON stay
    # as authored (provenance shows the reusable definition, not one run).
    import json as _json

    tokens = {
        "{input}": input_path,
        "{out}": out_base,
        "{work}": work_dir,
        "{flow}": _json.dumps(graph.to_json()),
    }

    skipped = {
        n.id for n in graph.nodes
        if n.type in skip_types and not registry.get(n.type).OUTPUTS
    }
    consumers = Counter(
        (e.src.node, e.src.port)
        for e in graph.edges
        if e.dst.node not in skipped
    )
    hashes: dict = {}          # (node, port) -> recipe hash

    def _reload(node: str, port: str):
        return load_payload(hashes[(node, port)], port, cache_dir)

    store = PayloadStore(
        consumers,
        reload=_reload,
        spill_threshold=spill_threshold if cache else None,
    )
    summary: dict = {}

    for node_id in _topo_order(graph):
        node = graph.node(node_id)
        if node_id in skipped:
            summary[node_id] = {"cached": False, "seconds": 0.0,
                                "skipped": True}
            continue
        cls = registry.get(node.type)
        # Coerce first so Param defaults (which may contain tokens, e.g. a
        # sink's "{out}") are present, then resolve tokens on the result.
        params = _resolve_tokens(cls.coerce_params(node.params), tokens)
        upstream = {
            f"{e.dst.port}": hashes[(e.src.node, e.src.port)]
            for e in graph.in_edges(node_id)
        }
        source_sig = cls.source_signature(params)
        node_hash = recipe_hash(node.type, params, upstream, source_sig)
        for port in cls.OUTPUTS:
            hashes[(node_id, port.name)] = node_hash

        is_sink = not cls.OUTPUTS
        started = time.monotonic()
        cached_hit = (
            cache
            and not is_sink
            and all(
                has_payload(node_hash, port.name, cache_dir)
                for port in cls.OUTPUTS
            )
        )

        if cached_hit:
            # Outputs come from the cache; inputs are not consumed, but the
            # upstream refcounts still must fall so payloads are evicted.
            for edge in graph.in_edges(node_id):
                store.release(edge.src.node, edge.src.port)
            outputs = {
                port.name: load_payload(node_hash, port.name, cache_dir)
                for port in cls.OUTPUTS
                if consumers.get((node_id, port.name), 0) > 0
            }
        else:
            inputs = {
                e.dst.port: store.get(e.src.node, e.src.port)
                for e in graph.in_edges(node_id)
            }
            stage = cls()
            stage.reporter = reporter
            reporter.node_started(node_id, cls.label or cls.id)
            try:
                outputs = stage.run(inputs, params)
            except Exception as exc:
                raise FlowError(f"node {node_id!r} ({node.type}): {exc}") from exc
            if cache and not is_sink:
                for port_name, payload in outputs.items():
                    store_payload(node_hash, port_name, payload, cache_dir)

        for port_name, payload in outputs.items():
            store.put(node_id, port_name, payload)

        elapsed = time.monotonic() - started
        reporter.node_finished(node_id, elapsed, cached_hit)
        summary[node_id] = {"cached": cached_hit, "seconds": elapsed}

    return summary
