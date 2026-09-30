"""Headless flow executor: validate -> topo-sort -> run with caching.

Memory model: every producing node's outputs are handed to a background
``CacheWriter`` (the next node starts while the file is written), then held
in RAM only while downstream consumers remain (refcount), up to a total
``ram_budget`` in bytes. A payload that would push the held total over the
budget is spilled: it is not held, and each consumer reloads it from the
cache after its background write completes.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from pathlib import Path

import numpy as np

from karak.flow.cache import (
    CacheWriter,
    has_payload,
    load_payload,
    load_summary,
    payload_path,
    recipe_hash,
    sweep_stale_tmp,
)
from karak.flow.events import NullReporter, ParamValue, RunInfo
from karak.flow.graph import Graph
from karak.flow.validate import validate
from karak.provenance import karak_version
from karak.stages import registry


class FlowError(Exception):
    """Raised when a flow fails validation or execution."""


class CacheWriteError(FlowError):
    """A background cache write failed (for example, a full disk)."""


def _writer_call(method, *args) -> None:
    """Call a CacheWriter method; its error becomes a CacheWriteError."""
    try:
        method(*args)
    except Exception as exc:
        raise CacheWriteError(f"cache writer: {exc}") from exc


def _payload_nbytes(payload) -> int:
    total = 0
    for field in dataclasses.fields(payload):
        value = getattr(payload, field.name)
        if isinstance(value, np.ndarray):
            total += value.nbytes
    return total


class PayloadStore:
    """Refcounted in-RAM payload store with a total memory budget.

    ``consumers`` maps (node, port) -> number of downstream consumers.
    ``get`` decrements the count and evicts the payload once it reaches 0.
    A payload is spilled (not held; ``reload`` fetches it from the cache
    per consumer) when holding it would push the total above
    ``ram_budget`` bytes. ``ram_budget=None`` never spills. ``on_spill``
    is called as ``on_spill(node, port, nbytes, budget)`` once per spill.
    """

    def __init__(self, consumers: dict, reload=None,
                 ram_budget: int | None = None, on_spill=None):
        self._remaining = dict(consumers)
        self._in_ram: dict = {}
        self._sizes: dict = {}
        self._spilled: set = set()
        self._reload = reload
        self._budget = ram_budget
        self._on_spill = on_spill
        self.held_bytes = 0

    def put(self, node: str, port: str, payload) -> None:
        key = (node, port)
        if self._remaining.get(key, 0) <= 0:
            return  # unconsumed output: drop immediately
        nbytes = _payload_nbytes(payload)
        if (
            self._budget is not None
            and self._reload is not None
            and self.held_bytes + nbytes > self._budget
        ):
            self._spilled.add(key)
            if self._on_spill is not None:
                self._on_spill(node, port, nbytes, self._budget)
            return
        self._in_ram[key] = payload
        self._sizes[key] = nbytes
        self.held_bytes += nbytes

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
            if self._in_ram.pop(key, None) is not None:
                self.held_bytes -= self._sizes.pop(key, 0)
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


def _node_params(node, tokens: dict) -> dict:
    """A node's parameters, coerced and with run tokens resolved.

    Flows are complete (validated first), so no value comes from code.
    """
    cls = registry.get(node.type)
    params = cls.coerce_params(node.params, require_complete=True)
    return _resolve_tokens(params, tokens)


def _upstream_recipes(graph: Graph, node_id: str, hashes: dict) -> dict:
    """input port -> recipe hash of the output it consumes."""
    return {
        f"{e.dst.port}": hashes[(e.src.node, e.src.port)]
        for e in graph.in_edges(node_id)
    }


def _node_recipe(graph: Graph, node_id: str, params: dict, hashes: dict) -> str:
    """Recipe hash of one node, given the hashes of its upstream outputs."""
    node = graph.node(node_id)
    upstream = _upstream_recipes(graph, node_id, hashes)
    source_sig = registry.get(node.type).source_signature(params)
    return recipe_hash(node.type, params, upstream, source_sig)


def plan_recipes(graph: Graph, tokens: dict) -> dict[str, str]:
    """Recipe hash of every node, without running anything."""
    hashes: dict = {}
    recipes: dict[str, str] = {}
    for node_id in _topo_order(graph):
        node = graph.node(node_id)
        node_hash = _node_recipe(graph, node_id, _node_params(node, tokens), hashes)
        for port in registry.get(node.type).OUTPUTS:
            hashes[(node_id, port.name)] = node_hash
        recipes[node_id] = node_hash
    return recipes


def _emit(reporter, event: str, *args) -> None:
    """Call an optional reporter method; older reporters may lack it."""
    method = getattr(reporter, event, None)
    if method is not None:
        method(*args)


def _summarize(payload) -> str:
    summary = getattr(payload, "summary", None)
    return summary() if callable(summary) else type(payload).__name__


def run(
    graph: Graph,
    *,
    input_path: str = "",
    out_base: str = "",
    work_dir: str = "work",
    cache: bool = True,
    reporter=None,
    workers: int | None = None,
    skip_types: frozenset | set = frozenset(),
    ram_budget: int | None = None,
    record=None,
    cache_compression: str = "lzf",
) -> dict:
    """Execute a flow.

    Returns {node_id: {"cached": bool, "seconds": float,
    "write_seconds": float}}; ``write_seconds`` is the background cache
    writer's time for that node's outputs (0.0 for sinks, hits, no cache).

    ``record`` (a ``flow.record.RunRecord``) is started before validation
    and finished as ok, failed or interrupted, so every run leaves one.
    """
    if record is not None:
        record.start()
    try:
        summary = _execute(
            graph, input_path=input_path, out_base=out_base,
            work_dir=work_dir, cache=cache, reporter=reporter,
            workers=workers, skip_types=skip_types,
            ram_budget=ram_budget, record=record,
            cache_compression=cache_compression,
        )
    except KeyboardInterrupt:
        if record is not None:
            record.finish("interrupted", "interrupted (Ctrl-C)")
        raise
    except BaseException as exc:
        if record is not None:
            record.finish("failed", str(exc))
        raise
    if record is not None:
        record.finish("ok")
    return summary


def _execute(graph: Graph, *, input_path, out_base, work_dir, cache,
             reporter, workers, skip_types, ram_budget, record,
             cache_compression="lzf") -> dict:
    from karak.flow.budget import default_ram_budget
    from karak.stages.payloads import format_bytes

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
        recipe = hashes[(node, port)]
        if writer is not None:
            _writer_call(writer.wait_for, recipe, port)   # may still be queued
        return load_payload(recipe, port, cache_dir)

    def _on_spill(node, port, nbytes, budget):
        _emit(reporter, "log", "info",
              f"store: {node}.{port} ({format_bytes(nbytes)}) spilled to "
              f"cache, budget {format_bytes(budget)}")

    if not cache:
        budget = None
    else:
        budget = default_ram_budget() if ram_budget is None else ram_budget
    store = PayloadStore(consumers, reload=_reload, ram_budget=budget,
                         on_spill=_on_spill)
    summary: dict = {}
    run_started_at = time.monotonic()
    order = [n for n in _topo_order(graph) if n not in skipped]
    devices = set()
    for node_id in order:
        node = graph.node(node_id)
        coerced = registry.get(node.type).coerce_params(
            node.params, require_complete=True)
        if "device" in coerced:
            devices.add(str(coerced["device"]))
    _emit(reporter, "run_started", RunInfo(
        flow=graph.name,
        input_path=input_path,
        out_base=out_base,
        work_dir=str(work_dir),
        cache=cache,
        workers=workers,
        device=",".join(sorted(devices)) or "cpu",
        version=karak_version(),
        nodes=tuple((n, graph.node(n).type) for n in order),
        record="" if record is None else str(record.path),
    ))

    # Outputs go to disk on a background thread; the next node starts at
    # once. Created last so nothing above can leave the thread running.
    writer = None
    if cache:
        sweep_stale_tmp(cache_dir)   # debris of runs that were killed
        writer = CacheWriter(cache_dir, compression=cache_compression)

    def _drain_writer_log() -> None:
        """Emit the writer's log lines, then fail fast on a writer error."""
        if writer is not None:
            for line in writer.drain_log():
                _emit(reporter, "log", "info", line)
            _writer_call(writer.check)

    try:
        for node_id in _topo_order(graph):
            _drain_writer_log()
            node = graph.node(node_id)
            if node_id in skipped:
                summary[node_id] = {"cached": False, "seconds": 0.0,
                                    "skipped": True, "write_seconds": 0.0}
                if record is not None:
                    record.node(node_id, status="skipped")
                continue
            cls = registry.get(node.type)
            params = _node_params(node, tokens)
            node_hash = _node_recipe(graph, node_id, params, hashes)
            for port in cls.OUTPUTS:
                hashes[(node_id, port.name)] = node_hash
            # "default" means "equals the stage template", for display only
            defaults = _resolve_tokens(cls.coerce_params(cls.template()), tokens)
            _emit(reporter, "node_params", node_id, [
                ParamValue(p.name, params[p.name], params[p.name] == defaults[p.name])
                for p in cls.PARAMS
            ])

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
            _emit(reporter, "node_cache", node_id, node_hash, cached_hit,
                  str(cache_dir))
            if record is not None:
                record.node(node_id, params=params, recipe=node_hash,
                            status="cached" if cached_hit else "running")

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
                summaries = {
                    port.name: load_summary(node_hash, port.name, cache_dir)
                    or (_summarize(outputs[port.name]) if port.name in outputs
                        else "(no summary recorded)")
                    for port in cls.OUTPUTS
                }
            else:
                inputs = {
                    e.dst.port: store.get(e.src.node, e.src.port)
                    for e in graph.in_edges(node_id)
                }
                stage = cls()
                stage.reporter = reporter
                stage.workers = workers
                stage.node_id = node_id
                reporter.node_started(node_id, cls.label or cls.id)
                try:
                    outputs = stage.run(inputs, params)
                except Exception as exc:
                    _emit(reporter, "node_failed", node_id, str(exc))
                    if record is not None:
                        record.node(node_id, status="failed", error=str(exc))
                    raise FlowError(f"node {node_id!r} ({node.type}): {exc}") from exc
                summaries = {name: _summarize(p) for name, p in outputs.items()}
                if cache and not is_sink:
                    upstream = _upstream_recipes(graph, node_id, hashes)
                    for port_name, payload in outputs.items():
                        writer.submit(node_hash, port_name, payload,
                                      summaries[port_name], upstream,
                                      label=f"{node_id}.{port_name}")

            for port_name, payload in outputs.items():
                store.put(node_id, port_name, payload)

            elapsed = time.monotonic() - started
            if summaries:
                _emit(reporter, "node_outputs", node_id, summaries)
            reporter.node_finished(node_id, elapsed, cached_hit)
            summary[node_id] = {"cached": cached_hit, "seconds": elapsed,
                                "write_seconds": 0.0}
            if record is not None:
                record.node(
                    node_id,
                    status="cached" if cached_hit else "ran",
                    seconds=round(elapsed, 3),
                    outputs={
                        port: {
                            "file": (str(payload_path(node_hash, port, cache_dir))
                                     if cache and not is_sink else None),
                            "summary": text,
                        }
                        for port, text in summaries.items()
                    },
                )
            _drain_writer_log()
    except BaseException as failure:
        # A stage failure or Ctrl-C is propagating: finish the queued writes
        # so earlier outputs land, but never mask the original exception.
        # A second Ctrl-C in close() abandons the queue and propagates.
        if writer is not None:
            writer.close()
            for line in writer.drain_log():
                _emit(reporter, "log", "info", line)
            if not isinstance(failure, CacheWriteError):
                try:
                    writer.check()   # a writer error the failure hid
                except Exception as exc:
                    _emit(reporter, "log", "error", f"cache writer: {exc}")
        raise
    if writer is not None:
        try:
            _writer_call(writer.wait)   # a writer error fails the run
        finally:
            writer.close()
            for line in writer.drain_log():
                _emit(reporter, "log", "info", line)
        for node_id, entry in summary.items():
            entry["write_seconds"] = sum(
                seconds for label, seconds in writer.per_label.items()
                if label.startswith(node_id + ".")
            )

    _emit(reporter, "run_finished", summary, time.monotonic() - run_started_at)
    return summary
