"""Stage.recipe_revision: a stage whose results change for some params
under an unchanged recipe adds a revision to those recipes, so an old
cache entry is not reused; every other recipe stays the same."""

from __future__ import annotations

import pytest

from conftest import complete
from karak.flow.cache import recipe_hash
from karak.flow.executor import _node_recipe, run
from karak.flow.graph import Edge, Endpoint, Graph, Node
from karak.stages import registry
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import ClusterStats


class RevSource(Stage):
    id = "rev_source"
    OUTPUTS = [Port("num")]
    PARAMS = [Param("value", "int", 1)]

    def apply(self, inputs, params):
        return {"num": ClusterStats(stats={"value": params["value"]})}


class RevMiddle(Stage):
    id = "rev_middle"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]
    PARAMS = [Param("mode", "str", "old", choices=("old", "new"))]

    @classmethod
    def recipe_revision(cls, params):
        return "2" if params["mode"] == "new" else None

    def apply(self, inputs, params):
        return {"num": ClusterStats(stats={"mode": params["mode"]})}


class RevSink(Stage):
    id = "rev_down"
    INPUTS = [Port("num")]
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        return {"num": inputs["num"]}


@pytest.fixture(autouse=True)
def _stages():
    for cls in (RevSource, RevMiddle, RevSink):
        registry.register(cls)
    yield
    for cls in (RevSource, RevMiddle, RevSink):
        registry._REGISTRY.pop(cls.id, None)


def _graph(mode):
    graph = Graph(name="rev", nodes=(
        Node("a", "rev_source"), Node("b", "rev_middle", {"mode": mode}),
        Node("c", "rev_down"),
    ), edges=(
        Edge("e1", Endpoint("a", "num"), Endpoint("b", "num")),
        Edge("e2", Endpoint("b", "num"), Endpoint("c", "num")),
    ))
    return complete(graph)


def _old_recipes(graph):
    """Recipes as the executor computed them before recipe_revision."""
    hashes, out = {}, {}
    for node_id in ("a", "b", "c"):
        node = graph.node(node_id)
        upstream = {e.dst.port: hashes[(e.src.node, e.src.port)]
                    for e in graph.in_edges(node_id)}
        out[node_id] = recipe_hash(node.type, node.params, upstream, None)
        hashes[(node_id, "num")] = out[node_id]
    return out


def test_no_revision_keeps_the_old_recipe():
    graph = _graph("old")
    old = _old_recipes(graph)
    assert _node_recipe(graph, "b", graph.node("b").params,
                        {("a", "num"): old["a"]}) == old["b"]


def test_old_cache_entries_are_recomputed_under_a_revision(tmp_path):
    from karak.flow.cache import store_payload

    graph = _graph("new")
    cache_dir = tmp_path / "work" / "cache"
    # seed the old-semantics cache: same params, recipes without a revision
    for node_id, recipe in _old_recipes(graph).items():
        store_payload(recipe, "num", ClusterStats(stats={"stale": True}), cache_dir)

    summary = run(graph, out_base=str(tmp_path / "out"),
                  work_dir=str(tmp_path / "work"))
    assert summary["a"]["cached"] is True           # no revision: reused
    assert summary["b"]["cached"] is False          # revised: recomputed
    assert summary["c"]["cached"] is False          # downstream of it too


def test_hdbscan_cuda_subsample_recipes_carry_a_revision():
    from karak.stages.cluster import HdbscanGlobalStage, HdbscanTiledStage

    for cls in (HdbscanGlobalStage, HdbscanTiledStage):
        params = cls.template()
        # hdbscan_tiled always carries "deferred-1" (2026-10-06); the cuda
        # subsample tag shows only when it applies
        base = "deferred-1" if cls is HdbscanTiledStage else None
        assert cls.recipe_revision({**params, "device": "cpu",
                                    "subsample_n": 500}) == base
        assert cls.recipe_revision({**params, "device": "cuda",
                                    "subsample_n": 0}) == base
        assert "cuda-subsample-1" in cls.recipe_revision(
            {**params, "device": "cuda", "subsample_n": 500})


def test_builtin_recipes_do_not_change():
    # the shipped flows run hdb on cpu with subsample_n 0: no cuda revision;
    # the tiled ones carry "deferred-1" (2026-10-06), global none
    from karak.flow.builtins import builtin_flow

    for name, expected in (("global", None), ("tiled", "deferred-1"),
                           ("tiled-rare", "deferred-1")):
        hdb = builtin_flow(name).node("hdb")
        assert registry.get(hdb.type).recipe_revision(hdb.params) == expected
