"""Recipe hashes must not drift: they name every cached payload on disk.

The values below were captured from the code before flows became complete
(every parameter spelled out in the JSON). Identical effective parameters
must keep hashing identically, or every existing cache is invalidated.
Sinks are left out: they are never cached, and the export sink embeds the
flow JSON itself.

One deliberate change: in tiled-rare, the rare_phase node's merge_threshold
went from 0.0 ("reuse the tiled threshold", which actually took 0.92 from
code) to an explicit 0.92, and its unused noise_reassign_k param was
removed. Same results, new recipe, so rare and its downstream nodes (knn,
stats, fp) carry new hashes; everything upstream is unchanged.
"""

from __future__ import annotations

import pytest

from karak.flow.builtins import builtin_flow
from karak.flow.executor import plan_recipes
from karak.stages import registry

TOKENS = {"{input}": "/nonexistent/input", "{out}": "out/run",
          "{work}": "work", "{flow}": "{}"}

_SHARED = {
    "src": "c733f2990a76f1f1650fd2fc685dede3",
    "msk": "de849a4979083e3a9d90a9ad611e1324",
    "dn": "cb5acfb9994e901758fd9e32846a090a",
    "nrm": "2ebd4311b44fc1e4d315f4a007cf50e7",
    "pca": "eabf6a71a719b0b7fc882f2d3646f35b",
}

GOLDEN = {
    "global": {**_SHARED,
               "hdb": "c331ea63e2ffd50aefd9a1d59fe3cc4b",
               "knn": "6bf74f37ab251d3e87cfd122ad30df94",
               "stats": "b46d91f8abcad42a74cd3ee79e580f70",
               "fp": "94264ddfc490419f8a23599a0d5fda44"},
    "tiled": {**_SHARED,
              "hdb": "8eefd0cddf5bcecb394283a4ba1dac82",
              "knn": "72a8c5215a293e8ab14c912f9d4bc674",
              "stats": "fdf32660026f532dbcaf7275807f7f99",
              "fp": "326ef3acf9c39f1b83d4945a42401e6e"},
    "tiled-rare": {**_SHARED,
                   "hdb": "8eefd0cddf5bcecb394283a4ba1dac82",
                   "rare": "737b1b26501f7153af9a6c22395ea328",
                   "knn": "e07d439d307a85a42957d680a0f12141",
                   "stats": "d3c6f7ddf04adb10b46edb9591467eb0",
                   "fp": "e5024a65188e3da14e1ba42bca816f56"},
    "stepwise": {"src": _SHARED["src"], "msk": _SHARED["msk"]},
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_builtin_recipe_hashes_are_stable(name):
    graph = builtin_flow(name)
    hashes = plan_recipes(graph, TOKENS)
    producers = {nid: h for nid, h in hashes.items()
                 if registry.get(graph.node(nid).type).OUTPUTS}
    assert producers == GOLDEN[name]
