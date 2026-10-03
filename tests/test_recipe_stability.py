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
2026-09-30: normalize and pca gained a `device` param, so `nrm`, `pca` and
everything downstream carry new hashes.
2026-10-01: normalize gained `accumulate` (float64 by default; float32
reproduces the published baseline), so `nrm` and everything downstream carry
new hashes again.
2026-10-02: fingerprints gained `accumulate` (float64 by default; float32
reproduces the published baseline), so `fp` carries a new hash; `fp` is a
leaf, so no other hash changes.
2026-10-03: noise_assign gained a `device` param, so `knn` and everything
downstream (`stats`, `fp`) carry new hashes.
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
    "nrm": "e22f6388e1c0d49a47ba28742638754f",
    "pca": "700ce0030be91ee0d4d2e06cea32ef2c",
}

GOLDEN = {
    "global": {**_SHARED,
               "hdb": "001dae34839b919b681ba865cb8fecaa",
               "knn": "0998c21ffbc4bc42ae9116ab1c4d9327",
               "stats": "544faeef7033052a4950593773201b02",
               "fp": "368ab0228a129897bd986bfcd137ab87"},
    "tiled": {**_SHARED,
              "hdb": "8a54033635797cd25cc20df0f519cc4b",
              "knn": "5f0978cc4e043fbfcf6bea5c190d6758",
              "stats": "31c803d804ea99e837df8241c1010415",
              "fp": "885b7b0f838c52d97f4d92d2f1f3c511"},
    "tiled-rare": {**_SHARED,
                   "hdb": "8a54033635797cd25cc20df0f519cc4b",
                   "rare": "34d2726979907355c0f8ae7c827b885d",
                   "knn": "9c40ec9e4eb21cf95b302fe80ca7d281",
                   "stats": "2c44d018a769817a8d9ae27365ca3445",
                   "fp": "f4229c209a91147a8b21b6d3ccc4f5e7"},
    "stepwise": {"src": _SHARED["src"], "msk": _SHARED["msk"],
                 "dn": _SHARED["dn"], "nrm": _SHARED["nrm"],
                 "pca": _SHARED["pca"]},
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_builtin_recipe_hashes_are_stable(name):
    graph = builtin_flow(name)
    hashes = plan_recipes(graph, TOKENS)
    producers = {nid: h for nid, h in hashes.items()
                 if registry.get(graph.node(nid).type).OUTPUTS}
    assert producers == GOLDEN[name]
