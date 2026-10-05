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
2026-10-04: hdbscan_tiled and rare_phase gained `accumulate` (float64 tile
and rare-cluster fingerprint sums by default; float32 reproduces the
published baseline), so in tiled and tiled-rare `hdb`, `rare` and
everything downstream carry new hashes; global and stepwise do not change.
2026-10-05: rare_phase gained a `device` param, so in tiled-rare `rare` and
everything downstream carry new hashes.
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
_GLOBAL_HDB = "001dae34839b919b681ba865cb8fecaa"
_GLOBAL_STATS = "544faeef7033052a4950593773201b02"
_GLOBAL_FP = "368ab0228a129897bd986bfcd137ab87"
_GLOBAL_KNN = "0998c21ffbc4bc42ae9116ab1c4d9327"

GOLDEN = {
    "global": {**_SHARED, "hdb": _GLOBAL_HDB, "knn": _GLOBAL_KNN,
               "stats": _GLOBAL_STATS,
               "fp": _GLOBAL_FP},
    "tiled": {**_SHARED,
              "hdb": "650b15d84812cebb4d3cdf1a9c096bf8",
              "knn": "8c63956522d8073fbb99446576c7f437",
              "stats": "c2c95da3a1734cd85d26b3fda4164626",
              "fp": "141e6c8b6e7861f1d0f166da12b9c043"},
    "tiled-rare": {**_SHARED,
                   "hdb": "650b15d84812cebb4d3cdf1a9c096bf8",
                   "rare": "79294078ee2a071c51dac5730ee5ac7a",
                   "knn": "07698fa3938849eceabdb3d0a2cd05d5",
                   "stats": "fcd955c389911fb8b52842756c856e42",
                   "fp": "a3cfd2f9155b8141a333027786098143"},
    "stepwise": {"src": _SHARED["src"], "msk": _SHARED["msk"],
                 "dn": _SHARED["dn"], "nrm": _SHARED["nrm"],
                 "pca": _SHARED["pca"], "hdb": _GLOBAL_HDB,
                 "knn": _GLOBAL_KNN, "stats": _GLOBAL_STATS,
                 "fp": _GLOBAL_FP},
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_builtin_recipe_hashes_are_stable(name):
    graph = builtin_flow(name)
    hashes = plan_recipes(graph, TOKENS)
    producers = {nid: h for nid, h in hashes.items()
                 if registry.get(graph.node(nid).type).OUTPUTS}
    assert producers == GOLDEN[name]
