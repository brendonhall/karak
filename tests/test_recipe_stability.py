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
2026-10-06: hdbscan_tiled records deferred tiles and rare_phase leaves their
pixels to noise_assign (recipe revision "deferred-1" on both), so in tiled,
tiled-rare and paper `hdb`, `rare` and everything downstream carry new
hashes; global and stepwise do not change.
2026-10-06: paper gains src_hires, names, oliv, weath, pyx and phos; the
consumers of the final labels (stats, fp) now hash from phos. Other flows
unchanged.
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
    # paper: tiled-rare with the published NWA 4587 settings (own load
    # settings, so no hash is shared with the other flows)
    "paper": {"src": "0ade69c7021c1667b3362450f838bac0",
              "src_hires": "75ff6e71c096687d490e501e3078fa92",
              "msk": "a299625ddfd762b1af547bdbfca0a0b1",
              "dn": "b6aded777b5cfb0fc67e895c3c845d19",
              "nrm": "da983310b20bcb02b559790f0e03dbf4",
              "pca": "9c1401964c98ddd2f5e08334d556383f",
              "hdb": "edd534d8d3b1669e1edac6db07d3cfc6",
              "rare": "067bd67ac63614b485f69adc50b70803",
              "knn": "1d8f7101bf77f325d41b0a7a7a5af8e1",
              "names": "805bc7900a5b41c2b282f1c6167a21a9",
              "oliv": "7c89e275705fe4acc0e08381a7ec2e78",
              "weath": "f8ce6634025c3d8d32beb987941d2f6d",
              "pyx": "c0e302673cac9164b8e6aabbf029d2c1",
              "phos": "fe77dc839623a4486bb5ab14718c7f38",
              "stats": "e09d23e49eaa5fa68262603846b7db54",
              "fp": "5975488fea29c92f5b90527923a7af8d"},
    "global": {**_SHARED, "hdb": _GLOBAL_HDB, "knn": _GLOBAL_KNN,
               "stats": _GLOBAL_STATS,
               "fp": _GLOBAL_FP},
    "tiled": {**_SHARED,
              "hdb": "05c453b59e9cd3f7b803fb53d23a2b7c",
              "knn": "6fe4b7b56ba6c0db3f77e7caeffc7a9f",
              "stats": "58e95769d8ee71766cc89b0425d691d9",
              "fp": "2a23df67f7b73b932ceb1d93af988efb"},
    "tiled-rare": {**_SHARED,
                   "hdb": "05c453b59e9cd3f7b803fb53d23a2b7c",
                   "rare": "0406e9cf75fd015a186e095109ce03b5",
                   "knn": "e0aafdb253d639ff0624ed011d6e3a4a",
                   "stats": "0aecfc43fab2cc5b47511d1bd93bd5d3",
                   "fp": "8d75b0e580f9ced45896d6220729691e"},
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
