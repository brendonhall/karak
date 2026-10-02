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
    "nrm": "e3dcb4d85ce5f90b92d0301d7d583bd4",
    "pca": "45a7ea9a77dab82a9b598d1925bf9a26",
}

GOLDEN = {
    "global": {**_SHARED,
               "hdb": "06b13f0c6949ffeaf444b1cdd52697e9",
               "knn": "e794780fafbcb13effd4b08d2017e20a",
               "stats": "8622d05748c5a5fb27a50e63e2f6718a",
               "fp": "882515ae986434ccaefe2427c80ea09f"},
    "tiled": {**_SHARED,
              "hdb": "24e1eb7f0560456f4003e24ca4521264",
              "knn": "500ae8c4fcac49750f506e30f372f8bf",
              "stats": "ab30898c2404a832059079a303adc4dc",
              "fp": "6665869557395018dffa71cf4af69f2e"},
    "tiled-rare": {**_SHARED,
                   "hdb": "24e1eb7f0560456f4003e24ca4521264",
                   "rare": "8d2cb990caa0c7aa0414e54e576433e7",
                   "knn": "5a9efa0aacc5250abfe02c4c3160fef9",
                   "stats": "13d698af885e53e33d81eae0d902fddd",
                   "fp": "d45a6316582c29c959fa60087e2d4506"},
    "stepwise": {"src": _SHARED["src"], "msk": _SHARED["msk"],
                 "dn": _SHARED["dn"]},
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_builtin_recipe_hashes_are_stable(name):
    graph = builtin_flow(name)
    hashes = plan_recipes(graph, TOKENS)
    producers = {nid: h for nid, h in hashes.items()
                 if registry.get(graph.node(nid).type).OUTPUTS}
    assert producers == GOLDEN[name]
