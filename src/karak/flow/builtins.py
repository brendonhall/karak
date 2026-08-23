"""Builtin flows: the standard pipelines expressed as graphs.

The same three flows also ship as JSON under ``karak/flow/flows/`` (kept in
sync by a round-trip test) so the file format stays exercised and users can
copy one as a starting point for custom pipelines.
"""

from __future__ import annotations

from karak.flow.graph import Edge, Endpoint, Graph, Node


def _endpoint(spec: str) -> Endpoint:
    node, port = spec.split(".")
    return Endpoint(node, port)


def _edges(*specs: str) -> list[Edge]:
    edges = []
    for i, spec in enumerate(specs):
        src, dst = spec.split("->")
        edges.append(Edge(f"e{i + 1}", _endpoint(src), _endpoint(dst)))
    return edges


def _common_head() -> list[Node]:
    return [
        Node("src", "load_elements", {"input_dir": "{input}"}),
        Node("msk", "mask"),
        Node("dn", "denoise"),
        Node("nrm", "normalize"),
        Node("pca", "pca"),
    ]


def _common_tail() -> list[Node]:
    return [
        Node("knn", "noise_assign"),
        Node("stats", "cluster_stats"),
        Node("fp", "fingerprints"),
        Node("exp", "export_h5"),
        Node("qc_mask", "qc_mask"),
        Node("qc_denoise", "qc_denoise"),
        Node("qc_normalize", "qc_normalize"),
        Node("qc_scree", "qc_scree"),
        Node("qc_phase_map", "qc_phase_map"),
        Node("qc_cluster_summary", "qc_cluster_summary"),
        Node("qc_fingerprints", "qc_fingerprints"),
    ]


_HEAD_EDGES = [
    "src.cube->msk.cube",
    "src.cube->dn.cube",
    "msk.masks->dn.masks",
    "dn.cube->nrm.cube",
    "msk.masks->nrm.masks",
    "nrm.cube->pca.cube",
    "msk.masks->pca.masks",
]

# Edges shared by every flow once raw labels exist on node "cl"
# (hdbscan_global or the last tiled-labels producer).


def _tail_edges(labels_node: str) -> list[str]:
    return [
        f"{labels_node}.labels->knn.labels",
        "pca.features->knn.features",
        "knn.labels->stats.labels",
        "knn.labels->fp.labels",
        "dn.cube->fp.cube",
        # export
        "src.cube->exp.cube_raw",
        "src.bse->exp.bse",
        "msk.masks->exp.masks",
        "dn.cube->exp.cube_denoised",
        "nrm.cube->exp.cube_normalized",
        "pca.features->exp.features",
        f"{labels_node}.labels->exp.labels_raw",
        "knn.labels->exp.labels",
        "stats.stats->exp.stats",
        # qc
        "src.bse->qc_mask.bse",
        "msk.masks->qc_mask.masks",
        "src.cube->qc_mask.cube_raw",
        "src.cube->qc_denoise.cube_raw",
        "dn.cube->qc_denoise.cube_denoised",
        "src.bse->qc_denoise.bse",
        "msk.masks->qc_denoise.masks",
        "nrm.cube->qc_normalize.cube",
        "msk.masks->qc_normalize.masks",
        "pca.features->qc_scree.features",
        f"{labels_node}.labels->qc_phase_map.labels_raw",
        "knn.labels->qc_phase_map.labels",
        "src.bse->qc_phase_map.bse",
        "stats.stats->qc_phase_map.stats",
        "stats.stats->qc_cluster_summary.stats",
        "fp.fingerprints->qc_fingerprints.fingerprints",
    ]


def _tiled_qc_edges(tiles_node: str) -> list[str]:
    return [
        "src.bse->qc_tiled.bse",
        f"{tiles_node}.tiles->qc_tiled.tiles",
        "pca.features->qc_tiled.features",
        f"{tiles_node}.tiles->exp.tiles",
    ]


def global_flow() -> Graph:
    nodes = (
        *_common_head(),
        Node("hdb", "hdbscan_global"),
        *_common_tail(),
    )
    edges = _edges(*_HEAD_EDGES, "pca.features->hdb.features",
                   *_tail_edges("hdb"))
    return Graph(nodes=nodes, edges=tuple(edges), name="global")


def tiled_flow() -> Graph:
    nodes = (
        *_common_head(),
        Node("hdb", "hdbscan_tiled"),
        *_common_tail(),
        Node("qc_tiled", "qc_tiled"),
    )
    edges = _edges(
        *_HEAD_EDGES,
        "pca.features->hdb.features",
        "dn.cube->hdb.cube",
        *_tail_edges("hdb"),
        *_tiled_qc_edges("hdb"),
    )
    return Graph(nodes=nodes, edges=tuple(edges), name="tiled")


def tiled_rare_flow() -> Graph:
    nodes = (
        *_common_head(),
        Node("hdb", "hdbscan_tiled"),
        Node("rare", "rare_phase"),
        *_common_tail(),
        Node("qc_tiled", "qc_tiled"),
    )
    edges = _edges(
        *_HEAD_EDGES,
        "pca.features->hdb.features",
        "dn.cube->hdb.cube",
        "hdb.labels->rare.labels",
        "pca.features->rare.features",
        "dn.cube->rare.cube",
        "hdb.tiles->rare.tiles",
        *_tail_edges("rare"),
        *_tiled_qc_edges("rare"),
    )
    return Graph(nodes=nodes, edges=tuple(edges), name="tiled-rare")


_BUILTINS = {
    "global": global_flow,
    "tiled": tiled_flow,
    "tiled-rare": tiled_rare_flow,
}


def builtin_flow(name: str) -> Graph:
    return _BUILTINS[name]()


def builtin_names() -> list[str]:
    return sorted(_BUILTINS)


def override_params(graph: Graph, overrides: dict) -> Graph:
    """Return a new Graph with ``{"node.param": value}`` overrides applied."""
    updates: dict[str, dict] = {}
    for spec, value in overrides.items():
        node_id, param = spec.split(".", 1)
        graph.node(node_id)  # raises KeyError for unknown nodes
        updates.setdefault(node_id, {})[param] = value
    nodes = tuple(
        node if node.id not in updates
        else Node(
            id=node.id,
            type=node.type,
            params={**node.params, **updates[node.id]},
            ui=node.ui,
        )
        for node in graph.nodes
    )
    return Graph(
        nodes=nodes, edges=graph.edges, name=graph.name, version=graph.version
    )
