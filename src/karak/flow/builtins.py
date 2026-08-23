"""Builtin flows: the standard pipelines expressed as graphs.

The same three flows also ship as JSON under ``karak/flow/flows/`` (kept in
sync by a round-trip test) so the file format stays exercised and users can
copy one as a starting point for custom pipelines.

``flow_from_config`` converts a legacy YAML ``PipelineConfig`` into the
equivalent flow graph — the shim behind ``karak -c config.yaml``.
"""

from __future__ import annotations

from karak.flow.graph import Edge, Endpoint, Graph, Node


def _endpoint(spec: str) -> Endpoint:
    node, port = spec.split(".")
    return Endpoint(node, port)


def _assemble(
    name: str,
    *,
    tiled: bool,
    rare: bool = False,
    refine: bool = False,
    node_params: dict | None = None,
) -> Graph:
    """Build a standard pipeline graph.

    Node/edge ordering is stable: the three no-param variants must stay
    byte-identical to the shipped ``flows/*.json`` files.
    """
    node_params = node_params or {}

    def _node(node_id: str, stage_type: str) -> Node:
        return Node(node_id, stage_type, node_params.get(node_id, {}))

    nodes: list[Node] = [
        _node("src", "load_elements"),
        _node("msk", "mask"),
        _node("dn", "denoise"),
        _node("nrm", "normalize"),
        _node("pca", "pca"),
        _node("hdb", "hdbscan_tiled" if tiled else "hdbscan_global"),
    ]
    if rare:
        nodes.append(_node("rare", "rare_phase"))
    if refine:
        nodes.append(_node("ref", "refine"))
    nodes += [
        _node("knn", "noise_assign"),
        _node("stats", "cluster_stats"),
        _node("fp", "fingerprints"),
        _node("exp", "export_h5"),
        _node("qc_mask", "qc_mask"),
        _node("qc_denoise", "qc_denoise"),
        _node("qc_normalize", "qc_normalize"),
        _node("qc_scree", "qc_scree"),
        _node("qc_phase_map", "qc_phase_map"),
        _node("qc_cluster_summary", "qc_cluster_summary"),
        _node("qc_fingerprints", "qc_fingerprints"),
    ]
    if tiled:
        nodes.append(_node("qc_tiled", "qc_tiled"))

    # "{input}" default for the source directory unless the shim set one
    if "input_dir" not in nodes[0].params:
        nodes[0] = Node(
            "src", "load_elements",
            {**nodes[0].params, "input_dir": "{input}"},
        )

    labels_node = "rare" if rare else "hdb"       # raw-labels producer
    cleaned_node = "ref" if refine else "knn"     # cleaned-labels producer
    tiles_node = "rare" if rare else "hdb"

    specs = [
        "src.cube->msk.cube",
        "src.cube->dn.cube",
        "msk.masks->dn.masks",
        "dn.cube->nrm.cube",
        "msk.masks->nrm.masks",
        "nrm.cube->pca.cube",
        "msk.masks->pca.masks",
        "pca.features->hdb.features",
    ]
    if tiled:
        specs.append("dn.cube->hdb.cube")
    if rare:
        specs += [
            "hdb.labels->rare.labels",
            "pca.features->rare.features",
            "dn.cube->rare.cube",
            "hdb.tiles->rare.tiles",
        ]
    specs += [
        f"{labels_node}.labels->knn.labels",
        "pca.features->knn.features",
    ]
    if refine:
        specs += [
            "knn.labels->ref.labels",
            "dn.cube->ref.cube",
            "src.bse->ref.bse",
        ]
    specs += [
        f"{cleaned_node}.labels->stats.labels",
        f"{cleaned_node}.labels->fp.labels",
        "dn.cube->fp.cube",
        # export
        "src.cube->exp.cube_raw",
        "src.bse->exp.bse",
        "msk.masks->exp.masks",
        "dn.cube->exp.cube_denoised",
        "nrm.cube->exp.cube_normalized",
        "pca.features->exp.features",
        f"{labels_node}.labels->exp.labels_raw",
        f"{cleaned_node}.labels->exp.labels",
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
        f"{cleaned_node}.labels->qc_phase_map.labels",
        "src.bse->qc_phase_map.bse",
        "stats.stats->qc_phase_map.stats",
        "stats.stats->qc_cluster_summary.stats",
        "fp.fingerprints->qc_fingerprints.fingerprints",
    ]
    if tiled:
        specs += [
            "src.bse->qc_tiled.bse",
            f"{tiles_node}.tiles->qc_tiled.tiles",
            "pca.features->qc_tiled.features",
            f"{tiles_node}.tiles->exp.tiles",
        ]

    edges = tuple(
        Edge(f"e{i + 1}", _endpoint(src_dst.split("->")[0]),
             _endpoint(src_dst.split("->")[1]))
        for i, src_dst in enumerate(specs)
    )
    return Graph(nodes=tuple(nodes), edges=edges, name=name)


def global_flow() -> Graph:
    return _assemble("global", tiled=False)


def tiled_flow() -> Graph:
    return _assemble("tiled", tiled=True)


def tiled_rare_flow() -> Graph:
    return _assemble("tiled-rare", tiled=True, rare=True)


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


def _drop_none(params: dict) -> dict:
    return {k: v for k, v in params.items() if v is not None}


def flow_from_config(cfg) -> Graph:
    """Convert a legacy ``PipelineConfig`` into the equivalent flow graph."""
    tiled = cfg.cluster.strategy == "tiled"
    rare = tiled and cfg.cluster.rare_phase.enabled
    refine = cfg.cluster.refinement.enabled

    down, loader = cfg.downsample, cfg.loader
    hdb, pca = cfg.cluster.hdbscan, cfg.cluster.pca
    tiled_cfg, rare_cfg = cfg.cluster.tiled, cfg.cluster.rare_phase
    ref_cfg = cfg.cluster.refinement

    hdb_params = {
        "min_cluster_size": hdb.min_cluster_size,
        "min_samples": hdb.min_samples or 0,
        "subsample_n": hdb.subsample_n or 0,
        "random_state": hdb.random_state,
    }
    if tiled:
        hdb_params.update({
            "tile_size": tiled_cfg.tile_size,
            "merge_threshold": tiled_cfg.merge_threshold,
            "min_tile_pixels": tiled_cfg.min_tile_pixels or 0,
            "min_clusters_per_tile": tiled_cfg.min_clusters_per_tile,
        })

    figure = {"figure_dir": cfg.figure_dir}
    node_params: dict[str, dict] = {
        "src": _drop_none({
            "input_dir": cfg.input_dir,
            "file_glob": loader.file_glob,
            "filename_pattern": loader.filename_pattern,
            "bse_filename": loader.bse_filename,
            "colormap": loader.colormap,
            "exclude_elements": ",".join(cfg.exclude_elements),
            "bse_channel": cfg.bse_channel,
            "header_trim_px": down.header_trim_px,
            "bottom_trim_px": down.bottom_trim_px,
            "left_trim_px": down.left_trim_px,
            "right_trim_px": down.right_trim_px,
            "downsample_factor": down.downsample_factor,
        }),
        "msk": _drop_none({
            "min_object_size": cfg.mask.min_object_size,
            "valid_mask_path": cfg.mask.valid_mask_path,
        }),
        "dn": _drop_none({
            "method": cfg.denoise.method,
            "sigma_color": cfg.denoise.sigma_color,
            "sigma_spatial": cfg.denoise.sigma_spatial,
            "niter": cfg.denoise.niter,
            "kappa": cfg.denoise.kappa,
            "gamma": cfg.denoise.gamma,
            "option": cfg.denoise.option,
        }),
        "nrm": {"method": cfg.normalize.method},
        "pca": {
            "n_components": pca.n_components or 0,
            "subsample_fraction": pca.subsample_fraction or 0,
            "random_state": pca.random_state,
        },
        "hdb": hdb_params,
        "knn": {"k": hdb.noise_reassign_k},
        "exp": {"path": cfg.hdf5_output},
        "qc_mask": dict(figure),
        "qc_denoise": {**figure, "method": cfg.denoise.method},
        "qc_normalize": dict(figure),
        "qc_scree": dict(figure),
        "qc_phase_map": dict(figure),
        "qc_cluster_summary": dict(figure),
        "qc_fingerprints": dict(figure),
    }
    if tiled:
        node_params["qc_tiled"] = {
            **figure,
            "min_tile_pixels": (
                tiled_cfg.min_tile_pixels or 2 * hdb.min_cluster_size
            ),
        }
    if rare:
        node_params["rare"] = {
            "min_cluster_size": rare_cfg.min_cluster_size,
            "min_samples": rare_cfg.min_samples or 0,
            "subsample_n": rare_cfg.subsample_n or 0,
            "merge_threshold": rare_cfg.merge_threshold or 0,
            "noise_reassign_k": hdb.noise_reassign_k,
            "random_state": hdb.random_state,
        }
    if refine:
        node_params["ref"] = {
            "target_phase": ref_cfg.target_phase,
            "olivine_enabled": ref_cfg.olivine.enabled,
            "olivine_fe_threshold": ref_cfg.olivine.fe_threshold,
            "olivine_ca_threshold": ref_cfg.olivine.ca_threshold,
            "gmm_enabled": ref_cfg.gmm_split.enabled,
            "gmm_n_components": ref_cfg.gmm_split.n_components,
            "gmm_features": ",".join(ref_cfg.gmm_split.features),
            "gmm_bse_weight": ref_cfg.gmm_split.bse_weight,
            "gmm_subsample_n": ref_cfg.gmm_split.subsample_n or 0,
            "random_state": ref_cfg.gmm_split.random_state,
        }

    return _assemble(
        "from-config",
        tiled=tiled,
        rare=rare,
        refine=refine,
        node_params=node_params,
    )
