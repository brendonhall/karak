"""Export sink: write pipeline results to the provenance HDF5 file.

The HDF5 file is a *product* of a flow run, not runtime state — caching and
resume live in the flow executor. This sink reproduces the legacy group
layout (raw/bse/masks/denoised/normalized/clusters) so downstream notebooks
keep working, and embeds the flow definition plus library versions as root
attributes for provenance.
"""

from __future__ import annotations

import json

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register


def _node_params(flow: dict, stage_type: str) -> dict | None:
    """Params of the first node of the given type, or None if absent."""
    for node in flow.get("nodes", []):
        if node.get("type") == stage_type:
            return dict(node.get("params", {}))
    return None


def _params_for(flow: dict, stage_type: str) -> dict:
    """Complete params of the node that produced a group being written.

    The embedded flow is the complete flow that ran, so the values recorded
    in the HDF5 are the values used; nothing is filled in here.
    """
    from karak.stages import registry

    params = _node_params(flow, stage_type)
    missing = (None if params is None else
               [p.name for p in registry.get(stage_type).PARAMS
                if p.name not in params])
    if params is None or missing:
        raise StageError(
            f"export_h5: flow_json needs a complete {stage_type!r} node "
            f"(missing: {missing if params is not None else 'the node'})"
        )
    return params


@register
class ExportH5Stage(Stage):
    id = "export_h5"
    label = "Export HDF5"
    description = (
        "Write connected results to the provenance HDF5 file using the "
        "legacy group layout. Every input is optional; only connected "
        "groups are written."
    )
    INPUTS = [
        Port("cube_raw", space=Space.RAW, required=False,
             help="written to the raw/ group, one dataset per element"),
        Port("bse", required=False, help="written to bse/image"),
        Port("masks", required=False,
             help="written to masks/mineral and masks/valid"),
        Port("cube_denoised", space=Space.DENOISED, required=False,
             help="written to denoised/cube"),
        Port("cube_normalized", space=Space.NORMALIZED, required=False,
             help="written to normalized/{cube,means,stds}"),
        Port("features", required=False,
             help="supplies pca_variance_ratio and the component count"),
        Port("labels_raw", space=LabelState.RAW, required=False,
             help="written to clusters/raw_labels"),
        Port("labels", space=LabelState.CLEANED, required=False,
             help="written to clusters/cleaned_labels (+ probabilities)"),
        Port("stats", required=False,
             help="cluster statistics stored as clusters/ attributes"),
        Port("tiles", required=False,
             help="written to clusters/tiled/ (tile metadata + registry)"),
    ]
    OUTPUTS: list = []
    PARAMS = [
        Param("path", "str", "{out}.h5", "Output path"),
        Param("flow_json", "str", "{flow}", "Flow definition",
              "JSON of the executing flow, embedded for provenance"),
        Param("compression", "enum", "gzip", "Compression",
              "gzip (level 4, readable by any HDF5 tool), lzf (faster, h5py "
              "and PyTables only) or none; the denoised and normalized cubes "
              "also use the shuffle filter",
              choices=("gzip", "lzf", "none")),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.io import storage

        path = params["path"]
        compression = params["compression"]
        try:
            flow = json.loads(params["flow_json"] or "{}")
        except json.JSONDecodeError:
            flow = {}

        storage.create_pipeline_hdf5(path, flow)

        cube_raw = inputs.get("cube_raw")
        if cube_raw is not None:
            names = list(cube_raw.element_names)
            elements = {
                name: cube_raw.pixels[:, :, i] for i, name in enumerate(names)
            }
            storage.save_raw_data(path, elements, names, compression=compression)

        bse = inputs.get("bse")
        if bse is not None:
            factor = cube_raw.downsample_factor if cube_raw is not None else 1
            storage.save_bse(
                path, bse.pixels,
                original_shape=bse.pixels.shape,
                downsample_factor=factor, compression=compression,
            )

        masks = inputs.get("masks")
        if masks is not None:
            storage.save_mask(
                path, masks.mineral_mask, masks.valid_mask, masks.stats,
                _params_for(flow, "mask"), compression=compression,
            )

        denoised = inputs.get("cube_denoised")
        if denoised is not None:
            storage.save_denoised_data(
                path, denoised.pixels, list(denoised.element_names),
                _params_for(flow, "denoise"), compression=compression,
            )

        normalized = inputs.get("cube_normalized")
        if normalized is not None:
            storage.save_normalized_data(
                path, normalized.pixels, normalized.means, normalized.stds,
                list(normalized.element_names),
                method=_params_for(flow, "normalize")["method"],
                compression=compression,
            )

        labels = inputs.get("labels")
        stats = inputs.get("stats")
        features = inputs.get("features")
        if labels is not None and stats is not None and features is not None:
            labels_raw = inputs.get("labels_raw")
            tiles = inputs.get("tiles")
            cluster_params: dict = {
                "strategy": "tiled" if tiles is not None else "global",
            }
            for stage_type in ("pca", "hdbscan_global", "hdbscan_tiled",
                               "rare_phase", "noise_assign"):
                if _node_params(flow, stage_type) is not None:
                    cluster_params[stage_type] = _params_for(flow, stage_type)
            refinement_types = ("name_phases", "split_threshold", "split_gmm", "split_hires")
            refinement_nodes = [
                {"id": node["id"], "type": node["type"], "params": dict(node.get("params", {}))}
                for node in flow.get("nodes", []) if node.get("type") in refinement_types
            ]
            if refinement_nodes:
                cluster_params["refinement_nodes"] = refinement_nodes
            storage.save_cluster_data(
                path,
                labels_raw.labels if labels_raw is not None else labels.labels,
                labels.labels,
                labels.probabilities,
                features.explained_variance_ratio,
                labels.mineral_indices,
                stats.stats,
                features.n_kept,
                cluster_params,
                compression=compression,
            )
            if labels.names:
                storage.save_mineral_names(path, dict(labels.names))
            if labels.history:
                storage.save_subclustering(path, list(labels.history))
            if tiles is not None:
                storage.save_tiled_metadata(
                    path, list(tiles.tile_results), list(tiles.phase_registry)
                )

        return {}
