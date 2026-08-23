"""Export sink: write pipeline results to the provenance HDF5 file.

The HDF5 file is a *product* of a flow run, not runtime state — caching and
resume live in the flow executor. This sink reproduces the legacy group
layout (raw/bse/masks/denoised/normalized/clusters) so downstream notebooks
keep working, and embeds the flow definition plus library versions as root
attributes for provenance.
"""

from __future__ import annotations

import json

from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register


def _params_for(flow: dict, stage_type: str) -> dict:
    """Params of the first node of the given type in the flow definition."""
    for node in flow.get("nodes", []):
        if node.get("type") == stage_type:
            return dict(node.get("params", {}))
    return {}


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
        Port("cube_raw", space=Space.RAW, required=False),
        Port("bse", required=False),
        Port("masks", required=False),
        Port("cube_denoised", space=Space.DENOISED, required=False),
        Port("cube_normalized", space=Space.NORMALIZED, required=False),
        Port("features", required=False),
        Port("labels_raw", space=LabelState.RAW, required=False),
        Port("labels", space=LabelState.CLEANED, required=False),
        Port("stats", required=False),
        Port("tiles", required=False),
    ]
    OUTPUTS: list = []
    PARAMS = [
        Param("path", "str", "{out}.h5", "Output path"),
        Param("flow_json", "str", "{flow}", "Flow definition",
              "JSON of the executing flow, embedded for provenance"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.io import storage

        path = params["path"]
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
            storage.save_raw_data(path, elements, names)

        bse = inputs.get("bse")
        if bse is not None:
            factor = cube_raw.downsample_factor if cube_raw is not None else 1
            storage.save_bse(
                path, bse.pixels,
                original_shape=bse.pixels.shape,
                downsample_factor=factor,
            )

        masks = inputs.get("masks")
        if masks is not None:
            storage.save_mask(
                path, masks.mineral_mask, masks.valid_mask, masks.stats,
                _params_for(flow, "mask"),
            )

        denoised = inputs.get("cube_denoised")
        if denoised is not None:
            storage.save_denoised_data(
                path, denoised.pixels, list(denoised.element_names),
                _params_for(flow, "denoise"),
            )

        normalized = inputs.get("cube_normalized")
        if normalized is not None:
            storage.save_normalized_data(
                path, normalized.pixels, normalized.means, normalized.stds,
                list(normalized.element_names),
                method=_params_for(flow, "normalize").get("method", "zscore"),
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
                               "rare_phase", "noise_assign", "refine"):
                node_params = _params_for(flow, stage_type)
                if node_params:
                    cluster_params[stage_type] = node_params
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
            )
            if tiles is not None:
                storage.save_tiled_metadata(
                    path, list(tiles.tile_results), list(tiles.phase_registry)
                )

        return {}
