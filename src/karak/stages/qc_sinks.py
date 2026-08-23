"""QC sink stages: diagnostic figures per pipeline step.

Each sink consumes payloads and writes matplotlib figures (Agg backend) to
``figure_dir``. Sinks have no outputs and are never cached, so `--no-qc`
can drop them and warm re-runs regenerate figures from cached payloads.
"""

from __future__ import annotations

import json

from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register


_FIGURE_DIR = Param(
    "figure_dir", "str", "{out}/figures", "Figure directory",
    "Directory the diagnostic figures are written to",
)


@register
class QcMaskStage(Stage):
    id = "qc_mask"
    label = "QC: mask"
    description = "Mask coverage overlay with optional TIMA reference panel."
    INPUTS = [
        Port("bse"),
        Port("masks"),
        Port("cube_raw", space=Space.RAW, required=False,
             help="Supplies downsample/trim geometry for TIMA alignment"),
    ]
    OUTPUTS: list = []
    PARAMS = [
        _FIGURE_DIR,
        Param("tima_path", "str", None, "TIMA phase map",
              "Optional reference TIMA phase map image for comparison"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.qc.figures import generate_mask_qc

        cube = inputs.get("cube_raw")
        masks = inputs["masks"]
        generate_mask_qc(
            inputs["bse"].pixels,
            masks.mineral_mask,
            masks.valid_mask,
            masks.stats,
            params["tima_path"],
            cube.downsample_factor if cube is not None else 1,
            cube.header_trim_px if cube is not None else 0,
            params["figure_dir"],
        )
        return {}


@register
class QcDenoiseStage(Stage):
    id = "qc_denoise"
    label = "QC: denoise"
    description = "Before/after denoising comparison panels."
    INPUTS = [
        Port("cube_raw", space=Space.RAW),
        Port("cube_denoised", space=Space.DENOISED),
        Port("bse"),
        Port("masks"),
    ]
    OUTPUTS: list = []
    PARAMS = [
        _FIGURE_DIR,
        Param("method", "str", "bilateral", "Method label",
              "Denoise method name shown on the figure"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.qc.figures import generate_denoise_qc

        raw = inputs["cube_raw"]
        generate_denoise_qc(
            raw.pixels,
            inputs["cube_denoised"].pixels,
            inputs["bse"].pixels,
            inputs["masks"].mineral_mask,
            list(raw.element_names),
            params["method"],
            params["figure_dir"],
        )
        return {}


@register
class QcNormalizeStage(Stage):
    id = "qc_normalize"
    label = "QC: normalize"
    description = "Z-score histograms and channel correlation matrix."
    INPUTS = [Port("cube", space=Space.NORMALIZED), Port("masks")]
    OUTPUTS: list = []
    PARAMS = [_FIGURE_DIR]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.qc.figures import (
            generate_correlation_matrix,
            generate_zscore_histograms,
        )

        cube = inputs["cube"]
        mask = inputs["masks"].mineral_mask
        names = list(cube.element_names)
        generate_zscore_histograms(
            cube.pixels, mask, names, params["figure_dir"]
        )
        generate_correlation_matrix(
            cube.pixels, mask, names, params["figure_dir"]
        )
        return {}


@register
class QcScreeStage(Stage):
    id = "qc_scree"
    label = "QC: scree plot"
    description = "PCA explained-variance scree plot with selection cutoff."
    INPUTS = [Port("features")]
    OUTPUTS: list = []
    PARAMS = [_FIGURE_DIR]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.qc.figures import generate_scree_plot

        features = inputs["features"]
        generate_scree_plot(
            features.explained_variance_ratio,
            features.n_kept,
            params["figure_dir"],
        )
        return {}


@register
class QcPhaseMapStage(Stage):
    id = "qc_phase_map"
    label = "QC: phase map"
    description = "Raw vs cleaned phase map over the BSE image."
    INPUTS = [
        Port("labels_raw", space=LabelState.RAW),
        Port("labels", space=LabelState.CLEANED),
        Port("bse"),
        Port("stats"),
    ]
    OUTPUTS: list = []
    PARAMS = [_FIGURE_DIR]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.noise_assign import labels_to_image
        from karak.qc.figures import generate_phase_map

        raw, cleaned = inputs["labels_raw"], inputs["labels"]
        generate_phase_map(
            labels_to_image(raw.labels, raw.mineral_indices, raw.image_shape),
            labels_to_image(
                cleaned.labels, cleaned.mineral_indices, cleaned.image_shape
            ),
            inputs["bse"].pixels,
            inputs["stats"].stats,
            params["figure_dir"],
        )
        return {}


@register
class QcClusterSummaryStage(Stage):
    id = "qc_cluster_summary"
    label = "QC: cluster summary"
    description = "Cluster size and probability summary chart."
    INPUTS = [Port("stats")]
    OUTPUTS: list = []
    PARAMS = [_FIGURE_DIR]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.qc.figures import generate_cluster_summary

        generate_cluster_summary(inputs["stats"].stats, params["figure_dir"])
        return {}


@register
class QcTiledStage(Stage):
    id = "qc_tiled"
    label = "QC: tiled clustering"
    description = (
        "Tile grid overlay and phase discovery chart. Recomputes the tile "
        "grid from the features payload."
    )
    INPUTS = [Port("bse"), Port("tiles"), Port("features")]
    OUTPUTS: list = []
    PARAMS = [
        _FIGURE_DIR,
        Param("min_tile_pixels", "int", 2000, "Min tile pixels",
              "Must match the hdbscan_tiled setting for an accurate overlay",
              min=1),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.tiling import compute_tile_grid
        from karak.qc.tiled_figures import (
            generate_phase_discovery_chart,
            generate_tile_grid_overlay,
        )

        features = inputs["features"]
        tiles = inputs["tiles"]
        grid = compute_tile_grid(
            features.mineral_indices,
            features.image_shape,
            tiles.tile_size,
            params["min_tile_pixels"],
        )
        generate_tile_grid_overlay(
            inputs["bse"].pixels, grid, list(tiles.tile_results),
            params["figure_dir"],
        )
        generate_phase_discovery_chart(
            list(tiles.tile_results), list(tiles.phase_registry),
            params["figure_dir"],
        )
        return {}


@register
class QcFingerprintsStage(Stage):
    id = "qc_fingerprints"
    label = "QC: fingerprints"
    description = "Per-cluster chemical fingerprint chart."
    INPUTS = [Port("fingerprints")]
    OUTPUTS: list = []
    PARAMS = [
        _FIGURE_DIR,
        Param("mineral_names", "str", None, "Mineral names",
              'JSON mapping of cluster id to name, e.g. {"0": "olivine"}'),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.qc.figures import generate_fingerprint_chart

        names = None
        if params["mineral_names"]:
            names = {
                int(k): v
                for k, v in json.loads(params["mineral_names"]).items()
            }
        generate_fingerprint_chart(
            inputs["fingerprints"].data, params["figure_dir"],
            mineral_names=names,
        )
        return {}


@register
class QcNamedPhaseMapStage(Stage):
    id = "qc_named_phase_map"
    label = "QC: named phase map"
    description = "Final phase map with researcher-assigned mineral names."
    INPUTS = [Port("labels", space=LabelState.CLEANED), Port("bse")]
    OUTPUTS: list = []
    PARAMS = [
        _FIGURE_DIR,
        Param("mineral_names", "str", "{}", "Mineral names",
              'JSON mapping of cluster id to name, e.g. {"0": "olivine"}'),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.noise_assign import labels_to_image
        from karak.qc.figures import generate_named_phase_map

        labels = inputs["labels"]
        names = {
            int(k): v for k, v in json.loads(params["mineral_names"]).items()
        }
        generate_named_phase_map(
            labels_to_image(
                labels.labels, labels.mineral_indices, labels.image_shape
            ),
            inputs["bse"].pixels,
            names,
            params["figure_dir"],
        )
        return {}
