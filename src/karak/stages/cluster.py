"""HDBSCAN clustering stages: global and tiled-progressive."""

from __future__ import annotations

from karak.config import ClusterConfig, HDBSCANConfig, TiledConfig
from karak.clustering.hdbscan_cluster import run_hdbscan
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import (
    LabelState,
    Labels,
    Space,
    TiledArtifacts,
)
from karak.stages.registry import register


_HDBSCAN_PARAMS = [
    Param("min_cluster_size", "int", 1000, "Min cluster size", min=1,
          unit="px"),
    Param("min_samples", "int", 0, "Min samples",
          "HDBSCAN min_samples; 0 = defaults to min_cluster_size", min=0),
    Param("subsample_n", "int", 0, "Subsample N",
          "Max pixels for fitting (rest via approximate_predict); 0 = all",
          min=0),
    Param("random_state", "int", 42, "Random seed"),
]


def _hdbscan_config(params: dict) -> HDBSCANConfig:
    return HDBSCANConfig(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"] or None,
        subsample_n=params["subsample_n"] or None,
        random_state=params["random_state"],
    )


@register
class HdbscanGlobalStage(Stage):
    id = "hdbscan_global"
    label = "HDBSCAN (global)"
    description = "Single HDBSCAN run over all mineral-pixel features."
    INPUTS = [Port("features")]
    OUTPUTS = [Port("labels", space=LabelState.RAW)]
    PARAMS = _HDBSCAN_PARAMS

    def apply(self, inputs: dict, params: dict) -> dict:
        features = inputs["features"]
        labels, probabilities, _ = run_hdbscan(
            features.features, _hdbscan_config(params)
        )
        return {
            "labels": Labels(
                labels=labels,
                probabilities=probabilities,
                mineral_indices=features.mineral_indices,
                image_shape=features.image_shape,
                state=LabelState.RAW,
            )
        }


@register
class HdbscanTiledStage(Stage):
    id = "hdbscan_tiled"
    label = "HDBSCAN (tiled)"
    description = (
        "Per-tile HDBSCAN with cosine-similarity phase-registry merging. "
        "Unassigned pixels are left at -1 for the noise_assign stage."
    )
    INPUTS = [Port("features"), Port("cube", space=Space.DENOISED)]
    OUTPUTS = [Port("labels", space=LabelState.RAW), Port("tiles")]
    PARAMS = _HDBSCAN_PARAMS + [
        Param("tile_size", "int", 512, "Tile size", min=1, unit="px"),
        Param("merge_threshold", "float", 0.92, "Merge threshold",
              "Cosine similarity for matching tile clusters to the registry",
              min=0.0, max=1.0),
        Param("min_tile_pixels", "int", 0, "Min tile pixels",
              "Minimum mineral pixels per tile; 0 = 2 * min_cluster_size",
              min=0),
        Param("min_clusters_per_tile", "int", 3, "Min clusters per tile",
              "Tiles with fewer clusters defer to the k-NN pass", min=0),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.tiling import run_tiled_hdbscan

        features, cube = inputs["features"], inputs["cube"]
        config = ClusterConfig(
            strategy="tiled",
            hdbscan=_hdbscan_config(params),
            tiled=TiledConfig(
                tile_size=params["tile_size"],
                merge_threshold=params["merge_threshold"],
                min_tile_pixels=params["min_tile_pixels"] or None,
                min_clusters_per_tile=params["min_clusters_per_tile"],
            ),
        )
        raw_labels, _, probabilities, tile_results, phase_registry = (
            run_tiled_hdbscan(
                features.features,
                features.mineral_indices,
                features.image_shape,
                cube.pixels,
                config,
                skip_knn=True,
            )
        )
        return {
            "labels": Labels(
                labels=raw_labels,
                probabilities=probabilities,
                mineral_indices=features.mineral_indices,
                image_shape=features.image_shape,
                state=LabelState.RAW,
            ),
            "tiles": TiledArtifacts(
                tile_results=tuple(tile_results),
                phase_registry=tuple(phase_registry),
                tile_size=params["tile_size"],
            ),
        }
