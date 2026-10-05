"""HDBSCAN clustering stages: global and tiled-progressive."""

from __future__ import annotations

from karak.core_params import HDBSCANConfig, TiledConfig
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
    Param("device", "str", "cpu", "Device",
          "cpu (hdbscan package, exact baseline) or cuda (cuML; needs "
          "karak[cuda]). cuda results differ from cpu; with subsample_n "
          "both fit the same subsample.",
          choices=("cpu", "cuda")),
]


def _cuda_subsample_revision(params: dict) -> str | None:
    """cuda with subsample_n fitted every pixel before 2026-10-02 and now
    fits the subsample, under the same params: revise those recipes."""
    if params["device"] == "cuda" and params["subsample_n"]:
        return "cuda-subsample-1"
    return None


def hdbscan_config(params: dict) -> HDBSCANConfig:
    return HDBSCANConfig(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"] or None,
        subsample_n=params["subsample_n"] or None,
        random_state=params["random_state"],
    )


def tiled_config(params: dict) -> TiledConfig:
    return TiledConfig(
        tile_size=params["tile_size"],
        merge_threshold=params["merge_threshold"],
        min_tile_pixels=params["min_tile_pixels"] or None,
        min_clusters_per_tile=params["min_clusters_per_tile"],
        accumulate=params["accumulate"],
    )


@register
class HdbscanGlobalStage(Stage):
    id = "hdbscan_global"
    label = "HDBSCAN (global)"
    description = "Single HDBSCAN run over all mineral-pixel features."
    INPUTS = [Port("features", help="PCA features for all mineral pixels")]
    OUTPUTS = [
        Port("labels", space=LabelState.RAW,
             help="per-pixel phase labels; -1 = HDBSCAN noise"),
    ]
    PARAMS = _HDBSCAN_PARAMS

    @classmethod
    def recipe_revision(cls, params: dict) -> str | None:
        return _cuda_subsample_revision(params)

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.accel import resolve_workers

        features = inputs["features"]
        # --workers N: N core-distance jobs and N prediction processes on
        # cpu (same labels); serial keeps hdbscan's default of 4 jobs.
        workers = resolve_workers(self.workers)
        labels, probabilities, _ = run_hdbscan(
            features.features, hdbscan_config(params),
            device=params["device"],
            core_dist_n_jobs=workers if workers > 1 else None,
            predict_workers=workers,
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
    INPUTS = [
        Port("features", help="PCA features for all mineral pixels"),
        Port("cube", space=Space.DENOISED,
             help="denoised cube; used to fingerprint tile clusters for "
                  "registry matching"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.RAW,
             help="registry-unified labels; -1 = unassigned/deferred"),
        Port("tiles", help="per-tile diagnostics + the phase registry"),
    ]
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
        Param("accumulate", "enum", "float64", "Accumulate",
              "Precision of the tile-fingerprint sums: float64 is "
              "accurate; float32 reproduces the published baseline",
              choices=("float64", "float32")),
    ]

    @classmethod
    def recipe_revision(cls, params: dict) -> str | None:
        return _cuda_subsample_revision(params)

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.accel import resolve_workers
        from karak.clustering.tiling import run_tiled_hdbscan

        # The tiled path is host code (tile bookkeeping, registry merge,
        # process pool); per-tile cuML calls move each tile to the device.
        features = inputs["features"].to("cpu")
        cube = inputs["cube"].to("cpu")
        raw_labels, _, probabilities, tile_results, phase_registry = (
            run_tiled_hdbscan(
                features.features,
                features.mineral_indices,
                features.image_shape,
                cube.pixels,
                hdbscan_config(params),
                tiled_config(params),
                noise_reassign_k=None,
                skip_knn=True,
                workers=resolve_workers(self.workers),
                device=params["device"],
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
