"""Rare-phase stage: recluster unassigned pixels (Pass 2)."""

from __future__ import annotations

import copy

from karak.core_params import RarePhaseConfig
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import LabelState, Space, TiledArtifacts
from karak.stages.registry import register


def rare_phase_config(params: dict) -> RarePhaseConfig:
    return RarePhaseConfig(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"] or None,
        subsample_n=params["subsample_n"] or None,
        merge_threshold=params["merge_threshold"],
        accumulate=params["accumulate"],
    )


@register
class RarePhaseStage(Stage):
    id = "rare_phase"
    label = "Rare phases"
    description = (
        "Recluster still-unassigned pixels with more sensitive HDBSCAN "
        "parameters; novel clusters join the phase registry. Presence of "
        "this stage in a flow is what enables the two-pass workflow."
    )
    INPUTS = [
        Port("labels", space=LabelState.RAW,
             help="Pass-1 labels; -1 pixels are the recluster candidates"),
        Port("features", help="PCA features for all mineral pixels"),
        Port("cube", space=Space.DENOISED,
             help="denoised cube; fingerprints rare clusters for matching"),
        Port("tiles", help="phase registry from the tiled pass"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.RAW,
             help="labels with rare phases assigned; residual -1 remains"),
        Port("tiles", help="registry extended with the new rare phases"),
    ]
    PARAMS = [
        Param("min_cluster_size", "int", 50, "Min cluster size", min=1,
              unit="px"),
        Param("min_samples", "int", 0, "Min samples",
              "0 = defaults to min_cluster_size", min=0),
        Param("subsample_n", "int", 500_000, "Subsample N",
              "Max unassigned pixels to fit on; 0 = all", min=0),
        Param("merge_threshold", "float", 0.92, "Merge threshold",
              "Cosine similarity a rare cluster needs to join an existing "
              "registry phase; usually the tiled node's merge_threshold",
              min=0.5, max=1.0),
        Param("random_state", "int", 42, "Random seed"),
        Param("accumulate", "enum", "float64", "Accumulate",
              "Precision of the rare-cluster fingerprint sums: float64 is "
              "accurate; float32 reproduces the published baseline",
              choices=("float64", "float32")),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.tiling import recluster_unassigned

        labels, features = inputs["labels"], inputs["features"]
        tiles = inputs["tiles"]
        # recluster_unassigned extends the registry in place — work on a copy
        # so the input payload stays immutable.
        registry = copy.deepcopy(list(tiles.phase_registry))
        updated_labels, registry, _, _ = recluster_unassigned(
            features.features,
            labels.labels,
            inputs["cube"].pixels,
            features.mineral_indices,
            registry,
            rare_phase_config(params),
            random_state=params["random_state"],
        )
        return {
            "labels": labels.replace(labels=updated_labels),
            "tiles": tiles.replace(phase_registry=tuple(registry)),
        }
