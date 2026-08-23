"""Rare-phase stage: recluster unassigned pixels (Pass 2)."""

from __future__ import annotations

import copy

from karak.config import ClusterConfig, HDBSCANConfig, RarePhaseConfig
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import LabelState, Space, TiledArtifacts
from karak.stages.registry import register


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
        Port("labels", space=LabelState.RAW),
        Port("features"),
        Port("cube", space=Space.DENOISED),
        Port("tiles"),
    ]
    OUTPUTS = [Port("labels", space=LabelState.RAW), Port("tiles")]
    PARAMS = [
        Param("min_cluster_size", "int", 50, "Min cluster size", min=1,
              unit="px"),
        Param("min_samples", "int", 0, "Min samples",
              "0 = defaults to min_cluster_size", min=0),
        Param("subsample_n", "int", 500_000, "Subsample N",
              "Max unassigned pixels to fit on; 0 = all", min=0),
        Param("merge_threshold", "float", 0.0, "Merge threshold",
              "Cosine similarity vs existing registry; 0 = reuse the tiled "
              "merge threshold", min=0.0, max=1.0),
        Param("noise_reassign_k", "int", 5, "kNN k", min=1),
        Param("random_state", "int", 42, "Random seed"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.tiling import recluster_unassigned

        labels, features = inputs["labels"], inputs["features"]
        tiles = inputs["tiles"]
        config = ClusterConfig(
            strategy="tiled",
            hdbscan=HDBSCANConfig(
                noise_reassign_k=params["noise_reassign_k"],
                random_state=params["random_state"],
            ),
            rare_phase=RarePhaseConfig(
                enabled=True,
                min_cluster_size=params["min_cluster_size"],
                min_samples=params["min_samples"] or None,
                subsample_n=params["subsample_n"] or None,
                merge_threshold=params["merge_threshold"] or None,
            ),
        )
        # recluster_unassigned extends the registry in place — work on a copy
        # so the input payload stays immutable.
        registry = copy.deepcopy(list(tiles.phase_registry))
        updated_labels, registry, _, _ = recluster_unassigned(
            features.features,
            labels.labels,
            inputs["cube"].pixels,
            features.mineral_indices,
            registry,
            config,
        )
        return {
            "labels": labels.replace(labels=updated_labels),
            "tiles": tiles.replace(phase_registry=tuple(registry)),
        }
