"""Fingerprints stage: per-cluster chemical signatures."""

from __future__ import annotations

from karak.identification.fingerprint import (
    compute_fingerprints,
    flag_similar_clusters,
)
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import Fingerprints, LabelState, Space
from karak.stages.registry import register


@register
class FingerprintsStage(Stage):
    id = "fingerprints"
    label = "Chemical fingerprints"
    description = (
        "Per-cluster mean/std element intensities from the denoised cube, "
        "with cosine-similar cluster pairs flagged for review."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED),
        Port("cube", space=Space.DENOISED),
    ]
    OUTPUTS = [Port("fingerprints")]
    PARAMS = [
        Param("similarity_threshold", "float", 0.95, "Similarity threshold",
              "Cosine similarity above which cluster pairs are flagged",
              min=0.0, max=1.0),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        labels, cube = inputs["labels"], inputs["cube"]
        data = compute_fingerprints(
            cube.pixels,
            labels.labels,
            labels.mineral_indices,
            list(cube.element_names),
        )
        pairs = flag_similar_clusters(
            data, threshold=params["similarity_threshold"]
        )
        return {"fingerprints": Fingerprints(data=data, similar_pairs=pairs)}
