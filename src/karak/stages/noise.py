"""Noise-assign stage: kNN reassignment of unlabeled pixels."""

from __future__ import annotations

from karak.clustering.noise_assign import assign_noise_pixels
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import LabelState
from karak.stages.registry import register


@register
class NoiseAssignStage(Stage):
    id = "noise_assign"
    label = "Noise reassignment"
    description = (
        "Assign every remaining -1 pixel to its nearest phase by "
        "distance-weighted k-NN voting in feature space."
    )
    INPUTS = [Port("labels", space=LabelState.RAW), Port("features")]
    OUTPUTS = [Port("labels", space=LabelState.CLEANED)]
    PARAMS = [
        Param("k", "int", 5, "Neighbors", min=1),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        labels = inputs["labels"]
        cleaned = assign_noise_pixels(
            inputs["features"].features, labels.labels, k=params["k"]
        )
        return {
            "labels": labels.replace(labels=cleaned, state=LabelState.CLEANED)
        }
