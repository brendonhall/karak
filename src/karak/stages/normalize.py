"""Normalize stage: per-channel z-score over mineral pixels."""

from __future__ import annotations

from karak.preprocessing.compositional import zscore_normalize
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import Space
from karak.stages.registry import register


@register
class NormalizeStage(Stage):
    id = "normalize"
    label = "Normalize"
    description = (
        "Per-channel z-score normalization over mineral pixels; "
        "non-mineral pixels are set to 0."
    )
    INPUTS = [Port("cube", space=Space.DENOISED), Port("masks")]
    OUTPUTS = [Port("cube", space=Space.NORMALIZED)]
    PARAMS = [
        Param("method", "enum", "zscore", "Method", choices=("zscore",)),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        cube = inputs["cube"]
        normalized, means, stds = zscore_normalize(
            cube.pixels, inputs["masks"].mineral_mask
        )
        return {
            "cube": cube.replace(
                pixels=normalized,
                space=Space.NORMALIZED,
                means=means,
                stds=stds,
            )
        }
