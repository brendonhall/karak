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
    INPUTS = [
        Port("cube", space=Space.DENOISED, help="(H, W, C) denoised cube"),
        Port("masks", help="statistics are computed over mineral pixels only"),
    ]
    OUTPUTS = [
        Port("cube", space=Space.NORMALIZED,
             help="z-scored cube; per-channel means/stds ride on the payload"),
    ]
    PARAMS = [
        Param("method", "enum", "zscore", "Method", choices=("zscore",)),
        Param("accumulate", "enum", "float64", "Accumulate",
              "Precision of the mean/std sums: float64 is accurate; float32 "
              "reproduces the published NWA 4587 baseline (std up to 3.6 % "
              "off on 12.5 M pixels)",
              choices=("float64", "float32")),
        Param("device", "str", "cpu", "Device",
              "cpu or cuda (the executor moves the inputs; needs karak[cuda])",
              choices=("cpu", "cuda")),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        cube = inputs["cube"]
        normalized, means, stds = zscore_normalize(
            cube.pixels, inputs["masks"].mineral_mask,
            accumulate=params["accumulate"],
        )
        return {
            "cube": cube.replace(
                pixels=normalized,
                space=Space.NORMALIZED,
                means=means,
                stds=stds,
            )
        }
