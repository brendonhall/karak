"""Mask stage: mineral-pixel mask from the raw cube (+ optional polygon)."""

from __future__ import annotations

from karak.io.masks import (
    compute_mask_statistics,
    create_mineral_mask,
    load_valid_mask,
)
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import MaskSet, Space
from karak.stages.registry import register


@register
class MaskStage(Stage):
    id = "mask"
    label = "Mineral mask"
    description = (
        "Boolean mineral-pixel mask: all-zero pixels are background; an "
        "optional napari polygon CSV restricts the valid sample region."
    )
    INPUTS = [Port("cube", space=Space.RAW)]
    OUTPUTS = [Port("masks", help="mineral + valid masks with statistics")]
    PARAMS = [
        Param("min_object_size", "int", 100, "Min object size", min=0,
              help="Remove connected components smaller than this", unit="px"),
        Param("valid_mask_path", "str", None, "Valid-region CSV",
              "napari shapes CSV of the sample boundary polygon "
              "(coordinates in original, pre-downsample image space)"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        cube = inputs["cube"]
        valid = None
        if params["valid_mask_path"] is not None:
            valid = load_valid_mask(
                params["valid_mask_path"],
                cube.pixels.shape[:2],
                downsample_factor=cube.downsample_factor,
                header_trim_px=cube.header_trim_px,
                left_trim_px=cube.left_trim_px,
            )
        mineral = create_mineral_mask(
            cube.pixels, valid, min_object_size=params["min_object_size"]
        )
        return {
            "masks": MaskSet(
                mineral_mask=mineral,
                valid_mask=valid,
                stats=compute_mask_statistics(mineral, valid),
            )
        }
