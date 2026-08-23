"""Refine stage: split composite phases (olivine extraction + GMM split)."""

from __future__ import annotations

from karak.config import (
    GMMSplitConfig,
    OlivineExtractionConfig,
    RefinementConfig,
)
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register


@register
class RefineStage(Stage):
    id = "refine"
    label = "Phase refinement"
    description = (
        "Split a composite phase: threshold-based olivine extraction, then "
        "a GMM split of the remaining target-phase pixels."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED),
        Port("cube", space=Space.DENOISED),
        Port("bse"),
    ]
    OUTPUTS = [Port("labels", space=LabelState.CLEANED)]
    PARAMS = [
        Param("target_phase", "int", 2, "Target phase",
              "Cluster label of the phase to refine"),
        Param("olivine_enabled", "bool", False, "Olivine extraction"),
        Param("olivine_fe_threshold", "float", 0.6, "Olivine Fe threshold",
              min=0.0, max=1.0),
        Param("olivine_ca_threshold", "float", 0.10, "Olivine Ca threshold",
              min=0.0, max=1.0),
        Param("gmm_enabled", "bool", False, "GMM split"),
        Param("gmm_n_components", "int", 2, "GMM components", min=2),
        Param("gmm_features", "str", "Ca,Mg,Fe-K,BSE", "GMM features",
              "Comma-separated channel names; 'BSE' adds backscatter"),
        Param("gmm_bse_weight", "float", 1.0, "BSE weight", min=0.0),
        Param("gmm_subsample_n", "int", 500_000, "GMM subsample N",
              "Max pixels to fit GMM on; 0 = all", min=0),
        Param("random_state", "int", 42, "Random seed"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.refinement import refine_phases

        labels, cube = inputs["labels"], inputs["cube"]
        config = RefinementConfig(
            enabled=True,
            target_phase=params["target_phase"],
            olivine=OlivineExtractionConfig(
                enabled=params["olivine_enabled"],
                fe_threshold=params["olivine_fe_threshold"],
                ca_threshold=params["olivine_ca_threshold"],
            ),
            gmm_split=GMMSplitConfig(
                enabled=params["gmm_enabled"],
                n_components=params["gmm_n_components"],
                features=[
                    f.strip() for f in params["gmm_features"].split(",")
                    if f.strip()
                ],
                bse_weight=params["gmm_bse_weight"],
                subsample_n=params["gmm_subsample_n"] or None,
                random_state=params["random_state"],
            ),
        )
        refined = refine_phases(
            labels.labels.copy(),
            cube.pixels,
            inputs["bse"].pixels,
            labels.mineral_indices,
            list(cube.element_names),
            config,
        )
        return {"labels": labels.replace(labels=refined)}
