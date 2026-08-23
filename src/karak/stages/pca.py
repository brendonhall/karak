"""PCA stage: fit + project mineral pixels, with auto component selection."""

from __future__ import annotations

from karak.config import PCAConfig
from karak.clustering.pca import auto_n_components, fit_pca, select_components
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import PCAFeatures, Space
from karak.stages.registry import register


@register
class PCAStage(Stage):
    id = "pca"
    label = "PCA"
    description = (
        "Fit PCA on mineral pixels of the normalized cube and project them "
        "into reduced feature space. n_components=0 auto-selects the first "
        "component count reaching the cumulative-variance threshold."
    )
    INPUTS = [Port("cube", space=Space.NORMALIZED), Port("masks")]
    OUTPUTS = [Port("features")]
    PARAMS = [
        Param("n_components", "int", 0, "Components",
              "Number of components to keep; 0 = auto from variance", min=0),
        Param("variance_threshold", "float", 0.95, "Variance threshold",
              "Cumulative explained variance target for auto selection",
              min=0.0, max=1.0),
        Param("min_components", "int", 5, "Min components",
              "Floor for auto selection", min=1),
        Param("subsample_fraction", "float", 0.0, "Subsample fraction",
              "Fraction of mineral pixels used for fitting; 0 = all",
              min=0.0, max=1.0),
        Param("random_state", "int", 42, "Random seed"),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        cube, masks = inputs["cube"], inputs["masks"]
        n_components = params["n_components"] or None
        config = PCAConfig(
            n_components=n_components,
            subsample_fraction=params["subsample_fraction"] or None,
            random_state=params["random_state"],
        )
        model, features, mineral_indices = fit_pca(
            cube.pixels, masks.mineral_mask, config
        )
        evr = model.explained_variance_ratio_
        if n_components is not None:
            n_keep = n_components
        else:
            n_keep = auto_n_components(
                evr,
                variance_threshold=params["variance_threshold"],
                min_components=params["min_components"],
            )
        n_keep = min(n_keep, features.shape[1])
        return {
            "features": PCAFeatures(
                features=select_components(features, n_keep),
                mineral_indices=mineral_indices,
                image_shape=cube.pixels.shape[:2],
                explained_variance_ratio=evr,
                n_kept=n_keep,
            )
        }
