"""Denoise stage: edge-aware filtering of the raw cube."""

from __future__ import annotations

from karak.config import DenoiseConfig
from karak.preprocessing.denoise import denoise_cube
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import Space
from karak.stages.registry import register


@register
class DenoiseStage(Stage):
    id = "denoise"
    label = "Denoise"
    description = (
        "Edge-aware denoising (bilateral or Perona-Malik anisotropic "
        "diffusion) applied per channel on raw intensities."
    )
    INPUTS = [Port("cube", space=Space.RAW), Port("masks")]
    OUTPUTS = [Port("cube", space=Space.DENOISED)]
    PARAMS = [
        Param("method", "enum", "bilateral", "Method",
              choices=("bilateral", "anisotropic_diffusion")),
        Param("sigma_color", "float", None, "Color sigma",
              "Bilateral color sigma (None = auto from data range)", min=0.0),
        Param("sigma_spatial", "float", 1.0, "Spatial sigma", min=0.0),
        Param("niter", "int", 10, "Iterations", min=1),
        Param("kappa", "float", 50.0, "Kappa",
              "Conductance coefficient for diffusion", min=0.0),
        Param("gamma", "float", 0.1, "Gamma",
              "Diffusion speed (0-0.25 stable)", min=0.0, max=0.25),
        Param("option", "int", 2, "Perona-Malik option", choices=(1, 2)),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        cube = inputs["cube"]
        config = DenoiseConfig(
            method=params["method"],
            sigma_color=params["sigma_color"],
            sigma_spatial=params["sigma_spatial"],
            niter=params["niter"],
            kappa=params["kappa"],
            gamma=params["gamma"],
            option=params["option"],
        )
        denoised = denoise_cube(
            cube.pixels, inputs["masks"].mineral_mask, config
        )
        return {"cube": cube.replace(pixels=denoised, space=Space.DENOISED)}
