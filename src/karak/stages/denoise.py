"""Denoise stage: edge-aware filtering of the raw cube."""

from __future__ import annotations

from karak.core_params import DenoiseConfig
from karak.preprocessing.denoise import denoise_cube
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import Space
from karak.stages.registry import register


def denoise_config(params: dict) -> DenoiseConfig:
    return DenoiseConfig(
        method=params["method"],
        sigma_color=params["sigma_color"],
        sigma_spatial=params["sigma_spatial"],
        niter=params["niter"],
        kappa=params["kappa"],
        gamma=params["gamma"],
        option=params["option"],
    )


@register
class DenoiseStage(Stage):
    id = "denoise"
    label = "Denoise"
    description = (
        "Edge-aware denoising (bilateral, symmetric or joint bilateral, or "
        "Perona-Malik anisotropic diffusion) applied per channel on raw "
        "intensities."
    )
    INPUTS = [
        Port("cube", space=Space.RAW, help="(H, W, C) raw element cube"),
        Port("masks", help="mineral mask restricts smoothing to sample pixels"),
        Port("bse", required=False,
             help="BSE image: the range guide for method=joint_bilateral_bse"),
    ]
    OUTPUTS = [
        Port("cube", space=Space.DENOISED,
             help="(H, W, C) denoised cube, same channels and geometry"),
    ]
    PARAMS = [
        Param("method", "enum", "bilateral", "Method",
              "bilateral = scikit-image's filter (the published baseline, "
              "off-centre spatial table); bilateral_sym = symmetric kernel; "
              "joint_bilateral_total / joint_bilateral_bse = range weight "
              "from the summed channels / the BSE image (bse port)",
              choices=("bilateral", "bilateral_sym", "joint_bilateral_total",
                       "joint_bilateral_bse", "anisotropic_diffusion")),
        Param("sigma_color", "float", None, "Color sigma",
              "Bilateral range sigma (None = std of the channel, or of the "
              "guide for the joint methods)", min=0.0),
        Param("sigma_spatial", "float", 1.0, "Spatial sigma", min=0.0),
        Param("niter", "int", 10, "Iterations", min=1),
        Param("kappa", "float", 50.0, "Kappa",
              "Conductance coefficient for diffusion", min=0.0),
        Param("gamma", "float", 0.1, "Gamma",
              "Diffusion speed (0-0.25 stable)", min=0.0, max=0.25),
        Param("option", "int", 2, "Perona-Malik option", choices=(1, 2)),
        Param("device", "str", "cpu", "Device",
              "cpu or cuda (bilateral methods on the GPU via karak's CuPy "
              "kernel; needs karak[cuda]). Results match cpu within 1e-5.",
              choices=("cpu", "cuda")),
    ]

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.accel import resolve_workers

        cube = inputs["cube"]
        config = denoise_config(params)
        reporter, node_id, names = self.reporter, self.node_id, cube.element_names

        def on_channel(done: int, total: int, index: int) -> None:
            if reporter is not None:
                reporter.progress(node_id, done, total, names[index])

        bse = inputs.get("bse")
        denoised = denoise_cube(
            cube.pixels, inputs["masks"].mineral_mask, config,
            workers=resolve_workers(self.workers),
            device=params["device"],
            on_channel=on_channel,
            guide=None if bse is None else bse.pixels,
        )
        return {"cube": cube.replace(pixels=denoised, space=Space.DENOISED)}
