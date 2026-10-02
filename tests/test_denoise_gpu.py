"""GPU bilateral denoise: exact-tier parity with the CPU path."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import denoise_cfg, make_synthetic_scene
from karak.accel import cuda_available
from karak.preprocessing.denoise import bilateral_denoise_cube, denoise_cube
from karak.stages.base import StageError


def test_cuda_without_gpu_raises(monkeypatch):
    import karak.accel as accel

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    with pytest.raises(StageError):
        bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0, device="cuda")


def test_cuda_without_cucim_bilateral_is_a_clear_stage_error(monkeypatch):
    """cucim (25.6 through 26.8) ships no denoise_bilateral; the CUDA path
    must say so instead of leaking an ImportError."""
    import sys
    import types

    import karak.accel as accel

    monkeypatch.setattr(accel, "get_array_module", lambda device: np)
    for name in ("cucim", "cucim.skimage", "cucim.skimage.restoration"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    with pytest.raises(StageError, match="cucim.*denoise_bilateral.*device=cpu"):
        bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0, device="cuda")


def test_anisotropic_cuda_unsupported():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    config = denoise_cfg(
        method="anisotropic_diffusion",
        sigma_color=None,
        sigma_spatial=1.0,
        niter=10,
        kappa=50.0,
        gamma=0.1,
        option=2,
    )
    with pytest.raises(StageError, match="anisotropic"):
        denoise_cube(cube, mask, config, device="cuda")


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.xfail(strict=True, raises=StageError,
                   reason="cucim has no denoise_bilateral; a CuPy port is a follow-up")
def test_bilateral_gpu_matches_cpu():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cpu = bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0)
    gpu = bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0, device="cuda")
    assert gpu.dtype == cpu.dtype
    assert np.allclose(cpu, gpu, atol=1e-5)
