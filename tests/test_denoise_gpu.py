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
def test_bilateral_gpu_matches_cpu():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cpu = bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0)
    gpu = bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0, device="cuda")
    assert gpu.dtype == cpu.dtype
    assert np.allclose(cpu, gpu, atol=1e-5)
