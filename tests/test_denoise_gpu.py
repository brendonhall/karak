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
@pytest.mark.parametrize("method", ["bilateral", "bilateral_sym",
                                    "joint_bilateral_total", "joint_bilateral_bse"])
def test_bilateral_gpu_matches_cpu(method):
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    guide = np.random.default_rng(9).random(cube.shape[:2], dtype=np.float32)
    cfg = denoise_cfg(method=method, sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    cpu = denoise_cube(cube, mask, cfg, guide=guide)
    from karak.accel import to_numpy

    gpu = to_numpy(denoise_cube(cube, mask, cfg, guide=guide, device="cuda"))
    assert gpu.dtype == cpu.dtype
    np.testing.assert_allclose(cpu, gpu, atol=1e-5)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("method", ["bilateral", "bilateral_sym",
                                    "joint_bilateral_total", "joint_bilateral_bse"])
def test_device_inputs_stay_on_the_device_and_match_cpu(method):
    import cupy as cp

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    guide = np.random.default_rng(9).random(cube.shape[:2], dtype=np.float32)
    cfg = denoise_cfg(method=method, sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    cpu = denoise_cube(cube, mask, cfg, guide=guide)
    gpu = denoise_cube(cp.asarray(cube), cp.asarray(mask), cfg,
                       guide=cp.asarray(guide), device="cuda")
    assert isinstance(gpu, cp.ndarray)
    assert gpu.dtype == cp.float32
    np.testing.assert_allclose(cpu, cp.asnumpy(gpu), atol=1e-5)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_host_inputs_on_cuda_still_work_and_return_a_device_array():
    import cupy as cp

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cfg = denoise_cfg(method="bilateral", sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    out = denoise_cube(cube, mask, cfg, device="cuda")
    assert isinstance(out, cp.ndarray)
