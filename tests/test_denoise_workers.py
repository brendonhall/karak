"""Parity: parallel denoising is byte-identical to serial."""

from __future__ import annotations

import numpy as np

from conftest import make_synthetic_scene
from karak.preprocessing.denoise import (
    anisotropic_denoise_cube,
    bilateral_denoise_cube,
)


def _cube_and_mask():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    return cube, mask


def test_bilateral_parallel_parity():
    cube, mask = _cube_and_mask()
    serial = bilateral_denoise_cube(cube, mask, workers=1)
    parallel = bilateral_denoise_cube(cube, mask, workers=3)
    assert np.array_equal(serial, parallel)


def test_anisotropic_parallel_parity():
    cube, mask = _cube_and_mask()
    serial = anisotropic_denoise_cube(cube, mask, workers=1)
    parallel = anisotropic_denoise_cube(cube, mask, workers=3)
    assert np.array_equal(serial, parallel)
