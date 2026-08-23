"""Shared fixtures for the karak test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def synthetic_scene():
    """64x64x3 cube: background strip + two chemically distinct phases."""
    rng = np.random.default_rng(0)
    H = W = 64
    cube = np.zeros((H, W, 3), dtype=np.float32)

    # Phase A (columns 8..35): high channel 0
    cube[:, 8:36, 0] = 0.8
    cube[:, 8:36, 1] = 0.2
    cube[:, 8:36, 2] = 0.5
    # Phase B (columns 36..63): high channel 1
    cube[:, 36:, 0] = 0.2
    cube[:, 36:, 1] = 0.8
    cube[:, 36:, 2] = 0.4

    noise = rng.normal(0, 0.02, size=cube.shape).astype(np.float32)
    cube[:, 8:, :] = np.clip(cube[:, 8:, :] + noise[:, 8:, :], 0.01, 1.0)
    # Columns 0..7 stay exactly zero = background / epoxy
    return cube
