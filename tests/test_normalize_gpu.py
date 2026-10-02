"""z-score normalization on the device: exact tier."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.accel import cuda_available
from karak.preprocessing.compositional import zscore_normalize
from karak.stages.normalize import NormalizeStage


def test_normalize_declares_a_device_param():
    names = [p.name for p in NormalizeStage.PARAMS]
    assert names == ["method", "accumulate", "device"]
    device = NormalizeStage.PARAMS[2]
    assert device.default == "cpu" and device.choices == ("cpu", "cuda")


def test_means_and_stds_are_host_arrays_on_cpu():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    normalized, means, stds = zscore_normalize(cube, mask, accumulate="float64")
    assert isinstance(means, np.ndarray) and isinstance(stds, np.ndarray)
    assert normalized.dtype == np.float32


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_device_zscore_matches_cpu():
    import cupy as cp

    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    cpu, means, stds = zscore_normalize(cube, mask, accumulate="float64")
    gpu, gmeans, gstds = zscore_normalize(cp.asarray(cube), cp.asarray(mask), accumulate="float64")
    assert isinstance(gpu, cp.ndarray) and gpu.dtype == cp.float32
    assert isinstance(gmeans, np.ndarray) and isinstance(gstds, np.ndarray)
    np.testing.assert_allclose(cp.asnumpy(gpu), cpu, atol=1e-4)
    np.testing.assert_allclose(gmeans, means, atol=1e-6)
    np.testing.assert_allclose(gstds, stds, atol=1e-6)
    assert not cp.asnumpy(gpu)[~mask].any()
