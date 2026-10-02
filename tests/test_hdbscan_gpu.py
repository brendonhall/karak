"""cuML HDBSCAN device path: loose tier (shape and sanity only)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import hdbscan_cfg

from karak.accel import cuda_available
from karak.clustering.hdbscan_cluster import run_hdbscan
from karak.stages.base import StageError


def _features():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (500, 3))
    b = rng.normal(5, 0.1, (500, 3))
    return np.vstack([a, b]).astype(np.float32)


def test_cuda_without_gpu_raises(monkeypatch):
    import karak.accel as accel

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    with pytest.raises(StageError):
        run_hdbscan(_features(), hdbscan_cfg(min_cluster_size=100),
                    device="cuda")


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_hdbscan_shapes_and_sanity():
    features = _features()
    import cupy as cp

    labels, probs, _ = run_hdbscan(
        features, hdbscan_cfg(min_cluster_size=100), device="cuda",
    )
    labels, probs = cp.asnumpy(labels), cp.asnumpy(probs)
    assert labels.shape == (1000,)
    assert labels.dtype == np.int32
    assert probs.shape == (1000,)
    assert probs.dtype == np.float32
    assert len(set(labels.tolist()) - {-1}) == 2


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_device_features_in_device_labels_out():
    import cupy as cp

    labels, probs, _ = run_hdbscan(
        cp.asarray(_features()), hdbscan_cfg(min_cluster_size=100), device="cuda",
    )
    assert isinstance(labels, cp.ndarray) and labels.dtype == cp.int32
    assert isinstance(probs, cp.ndarray) and probs.dtype == cp.float32
    assert len(set(cp.asnumpy(labels).tolist()) - {-1}) == 2


def test_cpu_path_rejects_device_features():
    class FakeDeviceArray:
        __module__ = "cupy"
        shape = (10, 3)

    with pytest.raises(StageError, match="device array"):
        run_hdbscan(FakeDeviceArray(), hdbscan_cfg(min_cluster_size=5), device="cpu")
