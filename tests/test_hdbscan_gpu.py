"""cuML HDBSCAN device path: loose tier (shape and sanity only)."""

from __future__ import annotations

import numpy as np
import pytest

from karak.accel import cuda_available
from karak.clustering.hdbscan_cluster import run_hdbscan
from karak.config import HDBSCANConfig
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
        run_hdbscan(_features(), HDBSCANConfig(min_cluster_size=100),
                    device="cuda")


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_hdbscan_shapes_and_sanity():
    features = _features()
    labels, probs, _ = run_hdbscan(
        features, HDBSCANConfig(min_cluster_size=100), device="cuda",
    )
    assert labels.shape == (1000,)
    assert labels.dtype == np.int32
    assert probs.shape == (1000,)
    assert probs.dtype == np.float32
    assert len(set(labels.tolist()) - {-1}) == 2
