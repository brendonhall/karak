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


def _blobs(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, 5)) * 6
    return (centers[rng.integers(0, 4, n)]
            + rng.normal(size=(n, 5))).astype(np.float32)


def test_subsample_indices_are_one_seeded_draw():
    from karak.clustering.hdbscan_cluster import subsample_indices

    idx = subsample_indices(1000, 100, 42)
    expected = np.random.default_rng(42).choice(1000, size=100, replace=False)
    np.testing.assert_array_equal(idx, expected)
    assert subsample_indices(1000, None, 42) is None
    assert subsample_indices(1000, 1000, 42) is None


def test_predict_batch_rows_scale_with_memory_and_min_samples():
    from karak.clustering.hdbscan_cluster import predict_batch_rows

    free = 640_000_000                     # a quarter is 160 MB
    assert predict_batch_rows(1000, free) == 10_000
    assert predict_batch_rows(100, free) == 100_000
    assert predict_batch_rows(1000, 0) == 1024


def test_cpu_subsample_path_uses_the_shared_indices(monkeypatch):
    import karak.clustering.hdbscan_cluster as hc

    seen = {}
    real = hc.subsample_indices

    def spy(n, subsample_n, random_state):
        seen["idx"] = real(n, subsample_n, random_state)
        return seen["idx"]

    monkeypatch.setattr(hc, "subsample_indices", spy)
    features = _features()
    labels, _, clusterer = run_hdbscan(
        features, hdbscan_cfg(min_cluster_size=50, subsample_n=400), device="cpu")
    assert seen["idx"].size == 400
    np.testing.assert_array_equal(clusterer._raw_data, features[seen["idx"]])
    assert labels.shape == (1000,)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_subsample_fits_the_same_pixels_as_the_cpu(monkeypatch):
    import cupy as cp
    import karak.clustering.hdbscan_cluster as hc

    fitted = {}
    from cuml.cluster import HDBSCAN as Real

    class Spy(Real):
        def fit(self, X, *args, **kwargs):
            fitted["X"] = cp.asnumpy(X)
            return super().fit(X, *args, **kwargs)

    import cuml.cluster
    monkeypatch.setattr(cuml.cluster, "HDBSCAN", Spy)
    features = _blobs()
    cfg = hdbscan_cfg(min_cluster_size=200, subsample_n=4000, random_state=7)
    labels, probs, _ = run_hdbscan(features, cfg, device="cuda")
    idx = hc.subsample_indices(features.shape[0], 4000, 7)
    np.testing.assert_array_equal(fitted["X"], features[idx])
    assert labels.shape == (20_000,) and labels.dtype == cp.int32
    assert probs.shape == (20_000,) and probs.dtype == cp.float32


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_predict_does_not_depend_on_the_batch_size():
    import cupy as cp
    from cuml.cluster import HDBSCAN
    from karak.clustering.hdbscan_cluster import _cuml_predict

    features = cp.asarray(_blobs())
    model = HDBSCAN(min_cluster_size=200, min_samples=50, prediction_data=True)
    model.fit(features[:4000])
    a_lab, a_prob = _cuml_predict(model, features, 1024)
    b_lab, b_prob = _cuml_predict(model, features, 20_000)
    cp.testing.assert_array_equal(a_lab, b_lab)
    cp.testing.assert_allclose(a_prob, b_prob, rtol=1e-6)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_subsample_repeats_identically():
    import cupy as cp

    cfg = hdbscan_cfg(min_cluster_size=200, subsample_n=4000)
    a, _, _ = run_hdbscan(_blobs(), cfg, device="cuda")
    b, _, _ = run_hdbscan(_blobs(), cfg, device="cuda")
    cp.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_subsample_agrees_with_the_cpu():
    import cupy as cp
    from sklearn.metrics import adjusted_rand_score

    features = _blobs()
    cfg = hdbscan_cfg(min_cluster_size=200, subsample_n=4000)
    gpu, _, _ = run_hdbscan(features, cfg, device="cuda")
    cpu, _, _ = run_hdbscan(features, cfg, device="cpu")
    gpu = cp.asnumpy(gpu)
    assert len(set(gpu.tolist()) - {-1}) == len(set(cpu.tolist()) - {-1}) == 4
    assert adjusted_rand_score(cpu, gpu) > 0.95


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
def test_cuml_out_of_memory_is_a_stage_error(monkeypatch):
    import cuml.cluster

    class Boom:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            raise RuntimeError("std::bad_alloc: out_of_memory: CUDA error")

    monkeypatch.setattr(cuml.cluster, "HDBSCAN", Boom)
    with pytest.raises(StageError, match="subsample_n"):
        run_hdbscan(_blobs(2000), hdbscan_cfg(min_cluster_size=100, subsample_n=500),
                    device="cuda")
