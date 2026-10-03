"""noise_assign on the device: CuPy brute-force k-NN with sklearn's
distance-weighted vote. GPU tests run only where CUDA is available."""

from __future__ import annotations

import numpy as np
import pytest

from karak.accel import cuda_available
from karak.clustering.noise_assign import assign_noise_pixels
from karak.errors import StageError

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="no CUDA")


def _scene(n=20_000, d=9, n_classes=5, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, d)) * 3
    truth = rng.integers(0, n_classes, n)
    features = (centers[truth] + rng.normal(size=(n, d))).astype(np.float32)
    labels = truth.astype(np.int32)
    labels[rng.random(n) < noise] = -1
    return features, labels


def test_cuda_without_a_gpu_is_a_stage_error(monkeypatch):
    import karak.accel as accel

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    features, labels = _scene(200)
    with pytest.raises(StageError):
        assign_noise_pixels(features, labels, 5, device="cuda")


def test_cpu_rejects_device_arrays():
    class FakeDeviceArray:
        __module__ = "cupy"

    with pytest.raises(StageError, match="device array"):
        assign_noise_pixels(FakeDeviceArray(), np.zeros(3, np.int32), 5, device="cpu")


def test_stage_passes_the_device_to_the_core(monkeypatch):
    import karak.stages.noise as noise
    from karak.stages.payloads import Labels, LabelState, PCAFeatures

    seen = {}

    def fake(features, labels, k, *, device):
        seen.update(k=k, device=device)
        return labels

    monkeypatch.setattr(noise, "assign_noise_pixels", fake)
    features, labels = _scene(100)
    idx = np.zeros((100, 2), np.int32)
    noise.NoiseAssignStage().run(
        {"labels": Labels(labels, None, idx, (1, 100), LabelState.RAW),
         "features": PCAFeatures(features, idx, (1, 100), np.ones(9), 9)},
        {"k": 7, "device": "cuda"},
    )
    assert seen == {"k": 7, "device": "cuda"}


@needs_cuda
@pytest.mark.parametrize("d,k,n_classes", [(9, 5, 5), (3, 5, 2), (20, 12, 7), (9, 1, 5)])
def test_cuda_labels_match_the_cpu(d, k, n_classes):
    import cupy as cp

    features, labels = _scene(d=d, n_classes=n_classes)
    cpu = assign_noise_pixels(features, labels, k, device="cpu")
    gpu = assign_noise_pixels(features, labels, k, device="cuda")
    assert isinstance(gpu, cp.ndarray) and gpu.dtype == cp.int32
    np.testing.assert_array_equal(cp.asnumpy(gpu), cpu)


@needs_cuda
@pytest.mark.parametrize("d,k", [(9, 40), (49, 5), (200, 5)])
def test_cuda_large_k_or_many_features_match_the_cpu(d, k):
    import cupy as cp

    features, labels = _scene(n=8000, d=d)
    cpu = assign_noise_pixels(features, labels, k, device="cpu")
    gpu = assign_noise_pixels(features, labels, k, device="cuda")
    np.testing.assert_array_equal(cp.asnumpy(gpu), cpu)


@needs_cuda
@pytest.mark.parametrize("d,k", [(9, 40), (49, 5)])
def test_cuda_keeps_an_exact_match_far_from_the_origin(d, k):
    # review regression: 500 class-0 points within ~0.001 of a query at 5.0
    # and one class-1 point equal to it. A float32 search in the expanded
    # form |x|^2 - 2x.y + |y|^2 cancels here and loses the exact match.
    import cupy as cp

    rng = np.random.default_rng(15)
    q = np.full((1, d), 5, np.float32)
    x = np.vstack([(q + rng.normal(0, 0.001, (500, d))).astype(np.float32), q, q])
    y = np.zeros(502, np.int32)
    y[-2:] = [1, -1]
    cpu = assign_noise_pixels(x, y, k, device="cpu")
    gpu = cp.asnumpy(assign_noise_pixels(x, y, k, device="cuda"))
    assert cpu[-1] == gpu[-1] == 1


@needs_cuda
def test_cuda_refuses_too_many_candidates():
    features, labels = _scene(n=3000)
    with pytest.raises(StageError, match="candidates"):
        assign_noise_pixels(features, labels, 2000, device="cuda")


@needs_cuda
def test_cuda_keeps_clean_labels_and_device_inputs():
    import cupy as cp

    features, labels = _scene()
    gpu = assign_noise_pixels(cp.asarray(features), cp.asarray(labels), 5,
                              device="cuda")
    gpu = cp.asnumpy(gpu)
    clean = labels >= 0
    np.testing.assert_array_equal(gpu[clean], labels[clean])
    assert (gpu >= 0).all()


@needs_cuda
def test_cuda_zero_distance_neighbor_decides_the_vote():
    import cupy as cp

    # the query sits on a class-1 point; four class-0 points are close by
    ref = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], np.float32)
    ref = np.vstack([ref + [10, 10], [[10.0, 10.0]]]).astype(np.float32)
    ref_labels = np.array([0, 0, 0, 0, 0, 1], np.int32)
    ref[0] = [10.5, 10.5]                           # keep (10, 10) unique
    features = np.vstack([ref, [[10.0, 10.0]]]).astype(np.float32)
    labels = np.append(ref_labels, -1).astype(np.int32)
    cpu = assign_noise_pixels(features, labels, 5, device="cpu")
    gpu = cp.asnumpy(assign_noise_pixels(features, labels, 5, device="cuda"))
    assert cpu[-1] == gpu[-1] == 1


@needs_cuda
def test_cuda_tied_vote_goes_to_the_lowest_label():
    import cupy as cp

    # two neighbors at the same distance, one per class: the vote ties
    features = np.array([[1, 0], [-1, 0], [0, 0]], np.float32)
    labels = np.array([3, 2, -1], np.int32)
    cpu = assign_noise_pixels(features, labels, 2, device="cpu")
    gpu = cp.asnumpy(assign_noise_pixels(features, labels, 2, device="cuda"))
    assert cpu[-1] == gpu[-1] == 2


@needs_cuda
def test_cuda_batches_do_not_change_the_result(monkeypatch):
    import cupy as cp
    from karak.clustering import knn_gpu

    features, labels = _scene()
    whole = cp.asnumpy(assign_noise_pixels(features, labels, 5, device="cuda"))
    monkeypatch.setattr(knn_gpu, "batch_rows", lambda *a: 1024)
    batched = cp.asnumpy(assign_noise_pixels(features, labels, 5, device="cuda"))
    np.testing.assert_array_equal(whole, batched)


def test_batch_rows_scale_with_memory():
    from karak.clustering.knn_gpu import batch_rows

    assert batch_rows(9, 8, 5, 0) == 1024
    small = batch_rows(9, 8, 5, 1 << 30)
    assert batch_rows(9, 8, 5, 4 << 30) == pytest.approx(4 * small, rel=1e-3)
