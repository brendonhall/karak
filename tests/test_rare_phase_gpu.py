"""rare_phase on the device: the pass-2 HDBSCAN runs with cuML; the
registry merge stays on the host. GPU tests run only where CUDA is."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import rare_cfg
from karak.accel import cuda_available
from karak.clustering.tiling import recluster_unassigned
from karak.errors import StageError

needs_cuda = pytest.mark.skipif(not cuda_available(), reason="no CUDA")


def _scene(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, 5)) * 6
    truth = rng.integers(0, 4, n)
    features = (centers[truth] + rng.normal(size=(n, 5))).astype(np.float32)
    side = int(np.ceil(np.sqrt(n)))
    rows, cols = np.divmod(np.arange(n), side)
    indices = np.stack([rows, cols], axis=1).astype(np.int32)
    cube = np.zeros((side, side, 4), np.float32)
    cube[rows, cols, truth] = 1.0          # each blob has its own spectrum
    return features, np.full(n, -1, np.int32), cube, indices


def test_cuda_without_a_gpu_is_a_stage_error(monkeypatch):
    import karak.accel as accel

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    features, labels, cube, indices = _scene(2000)
    with pytest.raises(StageError):
        recluster_unassigned(features, labels, cube, indices, [],
                             rare_cfg(min_cluster_size=50, subsample_n=1000),
                             random_state=0, device="cuda")


def test_stage_passes_the_device(monkeypatch):
    import karak.clustering.tiling as tiling
    from karak.stages.payloads import (ElementCube, Labels, LabelState,
                                       PCAFeatures, Space, TiledArtifacts)
    from karak.stages.rare_phase import RarePhaseStage

    seen = []

    def fake(features, labels, cube, indices, registry, rare, *, random_state,
             workers, device, exclude_indices=None):
        seen.append(device)
        return labels, registry, 0, int((labels == -1).sum())

    monkeypatch.setattr(tiling, "recluster_unassigned", fake)
    features, labels, cube, indices = _scene(100)
    inputs = {
        "labels": Labels(labels, None, indices, cube.shape[:2], LabelState.RAW),
        "features": PCAFeatures(features, indices, cube.shape[:2], np.ones(5), 5),
        "cube": ElementCube(pixels=cube, element_names=tuple("abcd"), space=Space.DENOISED),
        "tiles": TiledArtifacts(tile_results=(), phase_registry=(), tile_size=32),
    }
    params = {**RarePhaseStage.template(), "device": "cuda"}
    RarePhaseStage().run(copy.copy(inputs), params)
    assert seen == ["cuda"]


@needs_cuda
def test_cuda_pass_two_agrees_with_the_cpu():
    from sklearn.metrics import adjusted_rand_score

    features, labels, cube, indices = _scene()
    cfg = rare_cfg(min_cluster_size=200, subsample_n=4000)
    cpu = recluster_unassigned(features, labels, cube, indices, [], cfg,
                               random_state=0, device="cpu")
    gpu = recluster_unassigned(features, labels, cube, indices, [], cfg,
                               random_state=0, device="cuda")
    assert isinstance(gpu[0], np.ndarray) and gpu[0].dtype == np.int32
    assert len(gpu[1]) == len(cpu[1]) == 4               # four new phases
    assert adjusted_rand_score(cpu[0], gpu[0]) > 0.95


@needs_cuda
def test_cuda_rare_phase_stage_takes_device_inputs():
    import cupy as cp
    from karak.stages.payloads import (ElementCube, Labels, LabelState,
                                       PCAFeatures, Space, TiledArtifacts)
    from karak.stages.rare_phase import RarePhaseStage

    features, labels, cube, indices = _scene()
    inputs = {
        "labels": Labels(labels, None, indices, cube.shape[:2], LabelState.RAW).to("cuda"),
        "features": PCAFeatures(features, indices, cube.shape[:2], np.ones(5), 5).to("cuda"),
        "cube": ElementCube(pixels=cube, element_names=tuple("abcd"),
                            space=Space.DENOISED).to("cuda"),
        "tiles": TiledArtifacts(tile_results=(), phase_registry=(), tile_size=32),
    }
    params = {**RarePhaseStage.template(), "device": "cuda",
              "min_cluster_size": 200, "subsample_n": 4000}
    out = RarePhaseStage().run(inputs, params)
    assert out["labels"].device == "cpu"
    assert len(out["tiles"].phase_registry) == 4
    assert (out["labels"].labels >= 0).mean() > 0.8
