"""rare_phase passes --workers to its pass-2 HDBSCAN, with identical labels."""

from __future__ import annotations

import copy
import os

import numpy as np

from conftest import rare_cfg
import karak.clustering.hdbscan_cluster as hc
from karak.clustering.tiling import recluster_unassigned


def _scene(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 4)) * 6
    features = (centers[rng.integers(0, 3, n)] + rng.normal(size=(n, 4))).astype(np.float32)
    side = int(np.ceil(np.sqrt(n)))
    rows, cols = np.divmod(np.arange(n), side)
    indices = np.stack([rows, cols], axis=1).astype(np.int32)
    cube = rng.random((side, side, 3)).astype(np.float32)
    return features, np.full(n, -1, np.int32), cube, indices


def test_pass_two_gives_the_same_labels_for_any_worker_count(monkeypatch):
    monkeypatch.setattr(hc, "_MIN_ROWS_PER_PREDICT_WORKER", 500)
    features, labels, cube, indices = _scene()
    cfg = rare_cfg(min_cluster_size=100, min_samples=20, subsample_n=1500)
    one = recluster_unassigned(features, labels, cube, indices, [], cfg,
                               random_state=0, workers=1)
    four = recluster_unassigned(features, labels, cube, indices, [], cfg,
                                random_state=0, workers=4)
    np.testing.assert_array_equal(one[0], four[0])
    assert [e.global_id for e in one[1]] == [e.global_id for e in four[1]]
    for a, b in zip(one[1], four[1]):
        np.testing.assert_array_equal(a.mean_fingerprint, b.mean_fingerprint)


def test_rare_phase_stage_passes_the_workers(monkeypatch):
    import karak.clustering.tiling as tiling
    from karak.stages.payloads import Labels, LabelState, PCAFeatures, TiledArtifacts, ElementCube, Space
    from karak.stages.rare_phase import RarePhaseStage

    seen = []

    def fake(features, labels, cube, indices, registry, rare, *, random_state, workers,
             device):
        seen.append(workers)
        return labels, registry, 0, int((labels == -1).sum())

    monkeypatch.setattr(tiling, "recluster_unassigned", fake)
    features, labels, cube, indices = _scene(100)
    inputs = {
        "labels": Labels(labels, None, indices, cube.shape[:2], LabelState.RAW),
        "features": PCAFeatures(features, indices, cube.shape[:2], np.ones(4), 4),
        "cube": ElementCube(pixels=cube, element_names=("a", "b", "c"), space=Space.DENOISED),
        "tiles": TiledArtifacts(tile_results=(), phase_registry=(), tile_size=32),
    }
    for workers in (None, 3, 0):
        stage = RarePhaseStage()
        stage.workers = workers
        stage.run(copy.copy(inputs))
    assert seen == [1, 3, os.cpu_count() or 1]
