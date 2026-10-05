"""hdbscan_tiled warns that cuML can select different clusters than the cpu."""

from __future__ import annotations

import numpy as np

import karak.stages.cluster as cluster
from karak.stages.payloads import ElementCube, PCAFeatures, Space


class _Recorder:
    def __init__(self):
        self.logs = []

    def log(self, level, msg):
        self.logs.append((level, msg))


def _inputs():
    rng = np.random.default_rng(0)
    rows, cols = np.nonzero(np.ones((8, 8), bool))
    idx = np.stack([rows, cols], axis=1).astype(np.int32)
    features = rng.normal(size=(64, 2)).astype(np.float32)
    cube = ElementCube(pixels=rng.random((8, 8, 2)).astype(np.float32),
                       element_names=("a", "b"), space=Space.DENOISED)
    return {"features": PCAFeatures(features, idx, (8, 8), np.ones(2), 2), "cube": cube}


def _run(monkeypatch, device, reporter=True):
    seen = {}

    def fake(features, indices, shape, cube, hdb, tiled, **kwargs):
        seen["device"] = kwargs["device"]
        n = len(features)
        return (np.zeros(n, np.int32), None, np.zeros(n, np.float32), [], [])

    monkeypatch.setattr("karak.clustering.tiling.run_tiled_hdbscan", fake)
    stage = cluster.HdbscanTiledStage()
    stage.reporter = _Recorder() if reporter else None
    params = {**cluster.HdbscanTiledStage.template(), "device": device}
    stage.run(_inputs(), params)
    return (stage.reporter.logs if reporter else None), seen


def test_tiled_cuda_warns_once_through_the_reporter(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="karak.stages.cluster"):
        logs, seen = _run(monkeypatch, "cuda")
    assert seen["device"] == "cuda"
    assert logs == [("warning", cluster.TILED_CUDA_WARNING)]
    assert cluster.TILED_CUDA_WARNING not in caplog.text   # not printed twice


def test_tiled_cuda_without_a_reporter_warns_in_the_log(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="karak.stages.cluster"):
        _run(monkeypatch, "cuda", reporter=False)
    assert "2 clusters against 8" in caplog.text and "device=cpu" in caplog.text


def test_tiled_cpu_does_not_warn(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="karak.stages.cluster"):
        logs, _ = _run(monkeypatch, "cpu")
    assert logs == [] and cluster.TILED_CUDA_WARNING not in caplog.text
