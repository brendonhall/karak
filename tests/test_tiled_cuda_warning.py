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


def test_tiled_cuda_warns_once_through_the_cli_log_capture(monkeypatch):
    from karak.cli.logs import capture_logs

    recorder = _Recorder()
    with capture_logs(recorder):
        logs, seen = _run(monkeypatch, "cuda", reporter=False)
    assert seen["device"] == "cuda"
    # compute_tile_grid also logs an info record; only warnings count here
    warnings = [r for r in recorder.logs if r[0] == "warning"]
    assert warnings == [("warning", cluster.TILED_CUDA_WARNING)]


def test_tiled_cuda_warns_under_the_executor_null_reporter(monkeypatch, caplog):
    # review regression: karak bench and programmatic runs attach NullReporter
    from karak.flow.events import NullReporter

    monkeypatch.setattr("karak.clustering.tiling.run_tiled_hdbscan",
                        lambda f, i, s, c, h, t, **kw: (np.zeros(len(f), np.int32), None,
                                                        np.zeros(len(f), np.float32), [], []))
    stage = cluster.HdbscanTiledStage()
    stage.reporter = NullReporter()
    with caplog.at_level("WARNING", logger="karak.stages.cluster"):
        stage.run(_inputs(), {**cluster.HdbscanTiledStage.template(), "device": "cuda"})
    assert caplog.text.count("hdbscan_tiled on cuda") == 1


def test_tiled_cuda_without_a_reporter_warns_in_the_log(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="karak.stages.cluster"):
        _run(monkeypatch, "cuda", reporter=False)
    assert "2 clusters against 8" in caplog.text and "device=cpu" in caplog.text


def test_tiled_cpu_does_not_warn(monkeypatch, caplog):
    with caplog.at_level("WARNING", logger="karak.stages.cluster"):
        _run(monkeypatch, "cpu", reporter=False)
    assert cluster.TILED_CUDA_WARNING not in caplog.text


def test_tiled_run_logs_the_tile_grid_once(caplog):
    # the stage recomputes the grid for the deferred pixels; only the run
    # itself logs it (no monkeypatch: the real tiled run calls the grid)
    stage = cluster.HdbscanTiledStage()
    stage.reporter = None
    with caplog.at_level("INFO", logger="karak.clustering.tiling"):
        stage.run(_inputs(), {**cluster.HdbscanTiledStage.template(), "device": "cpu"})
    assert caplog.text.count("Tile grid") == 1
