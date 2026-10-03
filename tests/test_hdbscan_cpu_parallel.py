"""hdbscan_global on cpu with --workers: parallel core distances and a
process pool for approximate_predict, with identical labels."""

from __future__ import annotations

import numpy as np

from conftest import hdbscan_cfg
import karak.clustering.hdbscan_cluster as hc
from karak.clustering.hdbscan_cluster import approximate_predict_parallel, run_hdbscan


def _blobs(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 4)) * 6
    return (centers[rng.integers(0, 3, n)]
            + rng.normal(size=(n, 4))).astype(np.float32)


def test_parallel_predict_equals_the_serial_call(monkeypatch):
    import hdbscan

    monkeypatch.setattr(hc, "_MIN_ROWS_PER_PREDICT_WORKER", 500)
    features = _blobs()
    model = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20,
                            prediction_data=True).fit(features[:1500])
    serial = hdbscan.approximate_predict(model, features)
    parallel = approximate_predict_parallel(model, features, 4)
    np.testing.assert_array_equal(parallel[0], serial[0])
    np.testing.assert_array_equal(parallel[1], serial[1])


def test_chunked_serial_predict_equals_one_call(monkeypatch):
    import hdbscan

    features = _blobs()
    model = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20,
                            prediction_data=True).fit(features[:1500])
    monkeypatch.setattr(hc, "predict_chunk_rows", lambda ms, w: 700)   # 9 chunks
    serial = hdbscan.approximate_predict(model, features)
    chunked = approximate_predict_parallel(model, features, 1)
    np.testing.assert_array_equal(chunked[0], serial[0])
    np.testing.assert_array_equal(chunked[1], serial[1])


def test_chunk_rows_share_the_budget_between_workers():
    rows_1 = hc.predict_chunk_rows(1000, 1)
    assert rows_1 == (2 << 30) // 32_000
    assert hc.predict_chunk_rows(1000, 16) == rows_1 // 16
    assert hc.predict_chunk_rows(10**7, 16) == 1000       # never below the floor


def test_small_inputs_predict_serially(monkeypatch):
    import hdbscan

    def no_pool(*args, **kwargs):
        raise AssertionError("a pool was started for a small input")

    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", no_pool)
    features = _blobs(1200)
    model = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20,
                            prediction_data=True).fit(features[:600])
    labels, _ = approximate_predict_parallel(model, features, 8)
    assert labels.shape == (1200,)


def test_subsample_run_gives_the_same_labels_for_any_worker_count(monkeypatch):
    monkeypatch.setattr(hc, "_MIN_ROWS_PER_PREDICT_WORKER", 500)
    features = _blobs()
    cfg = hdbscan_cfg(min_cluster_size=100, min_samples=20, subsample_n=1500)
    one = run_hdbscan(features, cfg, device="cpu")
    four = run_hdbscan(features, cfg, device="cpu", core_dist_n_jobs=4,
                       predict_workers=4)
    np.testing.assert_array_equal(one[0], four[0])
    np.testing.assert_array_equal(one[1], four[1])


def test_full_fit_gives_the_same_labels_for_any_job_count():
    features = _blobs(3000)
    cfg = hdbscan_cfg(min_cluster_size=100, min_samples=20)
    one = run_hdbscan(features, cfg, device="cpu", core_dist_n_jobs=1)
    many = run_hdbscan(features, cfg, device="cpu", core_dist_n_jobs=8)
    np.testing.assert_array_equal(one[0], many[0])
    np.testing.assert_array_equal(one[1], many[1])


def test_global_stage_passes_the_workers(monkeypatch):
    import os

    import karak.stages.cluster as cluster
    from karak.stages.payloads import PCAFeatures

    seen = []

    def fake(features, config, *, device, core_dist_n_jobs, predict_workers):
        seen.append((core_dist_n_jobs, predict_workers))
        n = features.shape[0]
        return np.zeros(n, np.int32), np.zeros(n, np.float32), None

    monkeypatch.setattr(cluster, "run_hdbscan", fake)
    features = _blobs(10)
    payload = PCAFeatures(features, np.zeros((10, 2), np.int32), (1, 10),
                          np.ones(4), 4)
    for workers in (None, 1, 3, 0):
        stage = cluster.HdbscanGlobalStage()
        stage.workers = workers
        stage.run({"features": payload})
    cores = os.cpu_count() or 1
    assert seen == [(None, 1), (None, 1), (3, 3), (cores, cores)]
