"""Parity: parallel per-tile HDBSCAN is byte-identical to serial."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import hdbscan_cfg, tiled_cfg
from karak.clustering.tiling import run_tiled_hdbscan


@pytest.fixture(scope="module")
def tiled_inputs():
    """A 64x64 scene with two blobs, split into 4 tiles of 32px."""
    rng = np.random.default_rng(0)
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    mineral = np.ones((H, W), dtype=bool)
    mineral_indices = np.column_stack(np.nonzero(mineral)).astype(np.int32)

    blob = (xx >= 32).astype(np.float32)
    features = np.column_stack([
        blob.ravel() + rng.normal(0, 0.05, H * W),
        (1 - blob).ravel() + rng.normal(0, 0.05, H * W),
    ]).astype(np.float32)

    cube = np.stack([blob, 1 - blob, np.ones((H, W), np.float32)], axis=-1)
    config = (
        hdbscan_cfg(min_cluster_size=50, random_state=0),
        tiled_cfg(tile_size=32, merge_threshold=0.9,
                  min_tile_pixels=100, min_clusters_per_tile=1),
    )
    return features, mineral_indices, (H, W), cube, config


def _run(tiled_inputs, workers):
    features, indices, shape, cube, config = tiled_inputs
    return run_tiled_hdbscan(
        features, indices, shape, cube, *config,
        noise_reassign_k=None, skip_knn=True, workers=workers,
    )


def test_parallel_tiles_byte_identical(tiled_inputs):
    raw1, clean1, prob1, tiles1, reg1 = _run(tiled_inputs, workers=1)
    raw2, clean2, prob2, tiles2, reg2 = _run(tiled_inputs, workers=3)
    assert np.array_equal(raw1, raw2)
    assert np.array_equal(prob1, prob2)
    assert len(reg1) == len(reg2)
    for e1, e2 in zip(reg1, reg2):
        assert e1.global_id == e2.global_id
        assert e1.n_pixels == e2.n_pixels
        assert np.allclose(e1.mean_fingerprint, e2.mean_fingerprint)


def _meminfo(tmp_path, monkeypatch, kilobytes):
    import karak.memory as memory

    path = tmp_path / "meminfo"
    path.write_text(f"MemAvailable:   {kilobytes} kB\n")
    monkeypatch.setattr(memory, "MEMINFO", str(path))


def test_parallel_fit_workers_counts_the_largest_fits():
    from karak.clustering.hdbscan_cluster import parallel_fit_workers

    gb = 10**9
    # a full 262 k-pixel tile at min_samples 1000 needs about 8.4 GB
    sizes = [262_144] * 68
    assert parallel_fit_workers(sizes, 1000, 16, available=25 * gb) == 2
    assert parallel_fit_workers(sizes, 1000, 16, available=200 * gb) == 16
    assert parallel_fit_workers(sizes, 1000, 16, available=1 * gb) == 1
    assert parallel_fit_workers(sizes, 1000, 16, available=None) == 16
    # small tiles next to one large one: the large one counts first
    assert parallel_fit_workers([262_144] + [1000] * 20, 1000, 4,
                                available=12 * gb) == 4
    assert parallel_fit_workers(sizes, 1000, 1, available=25 * gb) == 1


def test_pool_shrinks_to_what_fits_and_keeps_the_result(
        tiled_inputs, tmp_path, monkeypatch, caplog):
    import concurrent.futures

    serial = _run(tiled_inputs, workers=1)
    # 1,024-pixel tiles at min_samples 50 need about 1.6 MB each; 3 MB
    # available (80 % = 2.5 MB) fits one at a time, so no pool starts
    _meminfo(tmp_path, monkeypatch, 3000)

    def no_pool(*args, **kwargs):
        raise AssertionError("a pool was started although one fit fits at a time")

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", no_pool)
    with caplog.at_level("WARNING", logger="karak.clustering.tiling"):
        planned = _run(tiled_inputs, workers=4)
    assert "using 1 workers" in caplog.text
    np.testing.assert_array_equal(planned[0], serial[0])
    np.testing.assert_array_equal(planned[2], serial[2])


def test_pool_workers_skip_their_own_memory_check(tmp_path, monkeypatch):
    from karak.clustering.hdbscan_cluster import run_hdbscan
    from karak.errors import StageError

    _meminfo(tmp_path, monkeypatch, 1)    # 1 kB: any fit is refused
    features = np.random.default_rng(1).normal(size=(500, 2)).astype(np.float32)
    cfg = hdbscan_cfg(min_cluster_size=20, min_samples=5)
    with pytest.raises(StageError):
        run_hdbscan(features, cfg, device="cpu")
    labels, _, _ = run_hdbscan(features, cfg, device="cpu", check_memory=False)
    assert labels.shape == (500,)
