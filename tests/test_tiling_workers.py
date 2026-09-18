"""Parity: parallel per-tile HDBSCAN is byte-identical to serial."""

from __future__ import annotations

import numpy as np
import pytest

from karak.config import ClusterConfig, HDBSCANConfig, TiledConfig
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
    config = ClusterConfig(
        strategy="tiled",
        hdbscan=HDBSCANConfig(min_cluster_size=50, random_state=0),
        tiled=TiledConfig(tile_size=32, merge_threshold=0.9,
                          min_tile_pixels=100, min_clusters_per_tile=1),
    )
    return features, mineral_indices, (H, W), cube, config


def _run(tiled_inputs, workers):
    features, indices, shape, cube, config = tiled_inputs
    return run_tiled_hdbscan(
        features, indices, shape, cube, config,
        skip_knn=True, workers=workers,
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
