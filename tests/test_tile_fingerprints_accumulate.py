"""Tile and rare-cluster fingerprints accumulate in float64 by default;
float32 reproduces the published baseline (numpy's float32 axis-0 sums)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import rare_cfg
from karak.clustering.tiling import compute_tile_fingerprints, recluster_unassigned


def _tile(n_side=2000):
    """A 4 M-pixel 'tile' with one cluster: enough rows for float32 sums to drift."""
    rng = np.random.default_rng(0)
    cube = (0.3 + 0.05 * rng.standard_normal((n_side, n_side, 2))).astype(np.float32)
    rows, cols = np.nonzero(np.ones((n_side, n_side), bool))
    indices = np.stack([rows, cols], axis=1).astype(np.int32)
    return cube, indices, np.arange(rows.size), np.zeros(rows.size, np.int32)


def test_float64_tile_fingerprints_match_a_float64_reference():
    cube, indices, tile_idx, labels = _tile()
    fp = compute_tile_fingerprints(cube, indices, tile_idx, labels, accumulate="float64")
    ref = cube.reshape(-1, 2).astype(np.float64).mean(axis=0)
    assert fp[0].dtype == np.float32
    np.testing.assert_allclose(fp[0], ref, rtol=1e-6)


def test_float32_tile_fingerprints_reproduce_the_old_code_and_drift():
    cube, indices, tile_idx, labels = _tile()
    fp = compute_tile_fingerprints(cube, indices, tile_idx, labels, accumulate="float32")
    spectra = cube[indices[:, 0], indices[:, 1], :]
    np.testing.assert_array_equal(fp[0], np.mean(spectra, axis=0))
    ref = spectra.astype(np.float64).mean(axis=0)
    assert np.abs(fp[0] - ref).max() / ref.min() > 1e-5


def test_unknown_accumulate_is_an_error():
    cube, indices, tile_idx, labels = _tile(4)
    with pytest.raises(ValueError, match="accumulate"):
        compute_tile_fingerprints(cube, indices, tile_idx, labels, accumulate="float16")
    with pytest.raises(ValueError, match="accumulate"):
        recluster_unassigned(
            np.zeros((4, 2), np.float32), np.full(4, -1, np.int32),
            cube, indices[:4], [], replace(rare_cfg(), accumulate="float16"),
            random_state=0)


def test_stages_declare_accumulate_with_a_float64_template():
    from karak.stages.cluster import HdbscanGlobalStage, HdbscanTiledStage
    from karak.stages.rare_phase import RarePhaseStage

    for cls in (HdbscanTiledStage, RarePhaseStage):
        param = {p.name: p for p in cls.PARAMS}["accumulate"]
        assert param.default == "float64" and param.choices == ("float64", "float32")
    assert "accumulate" not in {p.name for p in HdbscanGlobalStage.PARAMS}
