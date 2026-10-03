"""Fingerprint means and standard deviations accumulate in float64 by
default; float32 reproduces the published baseline (numpy's float32
accumulation along axis 0)."""

from __future__ import annotations

import numpy as np
import pytest

from karak.identification.fingerprint import compute_fingerprints
from karak.stages.fingerprints import FingerprintsStage


def _large_cluster():
    """One 4 M-pixel cluster and one small one: enough rows in the large
    cluster for float32 row-by-row sums to drift."""
    rng = np.random.default_rng(0)
    cube = (0.3 + 0.05 * rng.standard_normal((2000, 2000, 2))).astype(np.float32)
    rows, cols = np.nonzero(np.ones((2000, 2000), bool))
    indices = np.stack([rows, cols], axis=1).astype(np.int32)
    labels = np.zeros(rows.size, np.int32)
    labels[:1000] = 1
    return cube, labels, indices


def _fingerprints(accumulate):
    cube, labels, indices = _large_cluster()
    data = compute_fingerprints(cube, labels, indices, ["Al", "Si"],
                                accumulate=accumulate)
    pixels = cube[indices[:, 0], indices[:, 1]]
    return data["fingerprints"], pixels[labels == 0]


def test_float64_accumulation_matches_a_float64_reference():
    fp, pixels = _fingerprints("float64")
    ref = pixels.astype(np.float64)
    assert fp[0]["mean"].dtype == np.float32 and fp[0]["std"].dtype == np.float32
    np.testing.assert_allclose(fp[0]["mean"], ref.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(fp[0]["std"], ref.std(axis=0), rtol=1e-5)


def test_float32_accumulation_reproduces_numpy_float32_and_drifts():
    fp, pixels = _fingerprints("float32")
    np.testing.assert_array_equal(fp[0]["mean"], pixels.mean(axis=0))
    np.testing.assert_array_equal(fp[0]["std"], pixels.std(axis=0))
    ref = pixels.astype(np.float64)
    # the baseline's error is real at this size, which is why float64 is the default
    assert np.abs(fp[0]["std"] - ref.std(axis=0)).max() / ref.std(axis=0).min() > 1e-4


def test_unknown_accumulate_is_an_error():
    cube, labels, indices = _large_cluster()
    with pytest.raises(ValueError, match="accumulate"):
        compute_fingerprints(cube, labels[:4], indices[:4], ["Al", "Si"],
                             accumulate="float16")


def test_stage_declares_accumulate_with_float64_template():
    params = {p.name: p for p in FingerprintsStage.PARAMS}
    assert [p.name for p in FingerprintsStage.PARAMS] == [
        "similarity_threshold", "accumulate"]
    assert params["accumulate"].default == "float64"
    assert params["accumulate"].choices == ("float64", "float32")
