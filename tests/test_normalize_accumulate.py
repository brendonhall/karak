"""z-score statistics accumulate in float64 by default; float32 reproduces
the published baseline (numpy's float32 accumulation along axis 0)."""

from __future__ import annotations

import numpy as np
import pytest

from karak.preprocessing.compositional import zscore_normalize
from karak.stages.normalize import NormalizeStage


def _large_cube():
    """4 M mineral pixels: enough rows for float32 row-by-row sums to drift."""
    rng = np.random.default_rng(0)
    cube = (0.3 + 0.05 * rng.standard_normal((2000, 2000, 2))).astype(np.float32)
    mask = np.ones((2000, 2000), bool)
    return cube, mask


def test_float64_accumulation_matches_a_float64_reference():
    cube, mask = _large_cube()
    _, means, stds = zscore_normalize(cube, mask, accumulate="float64")
    ref = cube[mask].astype(np.float64)
    assert means.dtype == np.float32 and stds.dtype == np.float32
    np.testing.assert_allclose(means, ref.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(stds, ref.std(axis=0), rtol=1e-5)


def test_float32_accumulation_reproduces_numpy_float32_and_drifts():
    cube, mask = _large_cube()
    _, means, stds = zscore_normalize(cube, mask, accumulate="float32")
    data = cube[mask]
    np.testing.assert_array_equal(means, data.mean(axis=0))
    np.testing.assert_array_equal(stds, data.std(axis=0))
    ref = data.astype(np.float64)
    # the baseline's error is real at this size, which is why float64 is the default
    assert np.abs(stds - ref.std(axis=0)).max() / ref.std(axis=0).min() > 1e-4


def test_float32_normalized_cube_is_identical_to_the_old_code():
    cube, mask = _large_cube()
    normalized, _, _ = zscore_normalize(cube, mask, accumulate="float32")
    data = cube[mask]
    expected = np.zeros_like(cube)
    expected[mask] = ((data - data.mean(axis=0)) / data.std(axis=0)).astype(np.float32)
    np.testing.assert_array_equal(normalized, expected)


def test_unknown_accumulate_is_an_error():
    cube, mask = _large_cube()
    with pytest.raises(ValueError, match="accumulate"):
        zscore_normalize(cube[:4, :4], mask[:4, :4], accumulate="float16")


def test_stage_declares_accumulate_with_float64_template():
    params = {p.name: p for p in NormalizeStage.PARAMS}
    assert [p.name for p in NormalizeStage.PARAMS] == ["method", "accumulate", "device"]
    assert params["accumulate"].default == "float64"
    assert params["accumulate"].choices == ("float64", "float32")
