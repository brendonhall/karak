"""Generic threshold and GMM splits in clustering/refinement.py."""

import numpy as np
import pytest

# Task 3.2 adds gmm_split to this import.
from karak.clustering.refinement import _extract_olivine, threshold_split


def _scene():
    """(labels, cube, indices): 20 pixels of phase 0, channels Fe, Ca, Mg."""
    n = 20
    cube = np.zeros((1, n, 3), dtype=np.float32)
    cube[0, :, 0] = np.linspace(0, 1, n)          # Fe rises along the row
    cube[0, :, 1] = 0.05                           # Ca low everywhere
    cube[0, :10, 2] = 0.9                          # Mg high in the first half
    indices = np.stack([np.zeros(n, int), np.arange(n)], 1).astype(np.int32)
    return np.zeros(n, dtype=np.int32), cube, indices


def test_threshold_split_matches_extract_olivine():
    labels, cube, idx = _scene()
    expected, exp_label = _extract_olivine(labels, cube, idx, ["Fe-K", "Ca", "Mg"],
                                           target_phase=0, fe_threshold=0.6, ca_threshold=0.1)
    got, label, n = threshold_split(labels, cube, idx, ["Fe-K", "Ca", "Mg"], 0,
                                    [("Fe-K", ">", 0.6), ("Ca", "<", 0.1)])
    np.testing.assert_array_equal(got, expected)
    assert label == exp_label == 1 and n == (got == 1).sum() > 0


def test_threshold_split_unknown_channel():
    labels, cube, idx = _scene()
    with pytest.raises(ValueError, match="'Ti'"):
        threshold_split(labels, cube, idx, ["Fe-K", "Ca", "Mg"], 0, [("Ti", ">", 0.1)])


def test_threshold_split_nothing_moves():
    labels, cube, idx = _scene()
    got, label, n = threshold_split(labels, cube, idx, ["Fe-K", "Ca", "Mg"], 0,
                                    [("Fe-K", ">", 5.0)])
    np.testing.assert_array_equal(got, labels)
    assert label == -1 and n == 0
