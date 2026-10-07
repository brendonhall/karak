"""Generic threshold and GMM splits in clustering/refinement.py."""

import numpy as np
import pytest

from karak.clustering.refinement import _extract_olivine, gmm_split, threshold_split


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


def _two_blobs():
    """40 pixels of phase 0: Cl low in the first 30, high in the last 10."""
    rng = np.random.default_rng(0)
    n = 40
    cube = np.zeros((1, n, 2), dtype=np.float32)       # channels Cl, Na
    cube[0, :30, 0] = rng.normal(0.05, 0.005, 30)
    cube[0, 30:, 0] = rng.normal(0.17, 0.005, 10)
    cube[0, :, 1] = rng.normal(0.3, 0.01, n)
    indices = np.stack([np.zeros(n, int), np.arange(n)], 1).astype(np.int32)
    return np.zeros(n, dtype=np.int32), cube, indices


def test_gmm_split_keep_parent_gives_smaller_component_the_new_label():
    labels, cube, idx = _two_blobs()
    got, new, info = gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["Cl", "Na"],
                               n_components=2, bse_weight=1.0, subsample_n=None,
                               random_state=0, keep_parent=True, order_by="")
    assert new == [1]
    assert (got[:30] == 0).all() and (got[30:] == 1).all()
    assert info["n_pixels"] == {1: 10}


def test_gmm_split_without_parent_orders_by_feature_mean():
    labels, cube, idx = _two_blobs()
    got, new, info = gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["Cl", "Na"],
                               n_components=2, bse_weight=1.0, subsample_n=None,
                               random_state=0, keep_parent=False, order_by="Cl")
    assert new == [1, 2]                      # low Cl -> 1, high Cl -> 2
    assert (got[:30] == 1).all() and (got[30:] == 2).all()
    assert not (got == 0).any()
    assert info["component_means"][1]["Cl"] < info["component_means"][2]["Cl"]


def test_gmm_split_ratio_and_bse_features():
    labels, cube, idx = _two_blobs()
    bse = np.full((1, 40), 0.5, dtype=np.float32)
    got, new, info = gmm_split(labels, cube, bse, idx, ["Cl", "Na"], 0,
                               ["Cl/(Cl+Na)", "BSE"], n_components=2, bse_weight=2.0,
                               subsample_n=None, random_state=0, keep_parent=True,
                               order_by="")
    assert new == [1] and info["method"].startswith("GMM")


def test_gmm_split_too_few_pixels_is_a_no_op():
    labels, cube, idx = _two_blobs()
    labels = labels.copy(); labels[5:] = 9             # phase 0 has 5 pixels
    got, new, info = gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["Cl"],
                               n_components=2, bse_weight=1.0, subsample_n=None,
                               random_state=0, keep_parent=False, order_by="")
    np.testing.assert_array_equal(got, labels)
    assert new == [] and "skipped" in info


def test_gmm_split_bse_token_without_bse_raises():
    labels, cube, idx = _two_blobs()
    with pytest.raises(ValueError, match="BSE"):
        gmm_split(labels, cube, None, idx, ["Cl", "Na"], 0, ["BSE"],
                  n_components=2, bse_weight=1.0, subsample_n=None,
                  random_state=0, keep_parent=True, order_by="")
