"""hires_split: one GMM at full resolution inside a set of phases."""

import numpy as np
import pytest

from karak.clustering.hires import hires_split, resolution_ratio


def test_resolution_ratio():
    assert resolution_ratio((4, 4), (8, 8)) == 2
    assert resolution_ratio((4, 4), (9, 8)) == 2        # one extra hires row is clipped
    assert resolution_ratio((4, 4), (4, 4)) == 1
    with pytest.raises(ValueError, match="integer"):
        resolution_ratio((4, 4), (10, 8))                # 2.5
    with pytest.raises(ValueError, match="integer"):
        resolution_ratio((4, 4), (8, 11))


def _case():
    """Working 4x4: phase 2 in the left two columns, phase 3 right. Hires
    8x8 with Ca and Mg: inside phase 2, Ca/(Ca+Mg) is low in the top half,
    high in the bottom half, except one working pixel whose children tie
    2:2 and one whose children are all undefined (Ca = Mg = 0)."""
    H = W = 4
    labels_img = np.broadcast_to(
        np.where(np.arange(W) < 2, 2, 3).astype(np.int32), (H, W))
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    idx = np.stack([rows.ravel(), cols.ravel()], 1).astype(np.int32)
    labels = labels_img.ravel()
    ca = np.zeros((8, 8), np.float32); mg = np.zeros((8, 8), np.float32)
    ca[:4, :4], mg[:4, :4] = 0.1, 0.9                 # low ratio, top-left block
    ca[4:, :4], mg[4:, :4] = 0.9, 0.1                 # high ratio, bottom-left block
    # working pixel (0, 1) -> hires rows 0..1, cols 2..3: tie, top-left child low
    ca[0, 2:4], mg[0, 2:4] = 0.1, 0.9
    ca[1, 2:4], mg[1, 2:4] = 0.9, 0.1
    # working pixel (3, 1) -> hires rows 6..7, cols 2..3: all undefined
    ca[6:8, 2:4] = 0; mg[6:8, 2:4] = 0
    cube = np.stack([ca, mg], -1)
    return labels, idx, (H, W), cube


def test_hires_split_outputs():
    labels, idx, shape, cube = _case()
    out_labels, hires, new, info = hires_split(
        labels, idx, shape, cube, ["Ca", "Mg"], [2], "Ca/(Ca+Mg)",
        n_components=2, subsample_n=None, random_state=0)
    assert new == [4, 5]                                   # low ratio -> 4, high -> 5
    assert hires.dtype == np.int16 and hires.shape == (8, 8)
    assert (hires[:, 4:] == -1).all()                      # outside the region
    assert (hires[:4, :2] == 4).all() and (hires[4:, :2] == 5).all()
    assert (hires[6:8, 2:4] == -1).all()                   # undefined children
    img = out_labels.reshape(shape)
    assert (img[:, 2:] == 3).all()                         # other phase untouched
    assert img[0, 0] == 4 and img[3, 0] == 5                # majority
    assert img[0, 1] == 4                                  # tie -> child at (0, 2)
    assert img[3, 1] == 2                                  # no defined child -> parent
    assert info["n_undefined"] == 4
    assert info["component_means"][4]["Ca/(Ca+Mg)"] < info["component_means"][5]["Ca/(Ca+Mg)"]


def test_hires_split_clips_an_odd_hires_image():
    labels, idx, shape, cube = _case()
    taller = np.concatenate([cube, cube[-1:]], axis=0)     # 9 x 8
    out_labels, hires, new, _ = hires_split(
        labels, idx, shape, taller, ["Ca", "Mg"], [2], "Ca/(Ca+Mg)",
        n_components=2, subsample_n=None, random_state=0)
    assert hires.shape == (9, 8) and (hires[8] == -1).all()
    assert new == [4, 5]
