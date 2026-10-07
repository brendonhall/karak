"""split_threshold and split_gmm stages: names, history, validation."""

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.clustering.refinement import gmm_split, threshold_split
from karak.stages import get
from karak.stages.base import StageError
from karak.stages.payloads import ElementCube, LabelState, Labels, Space


@pytest.fixture()
def scene():
    """Cleaned labels over the synthetic scene: phase 0 = columns 8..35
    (channel A high), phase 1 = columns 36..63 (channel B high)."""
    cube = make_synthetic_scene()
    H, W, _ = cube.shape
    rows, cols = np.nonzero(cube[:, :, 0] > 0)
    idx = np.stack([rows, cols], 1).astype(np.int32)
    labels = np.where(cols < 36, 0, 1).astype(np.int32)
    payload = Labels(labels=labels, probabilities=None, mineral_indices=idx,
                     image_shape=(H, W), state=LabelState.CLEANED,
                     names={0: "A phase", 1: "B phase"})
    return payload, ElementCube(pixels=cube, element_names=("A", "B", "C"),
                                space=Space.DENOISED)


def test_split_threshold_parity_and_history(scene):
    labels, cube = scene
    expected, new_label, n = threshold_split(
        labels.labels, cube.pixels, labels.mineral_indices, ["A", "B", "C"], 0,
        [("C", ">", 0.5)])
    assert n > 0
    out = get("split_threshold")().run(
        {"labels": labels, "cube": cube},
        {"target_phase": 0, "rule": "C > 0.5", "new_name": "High-C", "note": "test"})
    got = out["labels"]
    np.testing.assert_array_equal(got.labels, expected)
    assert got.names == {0: "A phase", 1: "B phase", new_label: "High-C"}
    rec = got.history[-1]
    assert rec["stage"] == "split_threshold" and rec["parent"] == 0
    assert rec["new_labels"] == [new_label] and rec["n_pixels"] == {new_label: n}
    assert rec["method"] == "C > 0.5" and rec["note"] == "test"
    assert labels.names == {0: "A phase", 1: "B phase"}   # input not mutated


def test_split_threshold_check_params():
    cls = get("split_threshold")
    assert cls.check_params({"target_phase": 0, "rule": "A >", "new_name": "x", "note": ""})
    assert cls.check_params({"target_phase": 0, "rule": "A > 1", "new_name": "", "note": ""})
    assert not cls.check_params({"target_phase": 0, "rule": "A > 1", "new_name": "x", "note": ""})


def test_split_threshold_unknown_channel_is_a_stage_error(scene):
    labels, cube = scene
    with pytest.raises(StageError, match="'Ti'"):
        get("split_threshold")().run(
            {"labels": labels, "cube": cube},
            {"target_phase": 0, "rule": "Ti > 0.5", "new_name": "x", "note": ""})


def test_split_gmm_parity_keep_parent(scene):
    labels, cube = scene
    expected, new, info = gmm_split(
        labels.labels, cube.pixels, None, labels.mineral_indices, ["A", "B", "C"], 1,
        ["A", "B"], n_components=2, bse_weight=1.0, subsample_n=None, random_state=0,
        keep_parent=True, order_by="")
    out = get("split_gmm")().run(
        {"labels": labels, "cube": cube},
        {"target_phase": 1, "features": "A,B", "n_components": 2, "bse_weight": 1.0,
         "subsample_n": 0, "random_state": 0, "keep_parent": True, "order_by": "",
         "new_names": "B minor", "note": "n"})
    got = out["labels"]
    np.testing.assert_array_equal(got.labels, expected)
    assert new == [2] and got.names[2] == "B minor"
    rec = got.history[-1]
    assert rec["stage"] == "split_gmm" and rec["new_labels"] == [2]
    assert rec["n_pixels"] == info["n_pixels"] and rec["method"] == info["method"]


def test_split_gmm_without_parent_empties_it_and_orders(scene):
    labels, cube = scene
    out = get("split_gmm")().run(
        {"labels": labels, "cube": cube},
        {"target_phase": 1, "features": "A,B", "n_components": 2, "bse_weight": 1.0,
         "subsample_n": 0, "random_state": 0, "keep_parent": False, "order_by": "B",
         "new_names": "Low B,High B", "note": ""})
    got = out["labels"]
    assert not (got.labels == 1).any()
    assert got.names[2] == "Low B" and got.names[3] == "High B"
    rec = got.history[-1]
    assert rec["new_labels"] == [2, 3]


def test_split_gmm_too_few_pixels_logs_and_passes_through(scene, caplog):
    labels, cube = scene
    few = labels.labels.copy(); few[few == 1] = 0; few[:5] = 1
    labels = labels.replace(labels=few)
    with caplog.at_level("WARNING"):
        out = get("split_gmm")().run(
            {"labels": labels, "cube": cube},
            {"target_phase": 1, "features": "A", "n_components": 2, "bse_weight": 1.0,
             "subsample_n": 0, "random_state": 0, "keep_parent": False, "order_by": "",
             "new_names": "x,y", "note": ""})
    np.testing.assert_array_equal(out["labels"].labels, few)
    assert out["labels"].history[-1]["new_labels"] == []
    assert "skipped" in caplog.text


def test_split_gmm_check_params():
    cls = get("split_gmm")
    base = {"target_phase": 1, "features": "A,B", "n_components": 2, "bse_weight": 1.0,
            "subsample_n": 0, "random_state": 0, "keep_parent": True, "order_by": "",
            "new_names": "x", "note": ""}
    assert not cls.check_params(base)
    assert cls.check_params({**base, "features": "A/(B+C)"})          # bad ratio
    assert cls.check_params({**base, "order_by": "C"})                 # not a feature
    assert cls.check_params({**base, "new_names": "x,y"})              # one too many
    assert cls.check_params({**base, "keep_parent": False})            # one too few
    assert not cls.check_params({**base, "keep_parent": False, "new_names": "x,y"})


def test_split_gmm_unknown_channel_is_a_stage_error(scene):
    labels, cube = scene
    with pytest.raises(StageError, match="'Ti'"):
        get("split_gmm")().run(
            {"labels": labels, "cube": cube},
            {"target_phase": 1, "features": "Ti", "n_components": 2, "bse_weight": 1.0,
             "subsample_n": 0, "random_state": 0, "keep_parent": True, "order_by": "",
             "new_names": "x", "note": ""})
