"""split_threshold and split_gmm stages: names, history, validation."""

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.clustering.refinement import threshold_split
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
