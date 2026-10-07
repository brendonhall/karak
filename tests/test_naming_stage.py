"""name_phases attaches researcher names to the base phases."""

import numpy as np
import pytest

from karak.stages import get
from karak.stages.base import StageError
from karak.stages.payloads import LabelState, Labels


def _labels(values):
    arr = np.asarray(values, dtype=np.int32)
    return Labels(labels=arr, probabilities=None,
                  mineral_indices=np.zeros((arr.size, 2), dtype=np.int32),
                  image_shape=(1, arr.size), state=LabelState.CLEANED)


def test_name_phases_sets_names_and_keeps_labels():
    out = get("name_phases")().run(
        {"labels": _labels([0, 1, 1, 2])},
        {"names": "0: Ilmenite; 1: Pyroxene; 2: Plagioclase", "note": "from fingerprints"})
    assert out["labels"].names == {0: "Ilmenite", 1: "Pyroxene", 2: "Plagioclase"}
    np.testing.assert_array_equal(out["labels"].labels, [0, 1, 1, 2])
    assert out["labels"].state is LabelState.CLEANED
    assert out["labels"].history == ()


def test_name_phases_merges_with_existing_names():
    base = _labels([0, 1]).replace(names={0: "Old", 5: "Kept"})
    out = get("name_phases")().run({"labels": base}, {"names": "0: New", "note": ""})
    assert out["labels"].names == {0: "New", 5: "Kept"}


def test_name_phases_rejects_a_label_the_data_does_not_have():
    with pytest.raises(StageError, match="label 7"):
        get("name_phases")().run({"labels": _labels([0, 1])},
                                 {"names": "0: A; 7: B", "note": ""})


def test_name_phases_check_params_reports_bad_text():
    errors = get("name_phases").check_params({"names": "0 A", "note": ""})
    assert errors and errors[0].startswith("names")
