"""Tests for the Stage contract: coerce -> check -> apply."""

from __future__ import annotations

import pytest

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.payloads import ElementCube, Space

import numpy as np


def _cube(space=Space.RAW):
    pixels = np.zeros((4, 4, 2), dtype=np.float32)
    return ElementCube(pixels=pixels, element_names=("Fe", "Mg"), space=space)


class DoublerStage(Stage):
    id = "doubler"
    label = "Doubler"
    description = "Doubles pixel values."
    INPUTS = [Port("cube", space=Space.RAW)]
    OUTPUTS = [Port("cube", space=Space.RAW)]
    PARAMS = [Param("factor", "float", 2.0, min=0.0, max=10.0)]

    def apply(self, inputs, params):
        cube = inputs["cube"]
        return {"cube": cube.replace(pixels=cube.pixels * params["factor"])}


def test_run_coerces_defaults_and_calls_apply():
    out = DoublerStage().run({"cube": _cube()})
    assert isinstance(out["cube"], ElementCube)


def test_run_applies_given_params():
    cube = _cube().replace(pixels=np.ones((4, 4, 2), dtype=np.float32))
    out = DoublerStage().run({"cube": cube}, {"factor": 3})
    assert out["cube"].pixels[0, 0, 0] == 3.0


def test_missing_required_input_raises():
    with pytest.raises(StageError):
        DoublerStage().run({})


def test_space_mismatch_raises():
    with pytest.raises(StageError):
        DoublerStage().run({"cube": _cube(space=Space.DENOISED)})


def test_unknown_param_raises():
    with pytest.raises(ValueError):
        DoublerStage.coerce_params({"nope": 1})


def test_schema_is_pure_json():
    import json
    schema = DoublerStage.schema()
    assert schema["id"] == "doubler"
    assert schema["params"][0]["name"] == "factor"
    assert schema["inputs"][0]["name"] == "cube"
    json.dumps(schema)  # must not raise
