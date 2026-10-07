"""Stage.check_params feeds karak validate and Stage.run."""

import pytest

from conftest import complete
from karak.errors import StageError
from karak.flow.graph import Graph, Node
from karak.flow.validate import validate
from karak.stages import registry
from karak.stages.base import Param, Stage


class _Picky(Stage):
    id = "temp_picky_stage"
    label = "Picky"
    description = "test stage"
    INPUTS: list = []
    OUTPUTS: list = []
    PARAMS = [Param("word", "str", "ok", "Word")]

    @classmethod
    def check_params(cls, params):
        return [] if params["word"] == "ok" else [f"word: {params['word']!r} is not ok"]

    def apply(self, inputs, params):
        return {}


@pytest.fixture()
def picky():
    registry._REGISTRY["temp_picky_stage"] = _Picky
    try:
        yield _Picky
    finally:
        registry._REGISTRY.pop("temp_picky_stage", None)


def test_validate_reports_check_params_errors(picky):
    graph = complete(Graph(nodes=(Node("p", "temp_picky_stage", {"word": "bad"}),)))
    issues = validate(graph)
    assert [(i.level, i.where) for i in issues] == [("error", "p")]
    assert "word: 'bad' is not ok" in issues[0].message


def test_run_raises_on_check_params_errors(picky):
    with pytest.raises(StageError, match="word: 'bad'"):
        picky().run({}, {"word": "bad"})
    assert picky().run({}, {"word": "ok"}) == {}
