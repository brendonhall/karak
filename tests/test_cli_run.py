"""`karak run`: reporter choice, log capture, exit codes, real stepwise run."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.cli.dashboard import DashboardReporter
from karak.cli.main import _run_reporter, main
from karak.cli.reporter import RichReporter
from karak.stages import registry
from karak.stages.base import Port, Stage


def test_reporter_is_plain_when_stdout_is_not_a_terminal():
    assert isinstance(_run_reporter(plain=False, isatty=False), RichReporter)


def test_reporter_is_plain_with_flag():
    assert isinstance(_run_reporter(plain=True, isatty=True), RichReporter)


def test_reporter_is_dashboard_on_a_terminal():
    assert isinstance(_run_reporter(plain=False, isatty=True), DashboardReporter)


class FailingStage(Stage):
    id = "cli_test_fail"
    label = "Always fails"
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        raise ValueError("boom from stage")


@pytest.fixture
def failing_stage():
    registry.register(FailingStage)
    yield
    registry._REGISTRY.pop("cli_test_fail", None)


def _karak_logger_state():
    logger = logging.getLogger("karak")
    return list(logger.handlers), logger.level, logger.propagate


def test_run_failure_exits_1_and_restores_logging(tmp_path, capsys, failing_stage):
    flow = tmp_path / "flow.json"
    flow.write_text(json.dumps({
        "version": 1, "name": "fail",
        "nodes": [{"id": "bad", "type": "cli_test_fail", "params": {}}],
        "edges": [],
    }))
    before = _karak_logger_state()
    rc = main(["run", str(flow), "--out", str(tmp_path / "o"), "--plain"])
    assert rc == 1
    assert "boom from stage" in capsys.readouterr().err
    assert _karak_logger_state() == before


def test_run_validation_failure_exits_1(tmp_path, capsys):
    flow = tmp_path / "flow.json"
    flow.write_text(json.dumps({
        "version": 1, "name": "broken",
        "nodes": [{"id": "x", "type": "no_such_stage", "params": {}}],
        "edges": [],
    }))
    before = _karak_logger_state()
    rc = main(["run", str(flow), "--out", str(tmp_path / "o"), "--plain"])
    assert rc == 1
    assert "failed validation" in capsys.readouterr().err
    assert _karak_logger_state() == before


def test_keyboard_interrupt_exits_130(tmp_path, monkeypatch, capsys):
    import karak.flow.executor as executor

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(executor, "run", interrupted)
    rc = main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"),
               "--plain"])
    assert rc == 130
    assert "interrupted" in capsys.readouterr().out


@pytest.fixture
def scene(tmp_path):
    import imageio.v2 as imageio
    import matplotlib

    import karak.io.loaders as loaders

    orig_dir, orig_mem = loaders._CACHE_DIR, loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}
    palette = (
        np.array(
            [matplotlib.colormaps["jet"](s)[:3] for s in np.linspace(0, 1, 64)]
        ) * 255
    ).astype(np.uint8)
    lut_path = tmp_path / "palette.npy"
    np.save(lut_path, palette)
    cube = make_synthetic_scene()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, element in enumerate(("A", "B", "C")):
        indices = np.round(cube[:, :, i] * 63).astype(np.uint8)
        imageio.imwrite(data_dir / f"s-01-{element}.png", palette[indices])
    imageio.imwrite(data_dir / "s-01-SEM.png", np.zeros((64, 64), dtype=np.uint8))
    yield data_dir, f"lut:{lut_path}"
    loaders._CACHE_DIR, loaders._FULL_LUT_CACHE = orig_dir, orig_mem


def test_stepwise_plain_run_end_to_end(tmp_path, capsys, scene):
    data_dir, colormap = scene
    argv = ["run", "--builtin", "stepwise", "--input", str(data_dir),
            "--out", str(tmp_path / "out" / "run"), "--plain",
            "--set", f"src.colormap={colormap}"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    for expected in [
        "flow=stepwise", "src.colormap = lut:", "src: 4/4",
        "src.cube -> ElementCube 32×32×3 float32", "src.bse -> BseImage",
        "run finished in", "(1 ran, 0 cached)", "Found 4 matching files",
    ]:
        assert expected in out, expected
    assert "nodes:" not in out

    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "src: cached (" in out
    assert "src.cube -> ElementCube 32×32×3" in out
    assert "(0 ran, 1 cached)" in out
