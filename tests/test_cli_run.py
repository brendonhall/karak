"""`karak run`: reporter choice, log capture, exit codes, real stepwise run."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
from pathlib import Path

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
        "version": 2, "name": "fail",
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
        "msk.masks -> MaskSet mineral",
        "dn.cube -> ElementCube 32×32×3 float32", "space=denoised",
        "nrm.cube -> ElementCube 32×32×3 float32", "space=normalized",
        "run finished in", "(4 ran, 0 cached)", "Found 4 matching files",
    ]:
        assert expected in out, expected
    assert "nodes:" not in out

    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "src: cached (" in out
    assert "src.cube -> ElementCube 32×32×3" in out
    assert "msk: cached (" in out
    assert "dn: cached (" in out
    assert "nrm: cached (" in out
    assert "(0 ran, 4 cached)" in out


def test_run_links_cached_outputs_to_their_upstream_recipes(tmp_path, scene):
    import json

    from karak.flow.cache import load_upstream

    data_dir, colormap = scene
    out = tmp_path / "out" / "run"
    assert main(["run", "--builtin", "stepwise", "--input", str(data_dir),
                 "--out", str(out), "--plain",
                 "--set", f"src.colormap={colormap}"]) == 0
    nodes = json.loads((out / "runs" / "latest" / "run.json").read_text())["nodes"]
    src_cube = nodes["src"]["outputs"]["cube"]["file"]
    masks = nodes["msk"]["outputs"]["masks"]["file"]
    src_recipe = Path(src_cube).name.split("__")[0]
    assert load_upstream(Path(masks)) == {"cube": src_recipe}
    assert load_upstream(Path(src_cube)) == {}


def test_unexpected_error_closes_reporter_as_failed(tmp_path, monkeypatch):
    import karak.cli.main as cli_main
    import karak.flow.executor as executor

    closed = []

    class Recorder(RichReporter):
        def close(self, status=None):
            closed.append(status)

    def disk_full(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(executor, "run", disk_full)
    monkeypatch.setattr(cli_main, "_run_reporter", lambda plain, isatty: Recorder())
    with pytest.raises(OSError):
        main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"), "--plain"])
    assert closed[0] == "failed"


def test_cache_compression_flag_reaches_the_executor_and_the_record(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update(kwargs)
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(executor, "run", fake_run)
    out = tmp_path / "o"
    assert main(["run", "--builtin", "stepwise", "--out", str(out), "--plain",
                 "--cache-compression", "none"]) == 0
    assert seen["cache_compression"] == "none"
    data = json.loads((out / "runs" / "latest" / "run.json").read_text())
    assert data["settings"]["cache_compression"] == "none"


def test_cache_compression_defaults_to_lzf(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update(kwargs)
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(executor, "run", fake_run)
    assert main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"), "--plain"]) == 0
    assert seen["cache_compression"] == "lzf"


def test_ram_budget_flag_is_gigabytes(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update(kwargs)
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(executor, "run", fake_run)
    out = tmp_path / "o"
    assert main(["run", "--builtin", "stepwise", "--out", str(out), "--plain",
                 "--ram-budget", "1.5"]) == 0
    assert seen["ram_budget"] == int(1.5 * 2**30)
    data = json.loads((out / "runs" / "latest" / "run.json").read_text())
    assert data["settings"]["ram_budget_gb"] == 1.5


class _CliSource(Stage):
    id = "cli_test_source"
    label = "Source"
    OUTPUTS = [Port("num")]

    def apply(self, inputs, params):
        from karak.stages.payloads import ClusterStats

        return {"num": ClusterStats(stats={"value": 1})}


def test_cache_writer_failure_exits_1_with_an_error_line(tmp_path, monkeypatch, capsys):
    import karak.flow.cache as cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    registry.register(_CliSource)
    try:
        flow = tmp_path / "flow.json"
        flow.write_text(json.dumps({
            "version": 2, "name": "writer",
            "nodes": [{"id": "src", "type": "cli_test_source", "params": {}}],
            "edges": [],
        }))
        rc = main(["run", str(flow), "--out", str(tmp_path / "o"), "--plain"])
    finally:
        registry._REGISTRY.pop(_CliSource.id, None)
    assert rc == 1
    err = capsys.readouterr().err
    assert "error: cache writer: disk full" in err
    assert "Traceback" not in err


@pytest.mark.parametrize("value", ["0", "-1"])
def test_ram_budget_must_be_positive(tmp_path, capsys, value):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"),
              "--plain", "--ram-budget", value])
    assert exc.value.code == 2
    assert "--ram-budget" in capsys.readouterr().err


def test_gpu_budget_flag_is_gigabytes(tmp_path, monkeypatch):
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update(kwargs)
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(executor, "run", fake_run)
    out = tmp_path / "o"
    assert main(["run", "--builtin", "stepwise", "--out", str(out), "--plain",
                 "--gpu-budget", "2"]) == 0
    assert seen["gpu_budget"] == 2 * 2**30
    data = json.loads((out / "runs" / "latest" / "run.json").read_text())
    assert data["settings"]["gpu_budget_gb"] == 2.0


def test_gpu_budget_must_be_positive(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"),
              "--plain", "--gpu-budget", "0"])
    assert exc.value.code == 2
    assert "--gpu-budget" in capsys.readouterr().err


@pytest.mark.skipif(not __import__("karak.accel", fromlist=["cuda_available"]).cuda_available(),
                    reason="no CUDA")
def test_stepwise_on_cuda_keeps_the_cube_on_the_device(tmp_path, scene):
    data_dir, colormap = scene
    out = tmp_path / "out" / "run"
    assert main(["run", "--builtin", "stepwise", "--input", str(data_dir),
                 "--out", str(out), "--plain", "--device", "cuda",
                 "--set", f"src.colormap={colormap}"]) == 0
    nodes = json.loads((out / "runs" / "latest" / "run.json").read_text())["nodes"]
    assert nodes["src"]["outputs"]["cube"]["device"] == "cpu"
    assert nodes["dn"]["outputs"]["cube"]["device"] == "cuda"
    assert nodes["nrm"]["outputs"]["cube"]["device"] == "cuda"


def test_explicit_set_device_overrides_the_device_flag(tmp_path, monkeypatch):
    import karak.accel as accel
    import karak.flow.executor as executor

    seen = {}

    def fake_run(graph, **kwargs):
        seen.update({n.id: n.params.get("device") for n in graph.nodes})
        kwargs["record"].start()
        kwargs["record"].finish("ok")
        return {}

    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(executor, "run", fake_run)
    assert main(["run", "--builtin", "global", "--out", str(tmp_path / "o"),
                 "--plain", "--device", "cuda", "--set", "hdb.device=cpu"]) == 0
    assert seen["hdb"] == "cpu"
    assert seen["dn"] == "cuda"
    assert seen["nrm"] == "cuda" and seen["pca"] == "cuda"


@pytest.mark.skipif(not __import__("karak.accel", fromlist=["cuda_available"]).cuda_available(),
                    reason="no CUDA")
def test_tiled_on_cuda_records_each_output_placement(tmp_path, scene):
    data_dir, colormap = scene
    out = tmp_path / "out" / "run"
    assert main(["run", "--builtin", "tiled", "--input", str(data_dir),
                 "--out", str(out), "--plain", "--no-qc", "--device", "cuda",
                 "--set", f"src.colormap={colormap}",
                 "--set", "hdb.min_cluster_size=50",
                 "--set", "hdb.min_samples=10",
                 "--set", "hdb.tile_size=32"]) == 0
    nodes = json.loads((out / "runs" / "latest" / "run.json").read_text())["nodes"]
    for node in ("dn", "nrm", "pca"):
        assert all(o["device"] == "cuda" for o in nodes[node]["outputs"].values()), node
    for node in ("knn", "stats", "fp"):
        assert all(o["device"] == "cpu" for o in nodes[node]["outputs"].values()), node
