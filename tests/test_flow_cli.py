"""Tests for the python -m karak.flow CLI subcommands."""

from __future__ import annotations

import json

import pytest

from karak.flow.__main__ import main


def test_schema_prints_palette(capsys):
    assert main(["schema"]) == 0
    palette = json.loads(capsys.readouterr().out)
    ids = {entry["id"] for entry in palette}
    assert "load_elements" in ids
    assert "export_h5" in ids


def test_validate_builtin_ok(capsys):
    assert main(["validate", "--builtin", "global"]) == 0
    assert "0 errors" in capsys.readouterr().out


def test_validate_broken_flow_fails(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "version": 1, "name": "bad",
        "nodes": [{"id": "a", "type": "no_such_stage", "params": {}}],
        "edges": [],
    }))
    assert main(["validate", str(bad)]) == 1
    assert "unknown stage" in capsys.readouterr().out


def test_run_requires_flow_or_builtin():
    with pytest.raises(SystemExit):
        main(["run"])


def test_run_parser_accepts_workers():
    from karak.flow.__main__ import build_parser

    args = build_parser().parse_args(
        ["run", "--builtin", "global", "--workers", "4"]
    )
    assert args.workers == 4


def test_run_parser_workers_default_none():
    from karak.flow.__main__ import build_parser

    args = build_parser().parse_args(["run", "--builtin", "global"])
    assert args.workers is None


def test_run_parser_accepts_device():
    from karak.flow.__main__ import build_parser

    args = build_parser().parse_args(
        ["run", "--builtin", "global", "--device", "cuda"]
    )
    assert args.device == "cuda"


def test_cuda_preflight_rejects_without_gpu(monkeypatch, tmp_path):
    import karak.accel as accel
    from karak.flow.__main__ import main as flow_main
    from karak.stages import registry
    from karak.stages.base import Param, Port, Stage

    class FakeCudaStage(Stage):
        id = "fake_cuda_stage"
        label = "Fake CUDA stage"
        OUTPUTS = [Port("x")]
        PARAMS = [Param("device", "str", "cpu", choices=("cpu", "cuda"))]

        def apply(self, inputs, params):
            return {}

    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    registry.register(FakeCudaStage)
    try:
        flow_path = tmp_path / "flow.json"
        flow_path.write_text(
            '{"version": 1, "name": "g", '
            '"nodes": [{"id": "a", "type": "fake_cuda_stage", "params": {}}], '
            '"edges": []}'
        )
        with pytest.raises(SystemExit, match="cuda"):
            flow_main(["run", str(flow_path), "--device", "cuda",
                       "--out", str(tmp_path / "o")])
    finally:
        registry._REGISTRY.pop(FakeCudaStage.id, None)


def _stepwise_sparse(path):
    path.write_text(json.dumps({
        "version": 1, "name": "stepwise",
        "nodes": [{"id": "src", "type": "load_elements",
                   "params": {"input_dir": "{input}", "downsample_factor": 4}}],
        "edges": [],
    }))


def test_flow_init_writes_a_complete_builtin(tmp_path, capsys):
    from karak.stages.load import LoadElementsStage

    out = tmp_path / "mine.json"
    assert main(["flow", "init", "--builtin", "stepwise", "-o", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["version"] == 2
    assert data["nodes"][0]["params"] == LoadElementsStage.template()
    assert str(out) in capsys.readouterr().out


def test_flow_init_refuses_to_overwrite_without_force(tmp_path):
    out = tmp_path / "mine.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        main(["flow", "init", "--builtin", "stepwise", "-o", str(out)])
    assert out.read_text() == "{}"
    assert main(["flow", "init", "--builtin", "stepwise", "-o", str(out),
                 "--force"]) == 0


def test_flow_complete_upgrades_a_sparse_file_in_place(tmp_path, capsys):
    flow = tmp_path / "old.json"
    _stepwise_sparse(flow)
    assert main(["flow", "complete", str(flow)]) == 0
    data = json.loads(flow.read_text())
    params = data["nodes"][0]["params"]
    assert data["version"] == 2
    assert params["downsample_factor"] == 4
    assert params["colormap"] == "tima:jet"
    out = capsys.readouterr().out
    assert "src: added 11 params" in out
    assert "colormap" in out


def test_flow_complete_to_another_file_keeps_the_original(tmp_path):
    flow, out = tmp_path / "old.json", tmp_path / "new.json"
    _stepwise_sparse(flow)
    before = flow.read_text()
    assert main(["flow", "complete", str(flow), "-o", str(out)]) == 0
    assert flow.read_text() == before
    assert json.loads(out.read_text())["version"] == 2


def test_flow_complete_on_a_complete_file_reports_nothing_added(tmp_path, capsys):
    out = tmp_path / "mine.json"
    main(["flow", "init", "--builtin", "tiled", "-o", str(out)])
    capsys.readouterr()
    assert main(["flow", "complete", str(out)]) == 0
    assert "already complete" in capsys.readouterr().out


def test_run_set_of_an_undeclared_param_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="has no param 'bogus'"):
        main(["run", "--builtin", "stepwise", "--out", str(tmp_path / "o"),
              "--set", "src.bogus=1"])
