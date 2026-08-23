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
