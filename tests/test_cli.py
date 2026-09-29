"""Tests for the karak CLI entry point."""

from __future__ import annotations

import json

import pytest

from karak.cli.main import main


def test_schema_subcommand(capsys):
    assert main(["schema"]) == 0
    palette = json.loads(capsys.readouterr().out)
    assert any(entry["id"] == "load_elements" for entry in palette)


def test_validate_subcommand(capsys):
    assert main(["validate", "--builtin", "global"]) == 0


def test_entry_point_reexport():
    from karak.cli import runner

    assert runner.main is main


def test_no_subcommand_prints_help_and_exits_2(capsys):
    assert main([]) == 2
    out = capsys.readouterr().out
    assert "karak flow init" in out
    assert "config.yaml" not in out


def test_yaml_mode_is_gone(capsys):
    assert main(["-c", "config.yaml"]) == 2
    assert "karak flow init" in capsys.readouterr().out
