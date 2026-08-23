"""Tests for load_elements' cache source signature."""

from __future__ import annotations

import os

from karak.stages import get


def _sig(input_dir):
    cls = get("load_elements")
    return cls.source_signature(cls.coerce_params({"input_dir": str(input_dir)}))


def test_signature_stable_for_unchanged_files(tmp_path):
    (tmp_path / "a-1-Fe.png").write_bytes(b"x")
    assert _sig(tmp_path) == _sig(tmp_path)


def test_signature_changes_when_file_added(tmp_path):
    (tmp_path / "a-1-Fe.png").write_bytes(b"x")
    before = _sig(tmp_path)
    (tmp_path / "a-1-Mg.png").write_bytes(b"y")
    assert _sig(tmp_path) != before


def test_signature_changes_when_file_modified(tmp_path):
    target = tmp_path / "a-1-Fe.png"
    target.write_bytes(b"x")
    before = _sig(tmp_path)
    target.write_bytes(b"xy")
    os.utime(target, ns=(1, 1))
    assert _sig(tmp_path) != before
