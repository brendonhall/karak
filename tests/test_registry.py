"""Tests for stage registration and discovery."""

from __future__ import annotations

import json

import pytest

from karak.stages import registry
from karak.stages.base import Stage


def test_register_and_get():
    @registry.register
    class TempStage(Stage):
        id = "temp_test_stage"
        label = "Temp"

    try:
        assert registry.get("temp_test_stage") is TempStage
    finally:
        registry._REGISTRY.pop("temp_test_stage", None)


def test_duplicate_id_rejected():
    @registry.register
    class TempStage(Stage):
        id = "temp_dup_stage"

    try:
        with pytest.raises(ValueError):
            @registry.register
            class TempStage2(Stage):
                id = "temp_dup_stage"
    finally:
        registry._REGISTRY.pop("temp_dup_stage", None)


def test_autoload_finds_phase_a_stages():
    import karak.stages  # noqa: F401 — package import fires autoload

    ids = {s["id"] for s in registry.list_stages()}
    assert {"load_elements", "mask", "denoise", "normalize"} <= ids


def test_list_stages_is_json_serializable():
    import karak.stages  # noqa: F401

    json.dumps(registry.list_stages())  # must not raise
