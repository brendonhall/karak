"""The shipped stage reference doc must match the live registry."""

from __future__ import annotations

from pathlib import Path

from karak.stages import list_stages
from karak.stages.reference import render_markdown

DOC = Path(__file__).parent.parent / "docs" / "stage_reference.md"


def test_every_registered_stage_documented():
    text = render_markdown()
    for schema in list_stages():
        assert f"## `{schema['id']}`" in text


def test_ports_and_params_rendered():
    text = render_markdown()
    # a producing stage with typed ports
    assert "`cube` | `denoised`" in text
    # an optional port is marked
    assert "no" in text  # required column for optional export inputs
    # a param with bounds shows its default
    assert "min_cluster_size" in text


def test_shipped_doc_matches_registry():
    """Regenerate with: uv run python -m karak.stages.reference"""
    assert DOC.read_text() == render_markdown()
