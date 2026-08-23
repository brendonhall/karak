"""Stage registry: @register decorator + discovery helpers."""

from __future__ import annotations

_REGISTRY: dict = {}


def register(cls):
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate stage id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls


def get(stage_id: str):
    return _REGISTRY[stage_id]


def list_stages() -> list[dict]:
    """The JSON node palette: every registered stage's schema."""
    return [cls.schema() for cls in _REGISTRY.values()]
