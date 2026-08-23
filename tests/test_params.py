"""Tests for karak.stages.base.Param coercion."""

from __future__ import annotations

import pytest

from karak.stages.base import Param


def test_none_returns_default():
    p = Param("degree", "int", 3)
    assert p.coerce(None) == 3


def test_float_casts_from_int_and_str():
    p = Param("sigma", "float", 1.0)
    assert p.coerce(2) == 2.0
    assert isinstance(p.coerce(2), float)
    assert p.coerce("2.5") == 2.5


def test_int_casts_from_str():
    p = Param("size", "int", 100)
    assert p.coerce("7") == 7
    assert isinstance(p.coerce("7"), int)


def test_int_rejects_fractional_float():
    p = Param("size", "int", 100)
    with pytest.raises(ValueError):
        p.coerce(2.5)


def test_bool_accepts_bool_and_common_strings():
    p = Param("enabled", "bool", False)
    assert p.coerce(True) is True
    assert p.coerce("true") is True
    assert p.coerce("false") is False


def test_enum_enforces_choices():
    p = Param("method", "enum", "bilateral",
              choices=("bilateral", "anisotropic_diffusion"))
    assert p.coerce("anisotropic_diffusion") == "anisotropic_diffusion"
    with pytest.raises(ValueError):
        p.coerce("gaussian")


def test_bounds_enforced():
    p = Param("k", "int", 5, min=1, max=64)
    assert p.coerce(64) == 64
    with pytest.raises(ValueError):
        p.coerce(0)
    with pytest.raises(ValueError):
        p.coerce(65)


def test_optional_param_none_default_passes_through():
    p = Param("sigma_color", "float", None)
    assert p.coerce(None) is None
    assert p.coerce("0.3") == 0.3


def test_str_param():
    p = Param("path", "str", None)
    assert p.coerce(None) is None
    assert p.coerce(12) == "12"
