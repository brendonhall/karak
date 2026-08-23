"""Tests for payload dataclasses and their HDF5 round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from karak.stages.payloads import (
    BseImage,
    ElementCube,
    MaskSet,
    Space,
    payload_from_h5,
)


def _cube():
    return ElementCube(
        pixels=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        element_names=("Fe", "Mg", "Ca", "Si"),
        space=Space.RAW,
        downsample_factor=2,
        header_trim_px=100,
    )


def test_replace_returns_new_instance():
    cube = _cube()
    denoised = cube.replace(space=Space.DENOISED)
    assert denoised.space is Space.DENOISED
    assert cube.space is Space.RAW  # original untouched


def test_payloads_are_frozen():
    with pytest.raises(Exception):
        _cube().pixels = None  # type: ignore[misc]


def _roundtrip(payload, tmp_path):
    import h5py

    path = tmp_path / "payload.h5"
    with h5py.File(path, "w") as fh:
        payload.to_h5(fh.create_group("p"))
    with h5py.File(path, "r") as fh:
        return payload_from_h5(fh["p"])


def test_element_cube_h5_roundtrip(tmp_path):
    cube = _cube().replace(
        space=Space.NORMALIZED,
        means=np.ones(4, dtype=np.float32),
        stds=np.full(4, 2.0, dtype=np.float32),
    )
    back = _roundtrip(cube, tmp_path)
    assert isinstance(back, ElementCube)
    np.testing.assert_array_equal(back.pixels, cube.pixels)
    assert back.element_names == cube.element_names
    assert back.space is Space.NORMALIZED
    np.testing.assert_array_equal(back.means, cube.means)
    assert back.downsample_factor == 2
    assert back.header_trim_px == 100


def test_element_cube_h5_roundtrip_none_means(tmp_path):
    back = _roundtrip(_cube(), tmp_path)
    assert back.means is None
    assert back.stds is None


def test_bse_image_h5_roundtrip(tmp_path):
    bse = BseImage(pixels=np.eye(3, dtype=np.float32))
    back = _roundtrip(bse, tmp_path)
    assert isinstance(back, BseImage)
    np.testing.assert_array_equal(back.pixels, bse.pixels)


def test_mask_set_h5_roundtrip(tmp_path):
    masks = MaskSet(
        mineral_mask=np.array([[True, False], [False, True]]),
        valid_mask=None,
        stats={"n_mineral": 2, "pct": 50.0},
    )
    back = _roundtrip(masks, tmp_path)
    assert isinstance(back, MaskSet)
    np.testing.assert_array_equal(back.mineral_mask, masks.mineral_mask)
    assert back.valid_mask is None
    assert back.stats == masks.stats
