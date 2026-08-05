"""Mask utility tests on tiny synthetic inputs."""

from __future__ import annotations

import numpy as np

from karak.io.masks import create_mineral_mask, load_valid_mask


def _write_napari_csv(path, shapes):
    """Write a napari shapes CSV.

    shapes : list of (shape_type, [(row, col), ...])
    """
    lines = ["index,shape-type,vertex-index,axis-0,axis-1"]
    for idx, (shape_type, vertices) in enumerate(shapes):
        for v_idx, (r, c) in enumerate(vertices):
            lines.append(f"{idx},{shape_type},{v_idx},{r},{c}")
    path.write_text("\n".join(lines) + "\n")


def test_load_valid_mask_square_polygon(tmp_path):
    csv_path = tmp_path / "valid.csv"
    square = [(5.0, 5.0), (5.0, 15.0), (15.0, 15.0), (15.0, 5.0)]
    _write_napari_csv(csv_path, [("polygon", square)])

    mask = load_valid_mask(csv_path, image_shape=(20, 20))

    assert mask.shape == (20, 20)
    assert mask.dtype == bool
    # Interior is inside, corners are outside
    assert mask[10, 10]
    assert not mask[0, 0]
    assert not mask[19, 19]
    assert not mask[2, 10]
    # Roughly an 11x11 filled square (rasterization may trim edges)
    assert 80 <= mask.sum() <= 140


def test_load_valid_mask_skips_path_shapes(tmp_path):
    """Open polylines ('path' shapes) must not be rasterized."""
    csv_path = tmp_path / "valid.csv"
    square = [(2.0, 2.0), (2.0, 8.0), (8.0, 8.0), (8.0, 2.0)]
    open_path = [(12.0, 2.0), (12.0, 18.0), (18.0, 18.0)]
    _write_napari_csv(csv_path, [("polygon", square), ("path", open_path)])

    mask = load_valid_mask(csv_path, image_shape=(20, 20))

    assert mask[5, 5]  # inside polygon
    assert not mask[15, 10]  # region touched only by the path shape


def test_load_valid_mask_downsample_scaling(tmp_path):
    """Vertices in full-res coordinates scale by the downsample factor."""
    csv_path = tmp_path / "valid.csv"
    # Full-res square (10,10)-(30,30) -> downsampled by 2 -> (5,5)-(15,15)
    square = [(10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0)]
    _write_napari_csv(csv_path, [("polygon", square)])

    mask = load_valid_mask(csv_path, image_shape=(20, 20), downsample_factor=2)

    assert mask[10, 10]
    assert not mask[2, 2]


def test_create_mineral_mask_zero_pixels_excluded():
    cube = np.zeros((20, 20, 3), dtype=np.float32)
    cube[5:15, 5:15, :] = 0.5  # mineral block

    mask = create_mineral_mask(cube, valid_mask=None, min_object_size=10)

    assert mask.shape == (20, 20)
    assert mask[10, 10]
    assert not mask[0, 0]
    assert mask.sum() == 100


def test_create_mineral_mask_respects_valid_mask():
    cube = np.zeros((20, 20, 3), dtype=np.float32)
    cube[5:15, 5:15, :] = 0.5

    valid = np.zeros((20, 20), dtype=bool)
    valid[:, :10] = True  # only left half of the image is valid

    mask = create_mineral_mask(cube, valid_mask=valid, min_object_size=10)

    assert mask[10, 7]  # mineral AND valid
    assert not mask[10, 12]  # mineral but outside valid region
    assert mask.sum() == 50
