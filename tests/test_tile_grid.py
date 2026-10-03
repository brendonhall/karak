"""compute_tile_grid: one sort by tile id gives the same tiles as one mask
per tile (the original implementation, kept here as the reference)."""

from __future__ import annotations

import numpy as np
import pytest

from karak.clustering.tiling import TileSpec, compute_tile_grid


def _reference_grid(mineral_indices, image_shape, tile_size, min_tile_pixels):
    H, W = image_shape
    rows, cols = mineral_indices[:, 0], mineral_indices[:, 1]
    tiles, tile_id = [], 0
    for r0 in range(0, H, tile_size):
        r1 = min(r0 + tile_size, H)
        for c0 in range(0, W, tile_size):
            c1 = min(c0 + tile_size, W)
            mask = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
            idx = np.where(mask)[0]
            if len(idx) < min_tile_pixels:
                continue
            tiles.append(TileSpec(tile_id, r0, r1, c0, c1, idx))
            tile_id += 1
    return tiles


def _indices(shape, fraction, seed):
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < fraction
    mask[: shape[0] // 3, : shape[1] // 4] = False      # an empty region
    rows, cols = np.nonzero(mask)
    return np.stack([rows, cols], axis=1).astype(np.int32)


@pytest.mark.parametrize("shape,tile_size,min_px", [
    ((100, 130), 32, 0),       # edge tiles narrower than tile_size
    ((100, 130), 32, 400),     # some tiles skipped, ids stay consecutive
    ((64, 64), 64, 1),         # one tile
    ((50, 70), 200, 0),        # tile larger than the image
    ((97, 31), 1, 1),          # one pixel per tile
])
def test_grid_matches_the_per_tile_mask_reference(shape, tile_size, min_px):
    indices = _indices(shape, 0.4, seed=tile_size)
    got = compute_tile_grid(indices, shape, tile_size, min_px)
    want = _reference_grid(indices, shape, tile_size, min_px)
    assert [(t.tile_id, t.row_start, t.row_end, t.col_start, t.col_end)
            for t in got] == [(t.tile_id, t.row_start, t.row_end,
                               t.col_start, t.col_end) for t in want]
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g.pixel_indices, w.pixel_indices)
        assert g.pixel_indices.dtype == w.pixel_indices.dtype


def test_grid_keeps_the_input_order_when_indices_are_not_sorted():
    indices = _indices((60, 60), 0.5, seed=1)
    perm = np.random.default_rng(2).permutation(len(indices))
    shuffled = indices[perm]
    got = compute_tile_grid(shuffled, (60, 60), 16, 0)
    want = _reference_grid(shuffled, (60, 60), 16, 0)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g.pixel_indices, w.pixel_indices)


def test_grid_without_mineral_pixels_matches_the_reference():
    empty = np.zeros((0, 2), np.int32)
    # min_tile_pixels=0 keeps the empty tiles, as the reference does
    got = compute_tile_grid(empty, (10, 10), 4, 0)
    assert [(t.tile_id, t.pixel_indices.size) for t in got] == [
        (t.tile_id, t.pixel_indices.size)
        for t in _reference_grid(empty, (10, 10), 4, 0)]
    assert compute_tile_grid(empty, (10, 10), 4, 1) == []
