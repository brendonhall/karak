"""Colormap (jet LUT) inversion test on a synthetic ramp.

Renders a jet-colormapped intensity ramp with matplotlib, inverts it with
karak's LUT machinery, and checks that the recovered scalar values are
monotonic and close to the true ramp.

The test uses a small (64-entry) jet palette via the ``lut:`` spec so the
one-time 256^3 lookup-table build stays fast (a full 4096-entry ``cmap:jet``
palette takes 30-60 s to index and is exercised identically by the same
code path).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

import karak.io.loaders as loaders

PALETTE_N = 64


@pytest.fixture(scope="module")
def isolated_lut_cache(tmp_path_factory):
    """Redirect the on-disk and in-memory LUT caches away from the repo.

    Module-scoped so the ~10 s one-time 256^3 LUT build is shared by all
    tests in this file instead of repeated per test.
    """
    tmp_path = tmp_path_factory.mktemp("lut")
    orig_cache_dir = loaders._CACHE_DIR
    orig_mem_cache = loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}
    yield tmp_path
    loaders._CACHE_DIR = orig_cache_dir
    loaders._FULL_LUT_CACHE = orig_mem_cache


def _jet_palette_npy(tmp_path) -> str:
    """Write a small jet palette LUT (.npy, (N, 3) uint8) and return its path."""
    scalars = np.linspace(0, 1, PALETTE_N)
    rgb = (
        np.array([matplotlib.colormaps["jet"](s)[:3] for s in scalars]) * 255
    ).astype(np.uint8)
    path = tmp_path / "jet_palette.npy"
    np.save(path, rgb)
    return str(path)


def test_jet_ramp_inversion_monotonic(isolated_lut_cache):
    tmp_path = isolated_lut_cache
    lut_path = _jet_palette_npy(tmp_path)

    # Render a jet-colormapped ramp exactly as SEM software would export it.
    true_values = np.linspace(0, 1, 256, dtype=np.float64)
    ramp_rgb = (
        np.array([matplotlib.colormaps["jet"](v)[:3] for v in true_values]) * 255
    ).astype(np.uint8)
    img = np.tile(ramp_rgb[np.newaxis, :, :], (4, 1, 1))  # (4, 256, 3)

    recovered = loaders.invert_colormap(img, f"lut:{lut_path}")

    assert recovered.shape == (4, 256)
    assert recovered.dtype == np.float32
    row = recovered[0].astype(np.float64)

    # Monotonic non-decreasing recovery along the ramp
    assert np.all(np.diff(row) >= 0), "recovered ramp is not monotonic"

    # Close to the true scalar values (nearest-neighbour palette quantization
    # limits accuracy to ~1/(2*(PALETTE_N-1)) plus colormap nonlinearity).
    max_err = np.max(np.abs(row - true_values))
    assert max_err < 0.05, f"max inversion error {max_err:.4f} exceeds tolerance"

    # End points recover exactly (palette entries coincide with ramp ends)
    assert row[0] == pytest.approx(0.0, abs=1e-6)
    assert row[-1] == pytest.approx(1.0, abs=1e-6)


def test_black_pixels_map_to_zero(isolated_lut_cache):
    """Black [0,0,0] pixels (below detection / no data) must invert to 0.0."""
    tmp_path = isolated_lut_cache
    lut_path = _jet_palette_npy(tmp_path)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    recovered = loaders.invert_colormap(img, f"lut:{lut_path}")
    assert np.all(recovered == 0.0)


def test_lut_disk_cache_reused(isolated_lut_cache):
    """Second call must reuse the on-disk cache, not rebuild the LUT."""
    tmp_path = isolated_lut_cache
    lut_path = _jet_palette_npy(tmp_path)
    spec = f"lut:{lut_path}"

    lut1 = loaders.get_full_lut(spec)
    cache_files = list((tmp_path / "lut_cache").glob("*.npy"))
    assert len(cache_files) == 1

    # Clear the in-memory cache; the disk cache must be picked up.
    loaders._FULL_LUT_CACHE.clear()
    lut2 = loaders.get_full_lut(spec)
    np.testing.assert_array_equal(lut1, lut2)
