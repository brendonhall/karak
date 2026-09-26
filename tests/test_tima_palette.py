"""The built-in TIMA jet palette (``'tima:jet'``).

TIMA renders element maps through a fixed 256-entry palette: black at
index 0 (zero counts), then a jet ramp from (0, 0, 132) to (128, 0, 0) in
steps of about 4. Matplotlib's jet cuts the cyan and yellow corners of
that ramp and places its breakpoints differently, so inverting TIMA
exports against it merges neighbouring levels and shifts others by up to
8 levels. The palette ships with karak, recovered from the
NWA 4587 exports, where all 256 entries occur.
"""

from __future__ import annotations

import numpy as np
import pytest

import karak.io.loaders as loaders
from karak.config import DownsampleConfig, LoaderConfig


@pytest.fixture(scope="module")
def isolated_lut_cache(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("lut")
    orig_cache_dir = loaders._CACHE_DIR
    orig_mem_cache = loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}
    yield tmp_path
    loaders._CACHE_DIR = orig_cache_dir
    loaders._FULL_LUT_CACHE = orig_mem_cache


def test_tima_palette_shape_and_endpoints():
    rgb, scalars = loaders._resolve_palette("tima:jet")
    assert rgb.shape == (256, 3) and rgb.dtype == np.uint8
    assert len(np.unique(rgb, axis=0)) == 256
    assert rgb[0].tolist() == [0, 0, 0]
    assert rgb[1].tolist() == [0, 0, 132]
    assert rgb[-1].tolist() == [128, 0, 0]
    np.testing.assert_array_equal(scalars, np.linspace(0, 1, 256, dtype=np.float32))


def test_tima_palette_is_one_continuous_ramp():
    rgb, _ = loaders._resolve_palette("tima:jet")
    steps = np.abs(np.diff(rgb[1:].astype(int), axis=0)).sum(axis=1)
    assert steps.min() >= 3 and steps.max() <= 10


def test_tima_palette_reaches_the_corners_matplotlib_cuts():
    rgb, _ = loaders._resolve_palette("tima:jet")
    colors = {tuple(c) for c in rgb.tolist()}
    assert (0, 253, 255) in colors  # cyan corner
    assert (255, 253, 0) in colors  # yellow corner
    assert (255, 0, 0) in colors  # pure red


def test_tima_inversion_recovers_every_level_exactly(isolated_lut_cache):
    rgb, scalars = loaders._resolve_palette("tima:jet")
    image = rgb[None, :, :]  # one row holding all 256 palette colors
    recovered = loaders.invert_colormap(image, colormap_spec="tima:jet")
    np.testing.assert_array_equal(recovered[0], scalars)


def test_loader_defaults_match_tima_exports():
    assert LoaderConfig().colormap == "tima:jet"
    assert DownsampleConfig().header_trim_px == 0
