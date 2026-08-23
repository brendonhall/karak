"""Parity tests: each stage's run() must equal the core function it wraps."""

from __future__ import annotations

import numpy as np
import pytest

from karak.config import DenoiseConfig, DownsampleConfig, LoaderConfig
from karak.io.loaders import build_compositional_cube, load_element_maps
from karak.io.masks import compute_mask_statistics, create_mineral_mask
from karak.preprocessing.compositional import zscore_normalize
from karak.preprocessing.denoise import denoise_cube
from karak.stages import get
from karak.stages.payloads import BseImage, ElementCube, MaskSet, Space


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_cube(synthetic_scene):
    return ElementCube(
        pixels=synthetic_scene,
        element_names=("A", "B", "C"),
        space=Space.RAW,
    )


@pytest.fixture()
def mask_set(synthetic_scene):
    mineral = create_mineral_mask(synthetic_scene, None, min_object_size=10)
    return MaskSet(
        mineral_mask=mineral,
        valid_mask=None,
        stats=compute_mask_statistics(mineral, None),
    )


@pytest.fixture(scope="module")
def isolated_lut_cache(tmp_path_factory):
    """Redirect the on-disk and in-memory LUT caches away from the repo."""
    import karak.io.loaders as loaders

    tmp_path = tmp_path_factory.mktemp("lut")
    orig_cache_dir = loaders._CACHE_DIR
    orig_mem_cache = loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}
    yield tmp_path
    loaders._CACHE_DIR = orig_cache_dir
    loaders._FULL_LUT_CACHE = orig_mem_cache


@pytest.fixture()
def png_dir(tmp_path, isolated_lut_cache):
    """Tiny legacy-convention dataset: 2 palette element maps + grayscale SEM.

    Uses a small 64-entry jet palette via the ``lut:`` spec so the one-time
    256^3 LUT build stays fast (full ``cmap:jet`` takes 30-60 s).
    """
    import imageio.v2 as imageio
    import matplotlib

    palette = (
        np.array(
            [matplotlib.colormaps["jet"](s)[:3] for s in np.linspace(0, 1, 64)]
        ) * 255
    ).astype(np.uint8)
    lut_path = tmp_path / "jet_palette.npy"
    np.save(lut_path, palette)

    rng = np.random.default_rng(1)
    for element in ("Fe", "Mg"):
        indices = rng.integers(0, 64, size=(16, 16))
        imageio.imwrite(tmp_path / f"samp-01-{element}.png", palette[indices])
    gray = (rng.random((16, 16)) * 255).astype(np.uint8)
    imageio.imwrite(tmp_path / "samp-01-SEM.png", gray)
    return tmp_path, f"lut:{lut_path}"


# ---------------------------------------------------------------------------
# load_elements
# ---------------------------------------------------------------------------

def test_load_elements_parity(png_dir):
    input_dir, colormap = png_dir
    downsample = DownsampleConfig(
        header_trim_px=0, bottom_trim_px=0, left_trim_px=0, right_trim_px=0,
        downsample_factor=1,
    )
    elements, bse, names = load_element_maps(
        str(input_dir), downsample, exclude_elements=[], bse_channel="SEM",
        loader_config=LoaderConfig(colormap=colormap),
    )
    expected_cube = build_compositional_cube(elements, names)

    out = get("load_elements")().run({}, {
        "input_dir": str(input_dir),
        "colormap": colormap,
        "exclude_elements": "",
        "header_trim_px": 0,
        "downsample_factor": 1,
    })

    cube, bse_out = out["cube"], out["bse"]
    assert isinstance(cube, ElementCube)
    assert isinstance(bse_out, BseImage)
    assert cube.space is Space.RAW
    assert cube.element_names == tuple(names)
    np.testing.assert_array_equal(cube.pixels, expected_cube)
    np.testing.assert_array_equal(bse_out.pixels, bse)
    # geometry metadata travels in the payload
    assert cube.downsample_factor == 1
    assert cube.header_trim_px == 0


def test_load_elements_records_geometry(png_dir):
    input_dir, colormap = png_dir
    out = get("load_elements")().run({}, {
        "input_dir": str(input_dir),
        "colormap": colormap,
        "exclude_elements": "",
        "header_trim_px": 4,
        "downsample_factor": 2,
    })
    assert out["cube"].downsample_factor == 2
    assert out["cube"].header_trim_px == 4


# ---------------------------------------------------------------------------
# mask
# ---------------------------------------------------------------------------

def test_mask_parity(raw_cube, synthetic_scene):
    expected = create_mineral_mask(synthetic_scene, None, min_object_size=10)
    expected_stats = compute_mask_statistics(expected, None)

    out = get("mask")().run({"cube": raw_cube}, {"min_object_size": 10})

    masks = out["masks"]
    assert isinstance(masks, MaskSet)
    np.testing.assert_array_equal(masks.mineral_mask, expected)
    assert masks.valid_mask is None
    assert masks.stats == expected_stats


def test_mask_rejects_denoised_cube(raw_cube):
    from karak.stages.base import StageError

    with pytest.raises(StageError):
        get("mask")().run({"cube": raw_cube.replace(space=Space.DENOISED)})


# ---------------------------------------------------------------------------
# denoise
# ---------------------------------------------------------------------------

def test_denoise_parity(raw_cube, mask_set, synthetic_scene):
    expected = denoise_cube(
        synthetic_scene, mask_set.mineral_mask, DenoiseConfig(method="bilateral")
    )

    out = get("denoise")().run(
        {"cube": raw_cube, "masks": mask_set}, {"method": "bilateral"}
    )

    cube = out["cube"]
    assert cube.space is Space.DENOISED
    np.testing.assert_array_equal(cube.pixels, expected)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

def test_normalize_parity(raw_cube, mask_set, synthetic_scene):
    denoised = raw_cube.replace(space=Space.DENOISED)
    expected, means, stds = zscore_normalize(synthetic_scene, mask_set.mineral_mask)

    out = get("normalize")().run({"cube": denoised, "masks": mask_set})

    cube = out["cube"]
    assert cube.space is Space.NORMALIZED
    np.testing.assert_array_equal(cube.pixels, expected)
    np.testing.assert_array_equal(cube.means, means)
    np.testing.assert_array_equal(cube.stds, stds)
