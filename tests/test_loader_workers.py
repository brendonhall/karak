"""Parity: parallel loading is byte-identical to serial loading."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.config import DownsampleConfig
from karak.io.loaders import load_element_maps


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    """Palette PNGs + grayscale SEM (same recipe as test_flow_end_to_end)."""
    import imageio.v2 as imageio
    import matplotlib

    import karak.io.loaders as loaders

    tmp_path = tmp_path_factory.mktemp("loader_scene")
    orig_dir, orig_mem = loaders._CACHE_DIR, loaders._FULL_LUT_CACHE
    loaders._CACHE_DIR = tmp_path / "lut_cache"
    loaders._FULL_LUT_CACHE = {}

    palette = (
        np.array(
            [matplotlib.colormaps["jet"](s)[:3] for s in np.linspace(0, 1, 64)]
        ) * 255
    ).astype(np.uint8)
    lut_path = tmp_path / "palette.npy"
    np.save(lut_path, palette)

    cube = make_synthetic_scene()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i, element in enumerate(("A", "B", "C")):
        indices = np.round(cube[:, :, i] * 63).astype(np.uint8)
        imageio.imwrite(data_dir / f"s-01-{element}.png", palette[indices])
    imageio.imwrite(data_dir / "s-01-SEM.png", np.zeros((64, 64), dtype=np.uint8))

    yield data_dir, f"lut:{lut_path}"

    loaders._CACHE_DIR, loaders._FULL_LUT_CACHE = orig_dir, orig_mem


def _load(scene_dir, workers):
    from karak.config import LoaderConfig

    data_dir, colormap = scene_dir
    return load_element_maps(
        str(data_dir),
        DownsampleConfig(header_trim_px=0, downsample_factor=1),
        exclude_elements=[],
        loader_config=LoaderConfig(colormap=colormap),
        workers=workers,
    )


def test_parallel_load_is_byte_identical(scene_dir):
    e1, b1, n1 = _load(scene_dir, workers=1)
    e4, b4, n4 = _load(scene_dir, workers=4)
    assert n1 == n4
    assert np.array_equal(b1, b4)
    for name in n1:
        assert np.array_equal(e1[name], e4[name]), name


@pytest.mark.parametrize("workers", [1, 2])
def test_on_file_fires_once_per_file(scene_dir, workers):
    from karak.config import LoaderConfig

    data_dir, colormap = scene_dir
    calls = []
    load_element_maps(
        str(data_dir),
        DownsampleConfig(header_trim_px=0, downsample_factor=1),
        exclude_elements=[],
        loader_config=LoaderConfig(colormap=colormap),
        workers=workers,
        on_file=lambda done, total, element: calls.append((done, total, element)),
    )
    assert [c[0] for c in calls] == [1, 2, 3, 4]
    assert {c[1] for c in calls} == {4}
    assert sorted(c[2] for c in calls) == ["A", "B", "C", "SEM"]


def test_loader_logs_discovery_line(scene_dir, caplog):
    import logging

    from karak.config import LoaderConfig

    data_dir, colormap = scene_dir
    with caplog.at_level(logging.INFO, logger="karak.io.loaders"):
        load_element_maps(
            str(data_dir),
            DownsampleConfig(header_trim_px=0, downsample_factor=1),
            exclude_elements=["C"],
            loader_config=LoaderConfig(colormap=colormap),
        )
    assert (
        "Found 4 matching files: 3 to load (BSE channel 'SEM'), excluded: C"
        in caplog.text
    )


def test_load_stage_reports_progress(scene_dir):
    from karak.stages.load import LoadElementsStage

    data_dir, colormap = scene_dir
    events = []

    class Recorder:
        def progress(self, node_id, done, total, msg=""):
            events.append((node_id, done, total, msg))

    stage = LoadElementsStage()
    stage.reporter = Recorder()
    stage.node_id = "src"
    stage.run({}, {"input_dir": str(data_dir), "colormap": colormap,
                   "exclude_elements": ""})
    assert [e[1] for e in events] == [1, 2, 3, 4]
    assert all(e[0] == "src" and e[2] == 4 for e in events)
