"""Parity: parallel loading is byte-identical to serial loading."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import downsample_cfg, loader_cfg, make_synthetic_scene
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

    data_dir, colormap = scene_dir
    return load_element_maps(
        str(data_dir),
        downsample_cfg(header_trim_px=0, downsample_factor=1),
        exclude_elements=[],
        bse_channel="SEM",
        include_elements=None,
        loader_config=loader_cfg(colormap=colormap),
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

    data_dir, colormap = scene_dir
    calls = []
    load_element_maps(
        str(data_dir),
        downsample_cfg(header_trim_px=0, downsample_factor=1),
        exclude_elements=[],
        bse_channel="SEM",
        include_elements=None,
        loader_config=loader_cfg(colormap=colormap),
        workers=workers,
        on_file=lambda done, total, element: calls.append((done, total, element)),
    )
    assert [c[0] for c in calls] == [1, 2, 3, 4]
    assert {c[1] for c in calls} == {4}
    assert sorted(c[2] for c in calls) == ["A", "B", "C", "SEM"]


def test_loader_logs_discovery_line(scene_dir, caplog):
    import logging

    data_dir, colormap = scene_dir
    with caplog.at_level(logging.INFO, logger="karak.io.loaders"):
        load_element_maps(
            str(data_dir),
            downsample_cfg(header_trim_px=0, downsample_factor=1),
            exclude_elements=["C"],
            bse_channel="SEM",
            include_elements=None,
            loader_config=loader_cfg(colormap=colormap),
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


def _warn_and_return(value):
    import warnings

    warnings.warn("careful: big image", UserWarning)
    return value


def test_run_captured_returns_warnings_from_a_worker_process():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from karak.io.loaders import _run_captured

    ctx = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        value, messages = pool.submit(_run_captured, _warn_and_return, 7).result()
    assert value == 7
    assert messages == ["UserWarning: careful: big image"]


def test_image_warnings_are_logged_once_and_not_printed(scene_dir, caplog, monkeypatch):
    import logging
    import warnings

    from PIL import Image

    # 64x64 = 4096 px: over this limit PIL warns; the error limit is 2x.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 3000)
    data_dir, colormap = scene_dir
    with warnings.catch_warnings(record=True) as leaked, \
            caplog.at_level(logging.INFO, logger="karak.io.loaders"):
        warnings.simplefilter("always")
        load_element_maps(
            str(data_dir),
            downsample_cfg(header_trim_px=0, downsample_factor=1),
            exclude_elements=[],
            bse_channel="SEM",
            include_elements=None,
            loader_config=loader_cfg(colormap=colormap),
        )
    assert leaked == []
    bombs = [r for r in caplog.records
             if "DecompressionBombWarning" in r.getMessage()]
    assert len(bombs) == 1
    assert bombs[0].levelname == "WARNING"
    assert "s-01-A.png" in bombs[0].getMessage()
