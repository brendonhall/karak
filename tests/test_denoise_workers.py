"""Parity: parallel denoising is byte-identical to serial."""

from __future__ import annotations

import numpy as np

from conftest import make_synthetic_scene
from karak.preprocessing.denoise import (
    anisotropic_denoise_cube,
    bilateral_denoise_cube,
)


def _cube_and_mask():
    cube = make_synthetic_scene()
    mask = cube.sum(axis=-1) > 0
    return cube, mask


def test_bilateral_parallel_parity():
    cube, mask = _cube_and_mask()
    serial = bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0, workers=1)
    parallel = bilateral_denoise_cube(cube, mask, sigma_color=None, sigma_spatial=1.0, workers=3)
    assert np.array_equal(serial, parallel)


def test_anisotropic_parallel_parity():
    cube, mask = _cube_and_mask()
    serial = anisotropic_denoise_cube(
        cube, mask, niter=10, kappa=50.0, gamma=0.1, option=2, workers=1)
    parallel = anisotropic_denoise_cube(
        cube, mask, niter=10, kappa=50.0, gamma=0.1, option=2, workers=3)
    assert np.array_equal(serial, parallel)


def _channel_ticks(fn, workers, **kw):
    cube, mask = _cube_and_mask()
    ticks = []
    fn(cube, mask, workers=workers, on_channel=lambda d, t, i: ticks.append((d, t, i)), **kw)
    return ticks, cube.shape[-1]


def test_bilateral_on_channel_fires_once_per_channel_serial_and_parallel():
    for workers in (1, 3):
        ticks, n = _channel_ticks(bilateral_denoise_cube, workers,
                                  sigma_color=None, sigma_spatial=1.0)
        assert [d for d, _, _ in ticks] == list(range(1, n + 1)), workers
        assert all(t == n for _, t, _ in ticks)
        assert sorted(i for _, _, i in ticks) == list(range(n))


def test_anisotropic_on_channel_fires_once_per_channel_serial_and_parallel():
    for workers in (1, 3):
        ticks, n = _channel_ticks(anisotropic_denoise_cube, workers,
                                  niter=2, kappa=50.0, gamma=0.1, option=2)
        assert [d for d, _, _ in ticks] == list(range(1, n + 1)), workers
        assert sorted(i for _, _, i in ticks) == list(range(n))


def test_denoise_stage_reports_progress_per_element():
    from karak.stages.denoise import DenoiseStage
    from karak.stages.payloads import ElementCube, MaskSet, Space

    cube, mask = _cube_and_mask()
    names = tuple(f"E{i}" for i in range(cube.shape[-1]))
    events = []

    class Recorder:
        def progress(self, node_id, done, total, msg=""):
            events.append((node_id, done, total, msg))

    stage = DenoiseStage()
    stage.reporter = Recorder()
    stage.node_id = "dn"
    stage.run({"cube": ElementCube(pixels=cube, element_names=names, space=Space.RAW),
               "masks": MaskSet(mineral_mask=mask)}, {})
    assert [e[1] for e in events] == list(range(1, len(names) + 1))
    assert all(e[0] == "dn" and e[2] == len(names) for e in events)
    assert sorted(e[3] for e in events) == sorted(names)
