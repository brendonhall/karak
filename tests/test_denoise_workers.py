"""Parity: parallel denoising is byte-identical to serial."""

from __future__ import annotations

import numpy as np
import pytest

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


def _reference_cube(cube, mask, method, sigma_spatial=1.0):
    """What denoise_cube must produce for the new methods: per channel, the
    mask filled with the mineral mean, the numpy reference, then re-masked."""
    from karak.preprocessing.bilateral import bilateral_numpy

    guide = None
    if method == "joint_bilateral_total":
        guide = cube.sum(axis=-1).astype(np.float32)
        guide[~mask] = guide[mask].mean()
    out = np.zeros_like(cube)
    for i in range(cube.shape[-1]):
        ch = cube[:, :, i].copy()
        ch[~mask] = np.nanmean(ch[mask])
        out[:, :, i] = bilateral_numpy(ch, ch if guide is None else guide,
                                       sigma_color=None, sigma_spatial=sigma_spatial)
    out[~mask] = 0.0
    return out


@pytest.mark.parametrize("method", ["bilateral_sym", "joint_bilateral_total"])
@pytest.mark.parametrize("workers", [1, 3])
def test_new_bilateral_methods_match_the_reference_serial_and_parallel(method, workers):
    from conftest import denoise_cfg
    from karak.preprocessing.denoise import denoise_cube

    cube, mask = _cube_and_mask()
    cfg = denoise_cfg(method=method, sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    out = denoise_cube(cube, mask, cfg, workers=workers)
    np.testing.assert_allclose(out, _reference_cube(cube, mask, method), atol=1e-6)


def test_joint_bilateral_bse_uses_the_given_guide():
    from conftest import denoise_cfg
    from karak.preprocessing.bilateral import bilateral_numpy
    from karak.preprocessing.denoise import denoise_cube

    cube, mask = _cube_and_mask()
    bse = np.random.default_rng(7).random(cube.shape[:2], dtype=np.float32)
    cfg = denoise_cfg(method="joint_bilateral_bse", sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    out = denoise_cube(cube, mask, cfg, guide=bse)
    guide = bse.copy()
    guide[~mask] = guide[mask].mean()
    ch = cube[:, :, 0].copy()
    ch[~mask] = np.nanmean(ch[mask])
    expected = bilateral_numpy(ch, guide, sigma_color=None, sigma_spatial=1.0)
    expected[~mask] = 0.0
    np.testing.assert_allclose(out[:, :, 0], expected, atol=1e-6)


def test_joint_bilateral_bse_without_a_guide_is_a_stage_error():
    from conftest import denoise_cfg
    from karak.preprocessing.denoise import denoise_cube
    from karak.stages.base import StageError

    cube, mask = _cube_and_mask()
    cfg = denoise_cfg(method="joint_bilateral_bse", sigma_color=None, sigma_spatial=1.0,
                      niter=10, kappa=50.0, gamma=0.1, option=2)
    with pytest.raises(StageError, match="bse"):
        denoise_cube(cube, mask, cfg)


def test_stage_wires_the_optional_bse_port_to_the_joint_guide():
    from karak.stages.base import StageError
    from karak.stages.denoise import DenoiseStage
    from karak.stages.payloads import BseImage, ElementCube, MaskSet, Space

    cube, mask = _cube_and_mask()
    names = tuple(f"E{i}" for i in range(cube.shape[-1]))
    stage = DenoiseStage()
    assert [p.name for p in stage.INPUTS] == ["cube", "masks", "bse"]
    assert not [p for p in stage.INPUTS if p.name == "bse"][0].required
    inputs = {"cube": ElementCube(pixels=cube, element_names=names, space=Space.RAW),
              "masks": MaskSet(mineral_mask=mask)}
    with pytest.raises(StageError, match="src.bse -> dn.bse"):
        stage.run(inputs, {"method": "joint_bilateral_bse"})
    bse = np.random.default_rng(8).random(cube.shape[:2], dtype=np.float32)
    out = stage.run({**inputs, "bse": BseImage(pixels=bse)},
                    {"method": "joint_bilateral_bse"})
    assert out["cube"].space is Space.DENOISED
    assert out["cube"].pixels.shape == cube.shape
