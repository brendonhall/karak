"""Bilateral filter core: numpy reference against scikit-image, symmetry,
the joint (guided) variant, and the CuPy kernel against the reference."""

from __future__ import annotations

import numpy as np
import pytest
from skimage.restoration import denoise_bilateral

from karak.accel import cuda_available
from karak.preprocessing.bilateral import (
    bilateral_numpy,
    color_lut,
    spatial_weights,
    window_size,
)


def _image(seed=0, shape=(40, 50)):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def test_window_size_matches_scikit_image():
    assert window_size(1.0) == 7
    assert window_size(0.5) == 5
    assert window_size(2.0) == 13


def test_symmetric_spatial_weights_peak_at_the_centre():
    w = spatial_weights(1.0, exact_skimage=False)
    assert w.shape == (7, 7)
    assert w[3, 3] == 1.0
    np.testing.assert_allclose(w, w[::-1, ::-1])
    np.testing.assert_allclose(w[3, 4], np.exp(-0.5))


def test_scikit_image_spatial_weights_reproduce_the_stride_quirk():
    # scikit-image builds an 8x8 table for a 7x7 window and reads it with
    # stride 7, so the unit weight sits at offset (+2, -2), not the centre.
    w = spatial_weights(1.0, exact_skimage=True)
    assert w.shape == (7, 7)
    assert w[5, 1] == 1.0
    assert w[3, 3] < 1e-3
    assert (w[0] < 1e-3).all()


def test_color_lut_matches_scikit_image():
    from skimage.restoration._denoise import _compute_color_lut

    ours = color_lut(0.2, 0.9)
    theirs = _compute_color_lut(10000, 0.2, 0.9, dtype=np.float32)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("sigma_color", [None, 0.1])
def test_numpy_reference_matches_scikit_image(seed, sigma_color):
    img = _image(seed)
    ours = bilateral_numpy(img, img, sigma_color=sigma_color, sigma_spatial=1.0,
                           exact_skimage=True)
    theirs = denoise_bilateral(img, sigma_color=sigma_color, sigma_spatial=1.0)
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, theirs, atol=1e-5)


def test_symmetric_filter_commutes_with_flips_and_scikit_image_does_not():
    img = _image(3)
    sym = bilateral_numpy(img, img, sigma_color=0.1, sigma_spatial=1.0)
    sym_flipped = bilateral_numpy(img[::-1, ::-1], img[::-1, ::-1],
                                  sigma_color=0.1, sigma_spatial=1.0)
    np.testing.assert_allclose(sym, sym_flipped[::-1, ::-1], atol=1e-6)
    quirk = bilateral_numpy(img, img, sigma_color=0.1, sigma_spatial=1.0,
                            exact_skimage=True)
    assert np.abs(quirk - sym).max() > 0.05


def test_joint_filter_keeps_an_edge_the_channel_cannot_see():
    # The channel is flat noise; the guide has a sharp step. A joint filter
    # must not average across the step, a plain one does.
    rng = np.random.default_rng(4)
    channel = (0.5 + 0.05 * rng.standard_normal((30, 30))).astype(np.float32)
    channel[:, 15:] += 0.02   # a step far below the noise
    guide = np.zeros((30, 30), np.float32)
    guide[:, 15:] = 1.0
    joint = bilateral_numpy(channel, guide, sigma_color=0.1, sigma_spatial=1.0)
    plain = bilateral_numpy(channel, channel, sigma_color=None, sigma_spatial=1.0)
    step_joint = joint[:, 15:].mean() - joint[:, :15].mean()
    step_plain = plain[:, 15:].mean() - plain[:, :15].mean()
    assert step_joint > 0.015
    assert step_joint > step_plain
    # interior rows: the zero border darkens the first and last three rows
    assert joint[5:25, 3:12].std() < channel[5:25, 3:12].std() / 2


def test_constant_image_is_returned_unchanged():
    img = np.full((8, 9), 0.3, np.float32)
    np.testing.assert_array_equal(bilateral_numpy(img, img, sigma_color=None,
                                                  sigma_spatial=1.0), img)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("exact_skimage", [True, False])
@pytest.mark.parametrize("joint", [False, True])
def test_cupy_kernel_matches_the_numpy_reference(exact_skimage, joint):
    import cupy as cp

    from karak.preprocessing.bilateral import bilateral_cupy

    img = _image(5, (64, 70))
    guide = _image(6, (64, 70)) if joint else img
    ref = bilateral_numpy(img, guide, sigma_color=None, sigma_spatial=1.0,
                          exact_skimage=exact_skimage)
    gpu = bilateral_cupy(cp.asarray(img), cp.asarray(guide), sigma_color=None,
                         sigma_spatial=1.0, exact_skimage=exact_skimage)
    assert gpu.dtype == cp.float32
    np.testing.assert_allclose(cp.asnumpy(gpu), ref, atol=1e-5)
