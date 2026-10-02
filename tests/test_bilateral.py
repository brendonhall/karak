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


# --- review fixes: constant joint guide, padding inside the colour LUT ------

def _brute_force(image, guide, sigma_color, sigma_spatial):
    """Independent float64 reference: Gaussian weights evaluated directly
    (no LUT), symmetric window, zero padding for image and guide."""
    image = np.asarray(image, np.float64)
    guide = np.asarray(guide, np.float64)
    ext = 3
    pi = np.pad(image, ext)
    pg = np.pad(guide, ext)
    H, W = image.shape
    out = np.zeros((H, W))
    for r in range(H):
        for c in range(W):
            win_v = pi[r:r + 2 * ext + 1, c:c + 2 * ext + 1]
            win_g = pg[r:r + 2 * ext + 1, c:c + 2 * ext + 1]
            yy, xx = np.mgrid[-ext:ext + 1, -ext:ext + 1]
            w = (np.exp(-0.5 * (yy**2 + xx**2) / sigma_spatial**2)
                 * np.exp(-0.5 * (guide[r, c] - win_g) ** 2 / sigma_color**2))
            out[r, c] = (w * win_v).sum() / w.sum()
    return out


def _impulse():
    image = np.zeros((15, 15), np.float32)
    image[7, 7] = 1.0
    return image


def test_constant_joint_guide_still_smooths_spatially():
    """A flat guide makes every in-image range weight equal; the result is
    the normalized spatial kernel, not the unfiltered image."""
    image = _impulse()
    out = bilateral_numpy(image, np.ones_like(image), sigma_color=0.1,
                          sigma_spatial=1.0)
    centre = 1.0 / spatial_weights(1.0).sum()
    assert out[7, 7] == pytest.approx(centre, abs=1e-6)          # ~0.159241
    assert out[7, 7] == pytest.approx(0.159241, abs=1e-6)


def test_constant_joint_guide_with_automatic_sigma_uses_in_image_neighbours():
    """sigma_color=None on a flat guide (std 0): range weights are 1 inside
    the image and 0 for the padding, so a flat image stays flat at borders."""
    image = _impulse()
    out = bilateral_numpy(image, np.ones_like(image), sigma_color=None,
                          sigma_spatial=1.0)
    assert out[7, 7] == pytest.approx(0.159241, abs=1e-6)
    flat = bilateral_numpy(np.full((15, 15), 0.4, np.float32),
                           np.ones((15, 15), np.float32),
                           sigma_color=None, sigma_spatial=1.0)
    np.testing.assert_allclose(flat, 0.4, atol=1e-6)


def test_classic_filter_on_a_constant_image_is_still_the_identity():
    image = np.full((9, 9), 0.3, np.float32)
    np.testing.assert_array_equal(
        bilateral_numpy(image, image, sigma_color=None, sigma_spatial=1.0), image)


def test_positive_narrow_guide_does_not_overweight_the_zero_padding():
    """Differences to the zero padding exceed the guide's own span; the LUT
    must cover them instead of clamping them to its last (largest) entry."""
    image = np.ones((15, 15), np.float32)
    guide = np.tile(np.linspace(0.5, 0.51, 15, dtype=np.float32), (15, 1))
    out = bilateral_numpy(image, guide, sigma_color=0.1, sigma_spatial=1.0)
    ref = _brute_force(image, guide, 0.1, 1.0)
    assert out[0, 0] == pytest.approx(0.999996, abs=1e-5)
    np.testing.assert_allclose(out, ref, atol=1e-4)


def test_symmetric_filter_matches_an_independent_reference():
    rng = np.random.default_rng(11)
    image = rng.random((12, 13), dtype=np.float32)
    out = bilateral_numpy(image, image, sigma_color=0.15, sigma_spatial=1.0)
    np.testing.assert_allclose(out, _brute_force(image, image, 0.15, 1.0), atol=2e-4)


def test_joint_total_on_complementary_channels_smooths_both():
    from karak.preprocessing.denoise import bilateral_denoise_cube

    image = _impulse()
    cube = np.stack([image, 1.0 - image], axis=-1)
    out = bilateral_denoise_cube(cube, np.ones((15, 15), bool), sigma_color=None,
                                 sigma_spatial=1.0, method="joint_bilateral_total")
    assert out[7, 7, 0] == pytest.approx(0.159241, abs=1e-5)
    assert out[7, 7, 1] == pytest.approx(1 - 0.159241, abs=1e-5)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA")
@pytest.mark.parametrize("case", ["constant_guide", "constant_guide_auto", "narrow_guide"])
def test_cupy_kernel_matches_the_reference_on_the_review_cases(case):
    import cupy as cp

    from karak.preprocessing.bilateral import bilateral_cupy

    if case == "narrow_guide":
        image = np.ones((15, 15), np.float32)
        guide = np.tile(np.linspace(0.5, 0.51, 15, dtype=np.float32), (15, 1))
        sigma = 0.1
    else:
        image, guide = _impulse(), np.ones((15, 15), np.float32)
        sigma = None if case == "constant_guide_auto" else 0.1
    ref = bilateral_numpy(image, guide, sigma_color=sigma, sigma_spatial=1.0)
    gpu = bilateral_cupy(cp.asarray(image), cp.asarray(guide), sigma_color=sigma,
                         sigma_spatial=1.0)
    np.testing.assert_allclose(cp.asnumpy(gpu), ref, atol=1e-6)
