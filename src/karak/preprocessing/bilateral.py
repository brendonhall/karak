"""Bilateral filter core: a numpy reference and a CuPy kernel.

Both take a single (H, W) float32 channel and a guide image of the same
shape. With ``guide is image`` this is the classic bilateral filter; with
another guide it is the joint (cross) bilateral filter, whose range weight
comes from the guide. The output pixel is the weighted mean of a square
window: weight = spatial Gaussian of the offset times a Gaussian of the
guide difference, the latter read from a lookup table of ``bins`` entries,
as scikit-image does. Pixels outside the image count as 0 in the image and
the guide (scikit-image's ``mode='constant'``); for the symmetric and joint
filters the table spans the guide's range together with that 0, so border
weights are exact. A constant guide with ``sigma_color=None`` leaves only
the spatial kernel over in-image neighbours.

``exact_skimage=True`` reproduces ``skimage.restoration.denoise_bilateral``
(0.19 through 0.26) to float32 precision, including its spatial table: it
builds an (n+1) x (n+1) table for an n x n window and reads it with stride
n, so the unit weight sits at offset (+2, -2) for the 7 x 7 window instead
of at the centre. karak keeps that behaviour under ``method="bilateral"``
because the published baseline used it; ``bilateral_sym`` and the joint
methods use the symmetric table.
"""

from __future__ import annotations

from math import ceil

import numpy as np

BINS = 10000


def window_size(sigma_spatial: float) -> int:
    """scikit-image's default window: ``max(5, 2 * ceil(3 * sigma) + 1)``."""
    return max(5, 2 * int(ceil(3 * sigma_spatial)) + 1)


def spatial_weights(sigma_spatial: float, exact_skimage: bool = False) -> np.ndarray:
    """(win, win) float32 spatial weights for offsets -ext..ext.

    ``exact_skimage`` returns the weights scikit-image effectively applies
    (see the module docstring), not the symmetric Gaussian.
    """
    win = window_size(sigma_spatial)
    ext = (win - 1) // 2
    if exact_skimage:
        # scikit-image: np.arange(-win // 2, win // 2 + 1) has win + 1
        # points, and the (win+1)^2 table is indexed as [kr * win + kc].
        grid = np.arange(-win // 2, win // 2 + 1)
        rr, cc = np.meshgrid(grid, grid, indexing="ij")
        table = np.exp(-0.5 * (np.hypot(rr, cc) ** 2 / sigma_spatial**2),
                       dtype=np.float32).ravel()
        idx = np.arange(win)[:, None] * win + np.arange(win)[None, :]
        return table[idx]
    grid = np.arange(-ext, ext + 1)
    rr, cc = np.meshgrid(grid, grid, indexing="ij")
    return np.exp(-0.5 * (np.hypot(rr, cc) ** 2 / sigma_spatial**2),
                  dtype=np.float32)


def color_lut(sigma_color: float, max_value: float, bins: int = BINS) -> np.ndarray:
    """Gaussian of the guide difference, tabulated over [0, max_value)."""
    values = np.linspace(0, max_value, bins, endpoint=False)
    return np.exp(-0.5 * (values**2 / sigma_color**2), dtype=np.float32)


def _prepare(image, guide, sigma_color, sigma_spatial, exact_skimage, xp):
    """Shared setup: (image, guide, spatial table, colour table, dist_scale,
    inside_only), or None when the classic filter sees a constant image
    (the identity, as in scikit-image).

    Out-of-image pixels count as 0 in both the image and the guide, so for
    the symmetric and joint filters the colour table spans every difference
    a window can see, including those to the zero padding. ``inside_only``
    is set when the range term carries no information (a constant guide
    with ``sigma_color=None``, whose std is 0): every in-image neighbour then
    gets range weight 1 and the padding gets 0, so the result is the
    normalized spatial kernel over the image.
    """
    image = xp.ascontiguousarray(image, dtype=xp.float32)
    guide = image if guide is image else xp.ascontiguousarray(guide, dtype=xp.float32)
    gmin, gmax = float(guide.min()), float(guide.max())
    if gmin == gmax and (guide is image or exact_skimage):
        return None
    inside_only = False
    if exact_skimage:
        # scikit-image shifts a negative image up and does not shift the
        # result back; the LUT then spans [0, max - min).
        if gmin < 0:
            image = image - xp.float32(gmin)
            guide = image if guide is image else guide - xp.float32(gmin)
            gmax -= gmin
        span = gmax
    else:
        span = max(gmax, 0.0) - min(gmin, 0.0)   # include the zero padding
    if sigma_color is None:
        sigma_color = float(guide.std())
    if sigma_color == 0.0:
        inside_only = True
        sigma_color = 1.0                     # unused: range weights are 0/1
    if span == 0.0:
        span = 1.0                            # all differences are 0
    rlut = spatial_weights(sigma_spatial, exact_skimage)
    clut = color_lut(sigma_color, span)
    dist_scale = np.float32(BINS) / np.float32(span)
    return image, guide, rlut, clut, dist_scale, inside_only


def bilateral_numpy(image, guide, *, sigma_color, sigma_spatial,
                    exact_skimage: bool = False) -> np.ndarray:
    """Reference bilateral / joint bilateral filter, vectorised over the
    window offsets, float32 throughout."""
    prep = _prepare(image, guide, sigma_color, sigma_spatial, exact_skimage, np)
    if prep is None:
        return np.array(image, dtype=np.float32, copy=True)
    image, guide, rlut, clut, dist_scale, inside_only = prep
    win = rlut.shape[0]
    ext = (win - 1) // 2
    H, W = image.shape
    pimg = np.pad(image, ext)
    pguide = pimg if guide is image else np.pad(guide, ext)
    pinside = np.pad(np.ones((H, W), np.float32), ext) if inside_only else None
    total = np.zeros((H, W), np.float32)
    weight = np.zeros((H, W), np.float32)
    last = np.int64(BINS - 1)
    for kr in range(win):
        for kc in range(win):
            values = pimg[kr:kr + H, kc:kc + W]
            if inside_only:
                w = rlut[kr, kc] * pinside[kr:kr + H, kc:kc + W]
            else:
                gvalues = pguide[kr:kr + H, kc:kc + W]
                dist = np.abs(guide - gvalues)
                bins = np.minimum((dist * dist_scale).astype(np.int64), last)
                w = rlut[kr, kc] * clut[bins]
            total += values * w
            weight += w
    return total / weight


_KERNEL_SOURCE = r"""
const int r = i / W;
const int c = i % W;
const float centre = guide[i];
float total = 0.0f;
float weight = 0.0f;
for (int kr = 0; kr < win; ++kr) {
    const int rr = r + kr - ext;
    for (int kc = 0; kc < win; ++kc) {
        const int cc = c + kc - ext;
        float value = 0.0f;
        float gvalue = 0.0f;
        const bool inside = rr >= 0 && rr < H && cc >= 0 && cc < W;
        if (inside) {
            value = image[rr * W + cc];
            gvalue = guide[rr * W + cc];
        } else if (inside_only) {
            continue;
        }
        float range_w = 1.0f;
        if (!inside_only) {
            const float dist = fabsf(centre - gvalue);
            int bin = (int)(dist * dist_scale);
            if (bin > last) bin = last;
            range_w = clut[bin];
        }
        const float w = rlut[kr * win + kc] * range_w;
        total += value * w;
        weight += w;
    }
}
out = total / weight;
"""


def bilateral_cupy(image, guide, *, sigma_color, sigma_spatial,
                   exact_skimage: bool = False):
    """The same filter as ``bilateral_numpy`` on CuPy arrays, one thread
    per output pixel."""
    import cupy as cp

    prep = _prepare(image, guide, sigma_color, sigma_spatial, exact_skimage, cp)
    if prep is None:
        return cp.array(image, dtype=cp.float32, copy=True)
    image, guide, rlut, clut, dist_scale, inside_only = prep
    win = rlut.shape[0]
    H, W = image.shape
    kernel = cp.ElementwiseKernel(
        "raw float32 image, raw float32 guide, raw float32 rlut, "
        "raw float32 clut, int32 H, int32 W, int32 win, int32 ext, "
        "float32 dist_scale, int32 last, bool inside_only",
        "float32 out",
        _KERNEL_SOURCE,
        "karak_bilateral",
    )
    out = cp.empty((H, W), cp.float32)
    kernel(image, guide, cp.asarray(rlut.ravel()), cp.asarray(clut),
           np.int32(H), np.int32(W), np.int32(win), np.int32((win - 1) // 2),
           np.float32(dist_scale), np.int32(BINS - 1), bool(inside_only), out)
    return out
