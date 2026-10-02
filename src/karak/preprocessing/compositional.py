"""Compositional data handling: z-score normalization and validation.

v1.1: Z-score normalization replaces CLR/ILR transforms.  Jet-inverted
values are NOT true compositional data, so log-ratio transforms are
inappropriate.  Per-channel z-score normalization (zero mean, unit variance
on mineral pixels) is the correct approach.

scikit-bio is no longer required.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


ACCUMULATE = ("float64", "float32")


def zscore_normalize(
    cube: np.ndarray,
    mask: np.ndarray,
    *,
    accumulate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score normalization over mineral pixels.

    For each channel, compute the mean and standard deviation over mineral
    pixels only, then normalize: ``z = (x - mean) / std``.  Non-mineral
    pixels are set to 0.

    Works on numpy or CuPy arrays; the normalized cube comes back on the
    input's device, ``means`` and ``stds`` always as numpy ``(C,)`` float32.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) element cube (raw or denoised [0,1] values).
    mask : np.ndarray
        (H, W) boolean mask, True = mineral pixel.
    accumulate : str
        Precision of the mean and standard-deviation sums. ``"float64"``
        is accurate. ``"float32"`` reproduces the published NWA 4587
        baseline: numpy sums a float32 (N, C) array along axis 0 row by
        row in float32, and on 12.5 M pixels the standard deviations come
        out up to 3.6 % off. On CuPy the reduction is a tree either way.

    Returns
    -------
    normalized : np.ndarray
        (H, W, C) float32 z-score normalized cube.
    means : np.ndarray
        (C,) per-channel means on mineral pixels.
    stds : np.ndarray
        (C,) per-channel standard deviations on mineral pixels.
    """
    from karak.accel import to_numpy, xp as _xp

    xp = _xp(cube)
    H, W, C = cube.shape
    mineral_data = cube[mask]  # (N, C)

    if accumulate not in ACCUMULATE:
        raise ValueError(f"accumulate must be one of {ACCUMULATE}, got {accumulate!r}")
    # float32 output either way; only the sums run at the requested precision
    means = mineral_data.mean(axis=0, dtype=accumulate).astype(xp.float32)  # (C,)
    stds = mineral_data.std(axis=0, dtype=accumulate).astype(xp.float32)  # (C,)

    # Guard against zero-std channels (constant values)
    stds_safe = stds.copy()
    zero_std = stds_safe < 1e-10
    if bool(zero_std.any()):
        n_zero = int(zero_std.sum())
        logger.warning(
            "%d channels have near-zero std -- setting std to 1.0 to avoid division by zero",
            n_zero,
        )
        stds_safe[zero_std] = 1.0

    # Normalize
    normalized = xp.zeros((H, W, C), dtype=xp.float32)
    normalized[mask] = ((mineral_data - means) / stds_safe).astype(xp.float32)

    logger.info(
        "Z-score normalization: %d channels, %d mineral pixels, "
        "range [%.4f, %.4f]",
        C,
        int(mask.sum()),
        float(normalized[mask].min()),
        float(normalized[mask].max()),
    )
    return (
        normalized,
        to_numpy(means).astype(np.float32),
        to_numpy(stds).astype(np.float32),
    )


def validate_normalization(
    normalized_cube: np.ndarray,
    mask: np.ndarray,
    element_names: list[str],
) -> dict:
    """Validate z-score normalization: check per-channel mean ~0, std ~1.

    Parameters
    ----------
    normalized_cube : np.ndarray
        (H, W, C) z-score normalized cube.
    mask : np.ndarray
        (H, W) boolean mask.
    element_names : list[str]
        Ordered element names matching the C axis.

    Returns
    -------
    report : dict
        ``{element: {"mean": float, "std": float, "min": float, "max": float,
        "ok": bool}}``.  "ok" is True if |mean| < 0.01 and |std - 1| < 0.01.
    """
    mineral_data = normalized_cube[mask]  # (N, C)

    report: dict[str, dict] = {}
    all_ok = True

    for i, name in enumerate(element_names):
        ch = mineral_data[:, i]
        ch_mean = float(ch.mean())
        ch_std = float(ch.std())
        ch_min = float(ch.min())
        ch_max = float(ch.max())
        ok = abs(ch_mean) < 0.01 and abs(ch_std - 1.0) < 0.01

        report[name] = {
            "mean": ch_mean,
            "std": ch_std,
            "min": ch_min,
            "max": ch_max,
            "ok": ok,
        }

        if not ok:
            all_ok = False
            logger.warning(
                "Channel '%s' normalization off: mean=%.6f, std=%.6f",
                name,
                ch_mean,
                ch_std,
            )

    if all_ok:
        logger.info(
            "Normalization validation passed: all %d channels have "
            "mean ~0 and std ~1",
            len(element_names),
        )
    else:
        n_bad = sum(1 for v in report.values() if not v["ok"])
        logger.warning(
            "Normalization validation: %d/%d channels out of tolerance",
            n_bad,
            len(element_names),
        )

    return report
