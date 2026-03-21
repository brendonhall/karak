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


def zscore_normalize(
    cube: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score normalization over mineral pixels.

    For each channel, compute the mean and standard deviation over mineral
    pixels only, then normalize: ``z = (x - mean) / std``.  Non-mineral
    pixels are set to 0.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) element cube (raw or denoised [0,1] values).
    mask : np.ndarray
        (H, W) boolean mask, True = mineral pixel.

    Returns
    -------
    normalized : np.ndarray
        (H, W, C) float32 z-score normalized cube.
    means : np.ndarray
        (C,) per-channel means on mineral pixels.
    stds : np.ndarray
        (C,) per-channel standard deviations on mineral pixels.
    """
    H, W, C = cube.shape
    mineral_data = cube[mask]  # (N, C)

    means = mineral_data.mean(axis=0)  # (C,)
    stds = mineral_data.std(axis=0)  # (C,)

    # Guard against zero-std channels (constant values)
    stds_safe = stds.copy()
    zero_std = stds_safe < 1e-10
    if zero_std.any():
        n_zero = zero_std.sum()
        logger.warning(
            "%d channels have near-zero std -- setting std to 1.0 to avoid division by zero",
            n_zero,
        )
        stds_safe[zero_std] = 1.0

    # Normalize
    normalized = np.zeros((H, W, C), dtype=np.float32)
    normalized[mask] = ((mineral_data - means) / stds_safe).astype(np.float32)

    logger.info(
        "Z-score normalization: %d channels, %d mineral pixels, "
        "range [%.4f, %.4f]",
        C,
        mask.sum(),
        normalized[mask].min(),
        normalized[mask].max(),
    )
    return normalized, means, stds


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
