"""PCA dimensionality reduction for mineral phase clustering.

Fits PCA on mineral pixels from the z-score normalized cube, then
projects all mineral pixels into the reduced space.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.decomposition import PCA

if TYPE_CHECKING:
    from karak.config import PCAConfig

logger = logging.getLogger(__name__)


def fit_pca(
    normalized_cube: np.ndarray,
    mineral_mask: np.ndarray,
    config: PCAConfig,
) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Fit PCA on mineral pixels and project them into reduced space.

    Parameters
    ----------
    normalized_cube : np.ndarray
        (H, W, C) float32 z-score normalized element cube.
    mineral_mask : np.ndarray
        (H, W) boolean mask (True = mineral pixel).
    config : PCAConfig
        PCA configuration (n_components, subsample_fraction, random_state).

    Returns
    -------
    pca_model : sklearn.decomposition.PCA
        Fitted PCA model (access .explained_variance_ratio_ for scree plot).
    pca_features : np.ndarray
        (N_mineral, n_components) float32 PCA-transformed mineral pixels.
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 array of (row, col) coordinates for each
        mineral pixel, for mapping results back to image space.
    """
    H, W, C = normalized_cube.shape

    # Extract mineral pixel spectra
    mineral_spectra = normalized_cube[mineral_mask]  # (N_mineral, C)
    n_mineral = mineral_spectra.shape[0]
    logger.info("Extracted %d mineral pixels with %d channels", n_mineral, C)

    # Store pixel coordinates for spatial reconstruction
    rows, cols = np.where(mineral_mask)
    mineral_indices = np.stack([rows, cols], axis=1).astype(np.int32)

    # Determine n_components
    n_components = config.n_components if config.n_components is not None else C

    # Optionally subsample for fitting (project all pixels after)
    rng = np.random.default_rng(config.random_state)
    if config.subsample_fraction is not None and config.subsample_fraction < 1.0:
        n_fit = int(n_mineral * config.subsample_fraction)
        fit_idx = rng.choice(n_mineral, size=n_fit, replace=False)
        fit_spectra = mineral_spectra[fit_idx]
        logger.info("Subsampled %d/%d pixels for PCA fitting", n_fit, n_mineral)
    else:
        fit_spectra = mineral_spectra

    # Fit PCA
    pca_model = PCA(n_components=n_components, random_state=config.random_state)
    pca_model.fit(fit_spectra)

    cumvar = np.cumsum(pca_model.explained_variance_ratio_)
    logger.info(
        "PCA fitted: %d components, cumulative variance: %.1f%%",
        n_components,
        cumvar[-1] * 100,
    )

    # Transform ALL mineral pixels
    pca_features = pca_model.transform(mineral_spectra).astype(np.float32)

    return pca_model, pca_features, mineral_indices


def select_components(
    pca_features: np.ndarray,
    n_keep: int,
) -> np.ndarray:
    """Truncate PCA features to the first n_keep components.

    Parameters
    ----------
    pca_features : np.ndarray
        (N_mineral, n_components) full PCA features.
    n_keep : int
        Number of components to retain.

    Returns
    -------
    np.ndarray
        (N_mineral, n_keep) truncated PCA features.
    """
    if n_keep > pca_features.shape[1]:
        raise ValueError(
            f"n_keep={n_keep} exceeds available components {pca_features.shape[1]}"
        )
    return pca_features[:, :n_keep]


def auto_n_components(
    explained_variance_ratio: np.ndarray,
    variance_threshold: float = 0.95,
    min_components: int = 5,
) -> int:
    """Pick the component count reaching the cumulative-variance threshold.

    Returns the index (1-based) of the first component whose cumulative
    explained variance reaches ``variance_threshold``, floored at
    ``min_components``. If the threshold is never reached, all components
    are kept.
    """
    cumvar = np.cumsum(explained_variance_ratio)
    candidates = np.where(cumvar >= variance_threshold)[0]
    n_keep = (
        int(candidates[0] + 1)
        if len(candidates) > 0
        else len(explained_variance_ratio)
    )
    return max(n_keep, min_components)
