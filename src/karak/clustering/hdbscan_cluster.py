"""HDBSCAN clustering on PCA-reduced mineral pixel features.

Clusters mineral pixels to identify major mineral phases. Supports
subsampling for large images with approximate_predict for remaining pixels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import hdbscan
import numpy as np

if TYPE_CHECKING:
    from karak.config import HDBSCANConfig

logger = logging.getLogger(__name__)


def run_hdbscan(
    pca_features: np.ndarray,
    config: HDBSCANConfig,
) -> tuple[np.ndarray, np.ndarray, hdbscan.HDBSCAN]:
    """Run HDBSCAN on PCA-reduced mineral pixel features.

    If config.subsample_n is set and fewer than the total number of pixels,
    HDBSCAN is fitted on a random subsample and the remaining pixels are
    assigned via approximate_predict.

    Parameters
    ----------
    pca_features : np.ndarray
        (N_mineral, n_components) PCA-transformed mineral pixel features.
    config : HDBSCANConfig
        HDBSCAN configuration parameters.

    Returns
    -------
    labels : np.ndarray
        (N_mineral,) int32 cluster labels. -1 = noise/unclassified.
    probabilities : np.ndarray
        (N_mineral,) float32 membership probabilities [0, 1].
    clusterer : hdbscan.HDBSCAN
        Fitted HDBSCAN model (for approximate_predict if needed later).
    """
    n_mineral = pca_features.shape[0]
    min_samples = config.min_samples if config.min_samples is not None else config.min_cluster_size

    rng = np.random.default_rng(config.random_state)

    if config.subsample_n is not None and config.subsample_n < n_mineral:
        # Subsample fitting
        n_fit = config.subsample_n
        fit_idx = rng.choice(n_mineral, size=n_fit, replace=False)
        fit_features = pca_features[fit_idx]

        logger.info(
            "Fitting HDBSCAN on %d/%d subsampled pixels "
            "(min_cluster_size=%d, min_samples=%d)",
            n_fit, n_mineral, config.min_cluster_size, min_samples,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        clusterer.fit(fit_features)

        # Assign all pixels (including fitted ones) via approximate_predict
        # for consistency
        labels_all, probs_all = hdbscan.approximate_predict(clusterer, pca_features)
        labels = labels_all.astype(np.int32)
        probabilities = probs_all.astype(np.float32)

    else:
        # Fit on all mineral pixels
        logger.info(
            "Fitting HDBSCAN on all %d mineral pixels "
            "(min_cluster_size=%d, min_samples=%d)",
            n_mineral, config.min_cluster_size, min_samples,
        )

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=config.min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,
        )
        clusterer.fit(pca_features)

        labels = clusterer.labels_.astype(np.int32)
        probabilities = clusterer.probabilities_.astype(np.float32)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = 100.0 * n_noise / n_mineral

    logger.info(
        "HDBSCAN result: %d clusters, %d noise pixels (%.1f%%)",
        n_clusters, n_noise, noise_pct,
    )

    return labels, probabilities, clusterer


def compute_cluster_stats(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Compute summary statistics for HDBSCAN clustering results.

    Parameters
    ----------
    labels : np.ndarray
        (N_mineral,) int32 cluster labels (-1 = noise).
    probabilities : np.ndarray
        (N_mineral,) float32 membership probabilities.

    Returns
    -------
    dict
        Summary statistics including n_clusters, n_noise, noise_pct,
        per-cluster pixel counts, and mean probabilities.
    """
    n_total = len(labels)
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = 100.0 * n_noise / n_total if n_total > 0 else 0.0

    cluster_info = {}
    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        cluster_info[int(lbl)] = {
            "n_pixels": int(np.sum(mask)),
            "pct": 100.0 * np.sum(mask) / n_total,
            "mean_prob": float(np.mean(probabilities[mask])),
        }

    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": noise_pct,
        "n_total": n_total,
        "clusters": cluster_info,
    }
