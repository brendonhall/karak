"""Noise pixel reassignment via k-nearest-neighbor voting.

After HDBSCAN clustering, noise pixels (label -1) are reassigned to the
nearest cluster based on their PCA features. This produces a cleaned
label map alongside the raw HDBSCAN labels.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def assign_noise_pixels(
    pca_features: np.ndarray,
    labels: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Reassign noise pixels to nearest cluster via kNN.

    Parameters
    ----------
    pca_features : np.ndarray
        (N_mineral, n_components) PCA features for all mineral pixels.
    labels : np.ndarray
        (N_mineral,) int32 HDBSCAN labels (-1 = noise).
    k : int
        Number of neighbors for kNN voting.

    Returns
    -------
    np.ndarray
        (N_mineral,) int32 cleaned labels with no -1 values.
        Non-noise pixels retain their original labels.
    """
    noise_mask = labels == -1
    n_noise = int(np.sum(noise_mask))

    if n_noise == 0:
        logger.info("No noise pixels to reassign")
        return labels.copy()

    # Train kNN on non-noise pixels
    clean_mask = ~noise_mask
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(pca_features[clean_mask], labels[clean_mask])

    # Predict labels for noise pixels
    noise_labels = knn.predict(pca_features[noise_mask])

    cleaned = labels.copy()
    cleaned[noise_mask] = noise_labels.astype(np.int32)

    logger.info(
        "Reassigned %d noise pixels via %d-NN (%.1f%% of mineral pixels)",
        n_noise, k, 100.0 * n_noise / len(labels),
    )

    return cleaned


def labels_to_image(
    labels: np.ndarray,
    mineral_indices: np.ndarray,
    image_shape: tuple[int, int],
    background_value: int = -1,
) -> np.ndarray:
    """Map per-mineral-pixel labels back to a full image array.

    Parameters
    ----------
    labels : np.ndarray
        (N_mineral,) int32 cluster labels.
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 array of (row, col) coordinates.
    image_shape : tuple[int, int]
        (H, W) shape of the full image.
    background_value : int
        Value for non-mineral pixels (default -1).

    Returns
    -------
    np.ndarray
        (H, W) int32 label image. background_value for non-mineral pixels.
    """
    label_image = np.full(image_shape, background_value, dtype=np.int32)
    rows = mineral_indices[:, 0]
    cols = mineral_indices[:, 1]
    label_image[rows, cols] = labels
    return label_image
