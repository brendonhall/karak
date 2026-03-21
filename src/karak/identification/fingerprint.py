"""Per-cluster chemical fingerprint computation and similarity flagging.

Functions in this module operate on the raw denoised [0, 1] element cube
and cleaned cluster labels produced by the Phase 2 clustering pipeline.
They compute per-phase mean/std chemical signatures and flag potentially
over-split clusters based on cosine similarity.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


def compute_fingerprints(
    denoised_cube: np.ndarray,
    cleaned_labels: np.ndarray,
    mineral_indices: np.ndarray,
    element_names: list[str],
) -> dict:
    """Compute per-cluster chemical fingerprints from the denoised element cube.

    For each cluster, extracts the pixel spectra from ``denoised_cube`` using
    ``mineral_indices`` and computes the mean and standard deviation across
    pixels for each element channel.  Elements are ranked by the variance of
    their cluster means (most discriminating first).

    Parameters
    ----------
    denoised_cube : np.ndarray
        (H, W, C) float32 raw denoised element cube with values in [0, 1].
    cleaned_labels : np.ndarray
        (N_mineral,) int32 cleaned cluster labels (0 .. K-1, no noise).
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 (row, col) coordinates of mineral pixels.
    element_names : list[str]
        Ordered element names matching the C channels.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"fingerprints"`` : dict mapping cluster label (int) to a dict with
          ``"mean"`` (C,), ``"std"`` (C,), ``"n_pixels"`` (int), and
          ``"area_pct"`` (float).
        - ``"element_names"`` : list[str] -- original element order.
        - ``"element_order"`` : np.ndarray (C,) -- indices sorted by
          cross-cluster variance of means (descending, most discriminating
          first).
        - ``"n_clusters"`` : int
        - ``"n_mineral_pixels"`` : int
    """
    rows = mineral_indices[:, 0]
    cols = mineral_indices[:, 1]
    pixel_spectra = denoised_cube[rows, cols, :]  # (N_mineral, C)

    unique_labels = np.unique(cleaned_labels)
    n_clusters = len(unique_labels)
    n_mineral = len(cleaned_labels)

    logger.info(
        "Computing fingerprints for %d clusters from %d mineral pixels",
        n_clusters,
        n_mineral,
    )

    fingerprints: dict[int, dict] = {}
    mean_matrix = np.empty((n_clusters, pixel_spectra.shape[1]), dtype=np.float64)

    for i, label in enumerate(unique_labels):
        mask = cleaned_labels == label
        cluster_pixels = pixel_spectra[mask]
        n_pixels = int(cluster_pixels.shape[0])
        mean_vec = np.mean(cluster_pixels, axis=0)
        std_vec = np.std(cluster_pixels, axis=0)

        fingerprints[int(label)] = {
            "mean": mean_vec,
            "std": std_vec,
            "n_pixels": n_pixels,
            "area_pct": n_pixels / n_mineral * 100.0,
        }
        mean_matrix[i] = mean_vec

    # Element order: sort by variance of means across clusters (descending)
    cross_cluster_var = np.var(mean_matrix, axis=0)
    element_order = np.argsort(cross_cluster_var)[::-1]

    logger.info(
        "Top discriminating elements: %s",
        [element_names[idx] for idx in element_order[:5]],
    )

    return {
        "fingerprints": fingerprints,
        "element_names": element_names,
        "element_order": element_order,
        "n_clusters": n_clusters,
        "n_mineral_pixels": n_mineral,
    }


def flag_similar_clusters(
    fingerprints_dict: dict,
    threshold: float = 0.95,
) -> list[tuple[int, int, float]]:
    """Flag pairs of clusters with highly similar chemical fingerprints.

    Computes pairwise cosine similarity between cluster mean vectors and
    returns pairs exceeding the threshold.  This is a visual-review aid;
    the researcher decides whether flagged clusters should be merged.

    Parameters
    ----------
    fingerprints_dict : dict
        Output from :func:`compute_fingerprints`.
    threshold : float, optional
        Cosine similarity threshold above which clusters are flagged
        (default 0.95).

    Returns
    -------
    list[tuple[int, int, float]]
        List of ``(cluster_a, cluster_b, similarity)`` tuples where
        ``similarity > threshold``, sorted by similarity descending.
    """
    fp = fingerprints_dict["fingerprints"]
    labels = sorted(fp.keys())
    similar_pairs: list[tuple[int, int, float]] = []

    for a, b in combinations(labels, 2):
        mean_a = fp[a]["mean"]
        mean_b = fp[b]["mean"]

        # scipy.spatial.distance.cosine returns distance = 1 - similarity
        similarity = 1.0 - cosine(mean_a, mean_b)

        if similarity > threshold:
            similar_pairs.append((a, b, float(similarity)))

    # Sort by similarity descending
    similar_pairs.sort(key=lambda x: x[2], reverse=True)

    if similar_pairs:
        logger.warning(
            "Found %d cluster pairs with cosine similarity > %.2f",
            len(similar_pairs),
            threshold,
        )
        for a, b, sim in similar_pairs:
            logger.warning("  Clusters %d and %d: similarity = %.4f", a, b, sim)
    else:
        logger.info(
            "No cluster pairs exceed cosine similarity threshold of %.2f",
            threshold,
        )

    return similar_pairs
