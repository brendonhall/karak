"""End-to-end smoke test on a tiny synthetic dataset.

Runs the core pipeline stages (mask -> denoise -> normalize -> PCA ->
HDBSCAN -> kNN noise reassignment) on a 64x64, 3-channel synthetic image
containing two chemically distinct phases, without going through the CLI.
"""

from __future__ import annotations

import numpy as np
import pytest

from karak.clustering.hdbscan_cluster import run_hdbscan
from karak.clustering.noise_assign import assign_noise_pixels, labels_to_image
from karak.clustering.pca import fit_pca
from karak.config import DenoiseConfig, HDBSCANConfig, PCAConfig
from karak.io.masks import create_mineral_mask
from karak.preprocessing.compositional import zscore_normalize
from karak.preprocessing.denoise import denoise_cube


@pytest.fixture()
def synthetic_scene():
    """64x64x3 cube: background strip + two chemically distinct phases."""
    rng = np.random.default_rng(0)
    H = W = 64
    cube = np.zeros((H, W, 3), dtype=np.float32)

    # Phase A (columns 8..35): high channel 0
    cube[:, 8:36, 0] = 0.8
    cube[:, 8:36, 1] = 0.2
    cube[:, 8:36, 2] = 0.5
    # Phase B (columns 36..63): high channel 1
    cube[:, 36:, 0] = 0.2
    cube[:, 36:, 1] = 0.8
    cube[:, 36:, 2] = 0.4

    noise = rng.normal(0, 0.02, size=cube.shape).astype(np.float32)
    cube[:, 8:, :] = np.clip(cube[:, 8:, :] + noise[:, 8:, :], 0.01, 1.0)
    # Columns 0..7 stay exactly zero = background / epoxy
    return cube


def test_smoke_two_phase_recovery(synthetic_scene):
    cube = synthetic_scene
    H, W, C = cube.shape

    # Mask: background (all-zero) pixels excluded
    mask = create_mineral_mask(cube, valid_mask=None, min_object_size=10)
    assert mask.sum() == H * (W - 8)

    # Denoise raw [0,1] cube, then z-score normalize on mineral pixels
    denoised = denoise_cube(cube, mask, DenoiseConfig(method="bilateral"))
    normalized, means, stds = zscore_normalize(denoised, mask)
    assert normalized.shape == (H, W, C)
    assert means.shape == (C,)

    # PCA -> HDBSCAN -> kNN noise reassignment
    pca_model, features, mineral_indices = fit_pca(
        normalized, mask, PCAConfig(n_components=3, random_state=0)
    )
    assert features.shape == (mask.sum(), 3)

    labels, probabilities, clusterer = run_hdbscan(
        features, HDBSCANConfig(min_cluster_size=100, random_state=0)
    )
    n_clusters = len(set(labels[labels >= 0]))
    assert n_clusters == 2, f"expected 2 phases, found {n_clusters}"

    cleaned = assign_noise_pixels(features, labels, k=5)
    assert np.all(cleaned >= 0), "noise pixels remain after kNN reassignment"

    # Map back to image space and check spatial coherence
    label_img = labels_to_image(cleaned, mineral_indices, (H, W))
    assert label_img.shape == (H, W)
    assert np.all(label_img[:, :8] == -1)  # background stays unlabeled

    phase_a = label_img[:, 8:36].ravel()
    phase_b = label_img[:, 36:].ravel()
    # Each synthetic phase must map dominantly to a single, distinct cluster
    a_label = np.bincount(phase_a).argmax()
    b_label = np.bincount(phase_b).argmax()
    assert a_label != b_label
    assert (phase_a == a_label).mean() > 0.9
    assert (phase_b == b_label).mean() > 0.9
