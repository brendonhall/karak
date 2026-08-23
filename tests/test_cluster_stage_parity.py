"""Parity tests for the clustering stages against the core functions.

A module-scoped chain runs the preprocessing once on the synthetic scene;
each stage test compares its output with a direct core-function call using
identical, seeded parameters.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from conftest import make_synthetic_scene
from karak.config import (
    ClusterConfig,
    DenoiseConfig,
    GMMSplitConfig,
    HDBSCANConfig,
    OlivineExtractionConfig,
    PCAConfig,
    RefinementConfig,
    TiledConfig,
)
from karak.clustering.hdbscan_cluster import compute_cluster_stats, run_hdbscan
from karak.clustering.noise_assign import assign_noise_pixels
from karak.clustering.pca import fit_pca, select_components
from karak.clustering.refinement import refine_phases
from karak.clustering.tiling import final_knn_assign, run_tiled_hdbscan
from karak.identification.fingerprint import (
    compute_fingerprints,
    flag_similar_clusters,
)
from karak.io.masks import create_mineral_mask
from karak.preprocessing.compositional import zscore_normalize
from karak.preprocessing.denoise import denoise_cube
from karak.stages import get
from karak.stages.payloads import (
    BseImage,
    ElementCube,
    LabelState,
    Labels,
    MaskSet,
    PCAFeatures,
    Space,
    TiledArtifacts,
)


@pytest.fixture(scope="module")
def chain():
    """Preprocessed synthetic scene + core clustering results."""
    cube = make_synthetic_scene()
    H, W, _ = cube.shape
    mask = create_mineral_mask(cube, None, min_object_size=10)
    denoised = denoise_cube(cube, mask, DenoiseConfig(method="bilateral"))
    normalized, means, stds = zscore_normalize(denoised, mask)
    _, features_full, mineral_indices = fit_pca(
        normalized, mask, PCAConfig(n_components=3, random_state=0)
    )
    features = select_components(features_full, 3)
    labels, probabilities, _ = run_hdbscan(
        features, HDBSCANConfig(min_cluster_size=100, random_state=0)
    )
    return {
        "cube": cube,
        "shape": (H, W),
        "mask": mask,
        "denoised": denoised,
        "normalized": normalized,
        "means": means,
        "stds": stds,
        "features": features,
        "mineral_indices": mineral_indices,
        "labels": labels,
        "probabilities": probabilities,
    }


@pytest.fixture(scope="module")
def normalized_cube(chain):
    return ElementCube(
        pixels=chain["normalized"],
        element_names=("A", "B", "C"),
        space=Space.NORMALIZED,
        means=chain["means"],
        stds=chain["stds"],
    )


@pytest.fixture(scope="module")
def denoised_cube(chain):
    return ElementCube(
        pixels=chain["denoised"],
        element_names=("A", "B", "C"),
        space=Space.DENOISED,
    )


@pytest.fixture(scope="module")
def mask_set(chain):
    return MaskSet(mineral_mask=chain["mask"], valid_mask=None, stats={})


@pytest.fixture(scope="module")
def features_payload(chain):
    return PCAFeatures(
        features=chain["features"],
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"],
        explained_variance_ratio=np.array([0.7, 0.2, 0.1]),
        n_kept=3,
    )


@pytest.fixture(scope="module")
def raw_labels_payload(chain):
    return Labels(
        labels=chain["labels"],
        probabilities=chain["probabilities"],
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"],
        state=LabelState.RAW,
    )


# ---------------------------------------------------------------------------
# pca
# ---------------------------------------------------------------------------

def test_pca_parity(normalized_cube, mask_set, chain):
    out = get("pca")().run(
        {"cube": normalized_cube, "masks": mask_set},
        {"n_components": 3, "random_state": 0},
    )
    features = out["features"]
    assert isinstance(features, PCAFeatures)
    assert features.n_kept == 3
    np.testing.assert_array_equal(features.features, chain["features"])
    np.testing.assert_array_equal(
        features.mineral_indices, chain["mineral_indices"]
    )
    assert features.image_shape == chain["shape"]


def test_pca_auto_components(normalized_cube, mask_set):
    # n_components=0 = auto; 3-channel input caps the floor of 5 at 3
    out = get("pca")().run(
        {"cube": normalized_cube, "masks": mask_set},
        {"n_components": 0, "random_state": 0},
    )
    assert out["features"].n_kept == 3


# ---------------------------------------------------------------------------
# hdbscan_global
# ---------------------------------------------------------------------------

def test_hdbscan_global_parity(features_payload, chain):
    out = get("hdbscan_global")().run(
        {"features": features_payload},
        {"min_cluster_size": 100, "random_state": 0},
    )
    labels = out["labels"]
    assert isinstance(labels, Labels)
    assert labels.state is LabelState.RAW
    np.testing.assert_array_equal(labels.labels, chain["labels"])
    np.testing.assert_array_equal(labels.probabilities, chain["probabilities"])


# ---------------------------------------------------------------------------
# hdbscan_tiled
# ---------------------------------------------------------------------------

def _tiled_config() -> ClusterConfig:
    return ClusterConfig(
        strategy="tiled",
        hdbscan=HDBSCANConfig(min_cluster_size=100, random_state=0),
        tiled=TiledConfig(tile_size=32, min_clusters_per_tile=1),
    )


def test_hdbscan_tiled_parity(features_payload, denoised_cube, chain):
    expected_raw, _, expected_probs, expected_tiles, expected_registry = (
        run_tiled_hdbscan(
            chain["features"], chain["mineral_indices"], chain["shape"],
            chain["denoised"], _tiled_config(), skip_knn=True,
        )
    )

    out = get("hdbscan_tiled")().run(
        {"features": features_payload, "cube": denoised_cube},
        {"min_cluster_size": 100, "random_state": 0,
         "tile_size": 32, "min_clusters_per_tile": 1},
    )

    labels, tiles = out["labels"], out["tiles"]
    assert labels.state is LabelState.RAW
    np.testing.assert_array_equal(labels.labels, expected_raw)
    np.testing.assert_array_equal(labels.probabilities, expected_probs)
    assert isinstance(tiles, TiledArtifacts)
    assert len(tiles.tile_results) == len(expected_tiles)
    assert len(tiles.phase_registry) == len(expected_registry)
    for got, exp in zip(tiles.phase_registry, expected_registry):
        assert got.global_id == exp.global_id
        np.testing.assert_allclose(got.mean_fingerprint, exp.mean_fingerprint)


# ---------------------------------------------------------------------------
# rare_phase
# ---------------------------------------------------------------------------

def test_rare_phase_parity(features_payload, denoised_cube, chain):
    from karak.clustering.tiling import recluster_unassigned

    raw, _, probs, tile_results, registry = run_tiled_hdbscan(
        chain["features"], chain["mineral_indices"], chain["shape"],
        chain["denoised"], _tiled_config(), skip_knn=True,
    )
    tiles_payload = TiledArtifacts(
        tile_results=tuple(tile_results),
        phase_registry=tuple(copy.deepcopy(registry)),
        tile_size=32,
    )
    labels_payload = Labels(
        labels=raw, probabilities=probs,
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"], state=LabelState.RAW,
    )

    config = _tiled_config().model_copy(deep=True)
    config.rare_phase.enabled = True
    config.rare_phase.min_cluster_size = 20
    expected_labels, expected_registry, _, _ = recluster_unassigned(
        chain["features"], raw.copy(), chain["denoised"],
        chain["mineral_indices"], copy.deepcopy(registry), config,
    )

    registry_before = copy.deepcopy(tiles_payload.phase_registry)
    out = get("rare_phase")().run(
        {"labels": labels_payload, "features": features_payload,
         "cube": denoised_cube, "tiles": tiles_payload},
        {"min_cluster_size": 20, "random_state": 0},
    )

    np.testing.assert_array_equal(out["labels"].labels, expected_labels)
    assert out["labels"].state is LabelState.RAW
    assert len(out["tiles"].phase_registry) == len(expected_registry)
    # input payload registry must not be mutated
    assert len(tiles_payload.phase_registry) == len(registry_before)


# ---------------------------------------------------------------------------
# noise_assign (+ parity between the two kNN implementations)
# ---------------------------------------------------------------------------

def test_noise_assign_parity(features_payload, raw_labels_payload, chain):
    expected = assign_noise_pixels(chain["features"], chain["labels"], k=5)

    out = get("noise_assign")().run(
        {"labels": raw_labels_payload, "features": features_payload}, {"k": 5}
    )

    labels = out["labels"]
    assert labels.state is LabelState.CLEANED
    np.testing.assert_array_equal(labels.labels, expected)
    assert np.all(labels.labels >= 0)
    # probabilities travel through unchanged
    np.testing.assert_array_equal(labels.probabilities, chain["probabilities"])


def test_knn_implementations_agree(chain):
    """assign_noise_pixels must reproduce final_knn_assign so one stage
    serves both the global and tiled strategies."""
    ours = assign_noise_pixels(chain["features"], chain["labels"], k=5)
    tiled = final_knn_assign(chain["features"], chain["labels"], 5)
    np.testing.assert_array_equal(ours, tiled)


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------

def test_refine_parity(denoised_cube, chain):
    cleaned = assign_noise_pixels(chain["features"], chain["labels"], k=5)
    target = int(np.bincount(cleaned).argmax())
    config = RefinementConfig(
        enabled=True,
        target_phase=target,
        olivine=OlivineExtractionConfig(enabled=False),
        gmm_split=GMMSplitConfig(
            enabled=True, n_components=2, features=["A", "B"],
            subsample_n=None, random_state=0,
        ),
    )
    bse = np.zeros(chain["shape"], dtype=np.float32)
    expected = refine_phases(
        cleaned.copy(), chain["denoised"], bse,
        chain["mineral_indices"], ["A", "B", "C"], config,
    )

    cleaned_payload = Labels(
        labels=cleaned, probabilities=chain["probabilities"],
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"], state=LabelState.CLEANED,
    )
    out = get("refine")().run(
        {"labels": cleaned_payload, "cube": denoised_cube,
         "bse": BseImage(pixels=bse)},
        {"target_phase": target, "gmm_enabled": True,
         "gmm_features": "A,B", "gmm_subsample_n": 0, "random_state": 0},
    )

    assert out["labels"].state is LabelState.CLEANED
    np.testing.assert_array_equal(out["labels"].labels, expected)


# ---------------------------------------------------------------------------
# cluster_stats + fingerprints
# ---------------------------------------------------------------------------

def test_cluster_stats_parity(features_payload, raw_labels_payload, chain):
    cleaned = assign_noise_pixels(chain["features"], chain["labels"], k=5)
    expected = compute_cluster_stats(cleaned, chain["probabilities"])

    cleaned_payload = Labels(
        labels=cleaned, probabilities=chain["probabilities"],
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"], state=LabelState.CLEANED,
    )
    out = get("cluster_stats")().run({"labels": cleaned_payload})
    assert out["stats"].stats == expected


def test_fingerprints_parity(denoised_cube, chain):
    cleaned = assign_noise_pixels(chain["features"], chain["labels"], k=5)
    expected = compute_fingerprints(
        chain["denoised"], cleaned, chain["mineral_indices"], ["A", "B", "C"]
    )
    expected_pairs = flag_similar_clusters(expected, threshold=0.95)

    cleaned_payload = Labels(
        labels=cleaned, probabilities=chain["probabilities"],
        mineral_indices=chain["mineral_indices"],
        image_shape=chain["shape"], state=LabelState.CLEANED,
    )
    out = get("fingerprints")().run(
        {"labels": cleaned_payload, "cube": denoised_cube},
        {"similarity_threshold": 0.95},
    )

    fp = out["fingerprints"]
    assert set(fp.data["fingerprints"]) == set(expected["fingerprints"])
    for label, entry in expected["fingerprints"].items():
        np.testing.assert_allclose(
            fp.data["fingerprints"][label]["mean"], entry["mean"]
        )
    assert fp.similar_pairs == expected_pairs
