"""Tests for payload dataclasses and their HDF5 round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from karak.stages.payloads import (
    BseImage,
    ClusterStats,
    ElementCube,
    Fingerprints,
    LabelState,
    Labels,
    MaskSet,
    PCAFeatures,
    Space,
    TiledArtifacts,
    payload_from_h5,
)


def _cube():
    return ElementCube(
        pixels=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        element_names=("Fe", "Mg", "Ca", "Si"),
        space=Space.RAW,
        downsample_factor=2,
        header_trim_px=100,
    )


def test_replace_returns_new_instance():
    cube = _cube()
    denoised = cube.replace(space=Space.DENOISED)
    assert denoised.space is Space.DENOISED
    assert cube.space is Space.RAW  # original untouched


def test_payloads_are_frozen():
    with pytest.raises(Exception):
        _cube().pixels = None  # type: ignore[misc]


def _roundtrip(payload, tmp_path):
    import h5py

    path = tmp_path / "payload.h5"
    with h5py.File(path, "w") as fh:
        payload.to_h5(fh.create_group("p"))
    with h5py.File(path, "r") as fh:
        return payload_from_h5(fh["p"])


def test_element_cube_h5_roundtrip(tmp_path):
    cube = _cube().replace(
        space=Space.NORMALIZED,
        means=np.ones(4, dtype=np.float32),
        stds=np.full(4, 2.0, dtype=np.float32),
    )
    back = _roundtrip(cube, tmp_path)
    assert isinstance(back, ElementCube)
    np.testing.assert_array_equal(back.pixels, cube.pixels)
    assert back.element_names == cube.element_names
    assert back.space is Space.NORMALIZED
    np.testing.assert_array_equal(back.means, cube.means)
    assert back.downsample_factor == 2
    assert back.header_trim_px == 100


def test_element_cube_h5_roundtrip_none_means(tmp_path):
    back = _roundtrip(_cube(), tmp_path)
    assert back.means is None
    assert back.stds is None


def test_bse_image_h5_roundtrip(tmp_path):
    bse = BseImage(pixels=np.eye(3, dtype=np.float32))
    back = _roundtrip(bse, tmp_path)
    assert isinstance(back, BseImage)
    np.testing.assert_array_equal(back.pixels, bse.pixels)


def test_mask_set_h5_roundtrip(tmp_path):
    masks = MaskSet(
        mineral_mask=np.array([[True, False], [False, True]]),
        valid_mask=None,
        stats={"n_mineral": 2, "pct": 50.0},
    )
    back = _roundtrip(masks, tmp_path)
    assert isinstance(back, MaskSet)
    np.testing.assert_array_equal(back.mineral_mask, masks.mineral_mask)
    assert back.valid_mask is None
    assert back.stats == masks.stats


def test_pca_features_h5_roundtrip(tmp_path):
    features = PCAFeatures(
        features=np.ones((5, 3), dtype=np.float32),
        mineral_indices=np.zeros((5, 2), dtype=np.int32),
        image_shape=(8, 9),
        explained_variance_ratio=np.array([0.7, 0.2, 0.1]),
        n_kept=3,
    )
    back = _roundtrip(features, tmp_path)
    assert isinstance(back, PCAFeatures)
    np.testing.assert_array_equal(back.features, features.features)
    assert back.image_shape == (8, 9)
    assert back.n_kept == 3
    np.testing.assert_array_equal(
        back.explained_variance_ratio, features.explained_variance_ratio
    )


def test_labels_h5_roundtrip(tmp_path):
    labels = Labels(
        labels=np.array([0, 1, -1], dtype=np.int32),
        probabilities=np.array([0.9, 0.8, 0.0], dtype=np.float32),
        mineral_indices=np.zeros((3, 2), dtype=np.int32),
        image_shape=(4, 4),
        state=LabelState.RAW,
    )
    back = _roundtrip(labels, tmp_path)
    assert back.state is LabelState.RAW
    np.testing.assert_array_equal(back.labels, labels.labels)
    np.testing.assert_array_equal(back.probabilities, labels.probabilities)
    assert back.image_shape == (4, 4)


def test_labels_h5_roundtrip_none_probabilities(tmp_path):
    labels = Labels(
        labels=np.array([0], dtype=np.int32),
        probabilities=None,
        mineral_indices=np.zeros((1, 2), dtype=np.int32),
        image_shape=(2, 2),
        state=LabelState.CLEANED,
    )
    back = _roundtrip(labels, tmp_path)
    assert back.probabilities is None
    assert back.state is LabelState.CLEANED


def test_tiled_artifacts_h5_roundtrip(tmp_path):
    from karak.clustering.tiling import PhaseEntry, TileResult

    artifacts = TiledArtifacts(
        tile_results=(
            TileResult(
                tile_id=0, n_pixels=10, n_clusters=2, n_noise=1,
                local_labels=np.array([0, 1, -1], dtype=np.int32),
                merge_map={0: 0, 1: 1},
                new_phases=[0, 1],
            ),
        ),
        phase_registry=(
            PhaseEntry(
                global_id=0,
                mean_fingerprint=np.array([0.5, 0.2], dtype=np.float64),
                n_pixels=6,
                discovered_in_tile=0,
                tile_contributions={0: 6},
            ),
        ),
        tile_size=32,
    )
    back = _roundtrip(artifacts, tmp_path)
    assert isinstance(back, TiledArtifacts)
    assert back.tile_size == 32
    tr = back.tile_results[0]
    assert (tr.tile_id, tr.n_pixels, tr.n_clusters, tr.n_noise) == (0, 10, 2, 1)
    np.testing.assert_array_equal(tr.local_labels, [0, 1, -1])
    assert tr.merge_map == {0: 0, 1: 1}
    assert tr.new_phases == [0, 1]
    pe = back.phase_registry[0]
    assert (pe.global_id, pe.n_pixels, pe.discovered_in_tile) == (0, 6, 0)
    np.testing.assert_array_equal(pe.mean_fingerprint, [0.5, 0.2])
    assert pe.tile_contributions == {0: 6}


def test_cluster_stats_h5_roundtrip(tmp_path):
    stats = ClusterStats(stats={"n_clusters": 3, "noise_pct": 1.5})
    back = _roundtrip(stats, tmp_path)
    assert back.stats == stats.stats


def test_fingerprints_h5_roundtrip(tmp_path):
    data = {
        "fingerprints": {
            0: {"mean": np.array([0.1, 0.2]), "std": np.array([0.01, 0.02]),
                "n_pixels": 5, "area_pct": 50.0},
        },
        "element_names": ["Fe", "Mg"],
        "element_order": np.array([1, 0]),
        "n_clusters": 1,
        "n_mineral_pixels": 10,
    }
    payload = Fingerprints(data=data, similar_pairs=[(0, 1, 0.97)])
    back = _roundtrip(payload, tmp_path)
    assert set(back.data["fingerprints"]) == {0}  # int keys survive
    np.testing.assert_allclose(back.data["fingerprints"][0]["mean"], [0.1, 0.2])
    assert back.data["element_names"] == ["Fe", "Mg"]
    np.testing.assert_array_equal(back.data["element_order"], [1, 0])
    assert back.similar_pairs == [(0, 1, 0.97)]
