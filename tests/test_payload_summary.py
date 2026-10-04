"""One-line summaries that the run dashboard shows for each output."""

from __future__ import annotations

import numpy as np

from karak.stages.payloads import (
    BseImage,
    ClusterStats,
    ElementCube,
    Fingerprints,
    Labels,
    LabelState,
    MaskSet,
    PCAFeatures,
    Space,
    TiledArtifacts,
    format_bytes,
)


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_bytes(2_400) == "2 kB"
    assert format_bytes(104_000_000) == "104 MB"
    assert format_bytes(1_978_000_000) == "1.98 GB"


def test_element_cube_summary():
    cube = ElementCube(
        pixels=np.zeros((10, 20, 3), np.float32),
        element_names=("A", "B", "C"),
        space=Space.RAW,
    )
    assert cube.summary() == "ElementCube 10×20×3 float32 2 kB space=raw"


def test_bse_summary():
    bse = BseImage(pixels=np.zeros((1000, 1000), np.float32))
    assert bse.summary() == "BseImage 1000×1000 float32 4 MB"


def test_mask_set_summary():
    mineral = np.zeros((10, 10), bool)
    mineral.flat[:25] = True
    valid = np.zeros((10, 10), bool)
    valid[:5] = True
    assert MaskSet(mineral_mask=mineral).summary() == "MaskSet mineral 25.0%"
    assert (
        MaskSet(mineral_mask=mineral, valid_mask=valid).summary()
        == "MaskSet mineral 25.0% · valid 50.0%"
    )


def test_pca_summary():
    pca = PCAFeatures(
        features=np.zeros((100, 3), np.float32),
        mineral_indices=np.zeros((100, 2), np.int32),
        image_shape=(10, 10),
        explained_variance_ratio=np.array([0.5, 0.3, 0.1, 0.1]),
        n_kept=3,
    )
    assert pca.summary() == "PCAFeatures 100 px × 3 components (90.0% variance)"


def test_labels_summary():
    labels = Labels(
        labels=np.array([0, 0, 1, 2, -1, -1], np.int32),
        probabilities=None,
        mineral_indices=np.zeros((6, 2), np.int32),
        image_shape=(2, 3),
        state=LabelState.RAW,
    )
    assert labels.summary() == "Labels 6 px · 3 phases · 2 noise · state=raw"


def test_labels_summary_without_noise():
    labels = Labels(
        labels=np.array([0, 1], np.int32),
        probabilities=None,
        mineral_indices=np.zeros((2, 2), np.int32),
        image_shape=(1, 2),
        state=LabelState.CLEANED,
    )
    assert labels.summary() == "Labels 2 px · 2 phases · state=cleaned"


def test_tiled_cluster_stats_fingerprints_summaries():
    tiled = TiledArtifacts(tile_results=(), phase_registry=(), tile_size=1024)
    assert tiled.summary() == "TiledArtifacts 0 tiles · 0 phases · tile_size=1024"
    assert ClusterStats(stats={"a": 1, "b": 2}).summary() == "ClusterStats 2 entries"
    fp = Fingerprints(data={"p1": {}}, similar_pairs=[])
    assert fp.summary() == "Fingerprints 1 entries · 0 similar pairs"


def test_clustering_stats_and_fingerprints_count_phases():
    stats = ClusterStats(stats={"n_clusters": 2, "n_noise": 1234, "noise_pct": 1.0,
                                "n_total": 10_000,
                                "clusters": {0: {}, 1: {}}})
    assert stats.summary() == "ClusterStats 2 phases · 1,234 noise"
    fp = Fingerprints(
        data={"fingerprints": {0: {}, 1: {}, 2: {}},
              "element_names": ["Al", "Fe-K", "Si", "Ca"],
              "element_order": np.array([2, 1, 3, 0]),
              "n_clusters": 3, "n_mineral_pixels": 10},
        similar_pairs=[(0, 1, 0.97)])
    assert fp.summary() == "Fingerprints 3 phases · top Si, Fe-K, Ca · 1 similar pairs"
