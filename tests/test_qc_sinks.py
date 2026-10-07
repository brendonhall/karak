"""Smoke tests: every qc_* sink writes at least one figure file."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from karak.stages import get
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
)


H = W = 16


@pytest.fixture()
def payloads():
    rng = np.random.default_rng(0)
    pixels = rng.random((H, W, 2)).astype(np.float32)
    mask = np.ones((H, W), dtype=bool)
    indices = np.stack(
        np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), axis=-1
    ).reshape(-1, 2).astype(np.int32)
    n = len(indices)
    labels = (indices[:, 1] >= W // 2).astype(np.int32)
    return {
        "cube_raw": ElementCube(
            pixels=pixels, element_names=("Fe", "Mg"), space=Space.RAW,
        ),
        "cube_denoised": ElementCube(
            pixels=pixels, element_names=("Fe", "Mg"), space=Space.DENOISED,
        ),
        "cube_normalized": ElementCube(
            pixels=pixels, element_names=("Fe", "Mg"), space=Space.NORMALIZED,
            means=np.zeros(2), stds=np.ones(2),
        ),
        "bse": BseImage(pixels=np.zeros((H, W), dtype=np.float32)),
        "masks": MaskSet(mineral_mask=mask, valid_mask=None,
                         stats={"coverage_pct": 100.0, "n_mineral": n}),
        "features": PCAFeatures(
            features=rng.random((n, 2)).astype(np.float32),
            mineral_indices=indices, image_shape=(H, W),
            explained_variance_ratio=np.array([0.8, 0.2]), n_kept=2,
        ),
        "labels_raw": Labels(
            labels=labels, probabilities=np.ones(n, dtype=np.float32),
            mineral_indices=indices, image_shape=(H, W), state=LabelState.RAW,
        ),
        "labels": Labels(
            labels=labels, probabilities=np.ones(n, dtype=np.float32),
            mineral_indices=indices, image_shape=(H, W),
            state=LabelState.CLEANED,
        ),
        "stats": ClusterStats(stats={
            "n_clusters": 2, "n_noise": 0, "noise_pct": 0.0, "n_total": n,
            "cluster_sizes": {"0": n // 2, "1": n // 2},
            "cluster_pcts": {"0": 50.0, "1": 50.0},
            "mean_probability": 1.0,
        }),
        "fingerprints": Fingerprints(
            data={
                "fingerprints": {
                    0: {"mean": np.array([0.5, 0.1]),
                        "std": np.array([0.05, 0.01]),
                        "n_pixels": n // 2, "area_pct": 50.0},
                    1: {"mean": np.array([0.1, 0.5]),
                        "std": np.array([0.01, 0.05]),
                        "n_pixels": n // 2, "area_pct": 50.0},
                },
                "element_names": ["Fe", "Mg"],
                "element_order": np.array([0, 1]),
                "n_clusters": 2,
                "n_mineral_pixels": n,
            },
            similar_pairs=[],
        ),
    }


def _figures_written(tmp_path):
    return list(tmp_path.glob("*.png"))


def _run(sink_id, inputs, tmp_path, extra_params=None):
    params = {"figure_dir": str(tmp_path)}
    params.update(extra_params or {})
    get(sink_id)().run(inputs, params)
    figures = _figures_written(tmp_path)
    assert figures, f"{sink_id} wrote no figure"


def test_qc_mask(tmp_path, payloads):
    _run("qc_mask", {k: payloads[k] for k in ("bse", "masks", "cube_raw")},
         tmp_path)


def test_qc_denoise(tmp_path, payloads):
    _run("qc_denoise",
         {"cube_raw": payloads["cube_raw"],
          "cube_denoised": payloads["cube_denoised"],
          "bse": payloads["bse"], "masks": payloads["masks"]},
         tmp_path)


def test_qc_normalize(tmp_path, payloads):
    _run("qc_normalize",
         {"cube": payloads["cube_normalized"], "masks": payloads["masks"]},
         tmp_path)
    assert len(_figures_written(tmp_path)) == 2  # histograms + correlation


def test_qc_scree(tmp_path, payloads):
    _run("qc_scree", {"features": payloads["features"]}, tmp_path)


def test_qc_phase_map(tmp_path, payloads):
    _run("qc_phase_map",
         {"labels_raw": payloads["labels_raw"], "labels": payloads["labels"],
          "bse": payloads["bse"], "stats": payloads["stats"]},
         tmp_path)


def test_qc_cluster_summary(tmp_path, payloads):
    _run("qc_cluster_summary", {"stats": payloads["stats"]}, tmp_path)


def test_qc_tiled(tmp_path, payloads):
    from karak.clustering.tiling import PhaseEntry, TileResult

    tiles = TiledArtifacts(
        tile_results=(
            TileResult(tile_id=0, n_pixels=H * W, n_clusters=2, n_noise=0,
                       local_labels=np.zeros(H * W, dtype=np.int32),
                       merge_map={0: 0}, new_phases=[0]),
        ),
        phase_registry=(
            PhaseEntry(global_id=0, mean_fingerprint=np.array([0.5, 0.1]),
                       n_pixels=H * W, discovered_in_tile=0,
                       tile_contributions={0: H * W}),
        ),
        tile_size=8,
    )
    _run("qc_tiled",
         {"bse": payloads["bse"], "tiles": tiles,
          "features": payloads["features"]},
         tmp_path, {"min_tile_pixels": 1})
    assert len(_figures_written(tmp_path)) == 2  # overlay + discovery chart


def test_qc_fingerprints(tmp_path, payloads):
    _run("qc_fingerprints", {"fingerprints": payloads["fingerprints"]},
         tmp_path)


def test_qc_named_phase_map(tmp_path, payloads):
    _run("qc_named_phase_map",
         {"labels": payloads["labels"], "bse": payloads["bse"]},
         tmp_path, {"mineral_names": '{"0": "olivine", "1": "augite"}'})


def test_qc_named_phase_map_uses_the_label_names(tmp_path, payloads, monkeypatch):
    seen = {}

    def fake(_image, _bse, names, _figure_dir):
        seen["names"] = names

    monkeypatch.setattr("karak.qc.figures.generate_named_phase_map", fake)
    labels = payloads["labels"].replace(names={0: "Olivine", 1: "Augite"})
    get("qc_named_phase_map")().run(
        {"labels": labels, "bse": payloads["bse"]},
        {"figure_dir": str(tmp_path), "mineral_names": "{}"},
    )
    assert seen["names"] == {0: "Olivine", 1: "Augite"}
    get("qc_named_phase_map")().run(
        {"labels": labels, "bse": payloads["bse"]},
        {"figure_dir": str(tmp_path), "mineral_names": '{"0": "Fo"}'},
    )
    assert seen["names"] == {0: "Fo"}


def test_qc_fingerprints_uses_the_carried_names(tmp_path, payloads, monkeypatch):
    seen = {}

    def fake(_data, _figure_dir, mineral_names=None):
        seen["names"] = mineral_names

    monkeypatch.setattr("karak.qc.figures.generate_fingerprint_chart", fake)
    fp = payloads["fingerprints"]
    fp = fp.replace(data={**fp.data, "names": {0: "Olivine"}})
    get("qc_fingerprints")().run(
        {"fingerprints": fp}, {"figure_dir": str(tmp_path), "mineral_names": None}
    )
    assert seen["names"] == {0: "Olivine"}
    get("qc_fingerprints")().run(
        {"fingerprints": fp},
        {"figure_dir": str(tmp_path), "mineral_names": '{"0": "Fo"}'},
    )
    assert seen["names"] == {0: "Fo"}
    get("qc_fingerprints")().run(
        {"fingerprints": fp}, {"figure_dir": str(tmp_path), "mineral_names": "{}"}
    )
    assert seen["names"] == {0: "Olivine"}                 # "{}" falls back like null


def _gappy(payloads):
    """Labels 0, 1, 3: a split emptied label 2, and the largest label (3)
    exceeds the cluster count (3)."""
    fp = payloads["fingerprints"]
    d = dict(fp.data)
    d["fingerprints"] = {
        0: d["fingerprints"][0], 1: d["fingerprints"][1],
        3: {"mean": np.array([0.3, 0.3]), "std": np.array([0.02, 0.02]),
            "n_pixels": 10, "area_pct": 1.0},
    }
    d["n_clusters"] = 3
    return Fingerprints(data=d, similar_pairs=[])


def test_a_label_has_one_colour_in_every_figure(tmp_path, payloads, monkeypatch):
    """Labels 0, 1, 3 (2 absent): label 3 gets the same colour from the
    phase map and from the fingerprint chart, and the named map."""
    from karak.qc import figures

    assert figures.label_color(3) == tuple(float(c) for c in plt.cm.tab20(3))
    assert figures.label_color(23) == figures.label_color(3)      # fixed cycle of 20
    assert figures.label_color(-1) == (0.85, 0.85, 0.85, 1.0)

    asked = {}
    real = figures.label_color
    shape = (2, 3)
    cleaned = np.array([[0, 1, 3], [3, 1, 0]], dtype=np.int32)
    raw = np.array([[0, -1, 3], [3, 1, -1]], dtype=np.int32)
    bse = np.zeros(shape, dtype=np.float32)

    def calls(name, run):
        seen = []
        monkeypatch.setattr(
            figures, "label_color", lambda lab: (seen.append(int(lab)), real(lab))[1])
        out = run()
        monkeypatch.setattr(figures, "label_color", real)
        assert out and (tmp_path / out.rsplit("/", 1)[1]).exists()
        asked[name] = set(seen)

    calls("phase", lambda: figures.generate_phase_map(
        raw, cleaned, bse, {"n_clusters": 3, "noise_pct": 0.0}, tmp_path))
    calls("fingerprints", lambda: figures.generate_fingerprint_chart(
        _gappy(payloads).data, tmp_path))
    calls("named", lambda: figures.generate_named_phase_map(
        cleaned, bse, {0: "a", 1: "b", 3: "c"}, tmp_path))
    for name, labels in asked.items():
        assert 3 in labels, name
    # The colormap the phase map builds gives label 3 the fingerprint colour.
    cmap, vmax = figures._label_cmap({-1, 0, 1, 3})
    assert tuple(cmap(3 + 1)) == figures.label_color(3) and vmax == 4
