"""Tests for the export_h5 sink: same HDF5 layout as the legacy pipeline."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from karak.stages import get
from karak.stages.payloads import (
    BseImage,
    ClusterStats,
    ElementCube,
    LabelState,
    Labels,
    MaskSet,
    PCAFeatures,
    Space,
)


@pytest.fixture()
def payloads():
    H = W = 8
    pixels = np.random.default_rng(0).random((H, W, 2)).astype(np.float32)
    n = H * W
    indices = np.stack(
        np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), axis=-1
    ).reshape(-1, 2).astype(np.int32)
    return {
        "cube_raw": ElementCube(
            pixels=pixels, element_names=("Fe", "Mg"), space=Space.RAW,
            downsample_factor=2,
        ),
        "bse": BseImage(pixels=np.zeros((H, W), dtype=np.float32)),
        "masks": MaskSet(
            mineral_mask=np.ones((H, W), dtype=bool),
            valid_mask=None,
            stats={"coverage_pct": 100.0},
        ),
        "cube_denoised": ElementCube(
            pixels=pixels, element_names=("Fe", "Mg"), space=Space.DENOISED,
        ),
        "cube_normalized": ElementCube(
            pixels=pixels, element_names=("Fe", "Mg"), space=Space.NORMALIZED,
            means=np.zeros(2, dtype=np.float32),
            stds=np.ones(2, dtype=np.float32),
        ),
        "features": PCAFeatures(
            features=np.zeros((n, 2), dtype=np.float32),
            mineral_indices=indices,
            image_shape=(H, W),
            explained_variance_ratio=np.array([0.8, 0.2]),
            n_kept=2,
        ),
        "labels_raw": Labels(
            labels=np.zeros(n, dtype=np.int32),
            probabilities=np.ones(n, dtype=np.float32),
            mineral_indices=indices,
            image_shape=(H, W),
            state=LabelState.RAW,
        ),
        "labels": Labels(
            labels=np.zeros(n, dtype=np.int32),
            probabilities=np.ones(n, dtype=np.float32),
            mineral_indices=indices,
            image_shape=(H, W),
            state=LabelState.CLEANED,
        ),
        "stats": ClusterStats(
            stats={"n_clusters": 1, "n_noise": 0, "noise_pct": 0.0,
                   "n_total": n},
        ),
    }


def _flow_json():
    return json.dumps({
        "version": 1, "name": "test",
        "nodes": [
            {"id": "msk", "type": "mask", "params": {"min_object_size": 33}},
            {"id": "dn", "type": "denoise", "params": {"method": "bilateral"}},
        ],
        "edges": [],
    })


def test_export_writes_legacy_layout(tmp_path, payloads):
    path = str(tmp_path / "out.h5")
    get("export_h5")().run(
        payloads, {"path": path, "flow_json": _flow_json()}
    )

    with h5py.File(path, "r") as fh:
        # root provenance
        assert "pipeline_config" in fh.attrs
        assert "library_versions" in fh.attrs
        # groups + key datasets
        assert set(fh["raw"]) == {"Fe", "Mg"}
        assert fh["bse"]["image"].shape == (8, 8)
        assert fh["masks"]["mineral"].shape == (8, 8)
        assert fh["denoised"]["cube"].shape == (8, 8, 2)
        assert fh["denoised"].attrs["method"] == "bilateral"
        assert fh["normalized"]["cube"].shape == (8, 8, 2)
        assert fh["clusters"]["cleaned_labels"].shape == (64,)
        assert fh["clusters"].attrs["n_clusters"] == 1
        # per-group config provenance comes from the flow nodes
        assert "33" in fh["masks"].attrs["mask_config"]


def test_export_partial_inputs(tmp_path, payloads):
    path = str(tmp_path / "partial.h5")
    subset = {k: payloads[k] for k in ("cube_raw", "bse", "masks")}
    get("export_h5")().run(subset, {"path": path, "flow_json": "{}"})

    with h5py.File(path, "r") as fh:
        assert set(fh["raw"]) == {"Fe", "Mg"}
        assert "cube" not in fh["denoised"]
        assert "cleaned_labels" not in fh["clusters"]
