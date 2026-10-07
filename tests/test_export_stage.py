"""Tests for the export_h5 sink: same HDF5 layout as the legacy pipeline."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from conftest import complete
from karak.flow.graph import Graph, Node
from karak.stages import get
from karak.stages.base import StageError
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
    """The complete flow the executor embeds via the {flow} token."""
    graph = complete(Graph(name="test", nodes=(
        Node("msk", "mask", {"min_object_size": 33}),
        Node("dn", "denoise", {"method": "bilateral"}),
        Node("nrm", "normalize"),
    )))
    return json.dumps(graph.to_json())


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
        assert fh["denoised"].attrs["sigma_spatial"] == 1.0   # from the flow


def test_export_writes_the_deferred_flag_per_tile(tmp_path, payloads):
    from karak.clustering.tiling import PhaseEntry, TileResult
    from karak.stages.payloads import TiledArtifacts

    tile = TileResult(
        tile_id=0, n_pixels=64, n_clusters=1, n_noise=64,
        local_labels=np.full(64, -1, dtype=np.int32), merge_map={},
        new_phases=[], deferred=True,
    )
    phase = PhaseEntry(
        global_id=0, mean_fingerprint=np.zeros(2, dtype=np.float32),
        n_pixels=0, discovered_in_tile=0,
    )
    tiles = TiledArtifacts(
        tile_results=(tile,), phase_registry=(phase,), tile_size=8,
    )
    path = str(tmp_path / "tiled.h5")
    get("export_h5")().run(
        {**payloads, "tiles": tiles}, {"path": path, "flow_json": _flow_json()}
    )

    with h5py.File(path, "r") as fh:
        assert fh["clusters/tiled/tile_deferred"][()].tolist() == [True]


def test_export_partial_inputs(tmp_path, payloads):
    path = str(tmp_path / "partial.h5")
    subset = {k: payloads[k] for k in ("cube_raw", "bse", "masks")}
    get("export_h5")().run(subset, {"path": path, "flow_json": _flow_json()})

    with h5py.File(path, "r") as fh:
        assert set(fh["raw"]) == {"Fe", "Mg"}
        assert "cube" not in fh["denoised"]
        assert "cleaned_labels" not in fh["clusters"]


def test_export_refuses_a_flow_without_the_producing_node(tmp_path, payloads):
    """Recorded params come from the flow that ran, never from code."""
    subset = {k: payloads[k] for k in ("cube_raw", "masks")}
    with pytest.raises(StageError, match="complete 'mask' node"):
        get("export_h5")().run(
            subset, {"path": str(tmp_path / "x.h5"), "flow_json": "{}"}
        )


def _datasets(fh):
    found = {}
    fh.visititems(lambda name, obj: found.__setitem__(name, obj[()])
                  if isinstance(obj, h5py.Dataset) else None)
    return found


@pytest.mark.parametrize("compression", ["gzip", "lzf", "none"])
def test_export_compression_round_trips_every_dataset(tmp_path, payloads, compression):
    path = str(tmp_path / f"{compression}.h5")
    get("export_h5")().run(payloads, {"path": path, "flow_json": _flow_json(),
                                      "compression": compression})
    with h5py.File(path, "r") as fh:
        np.testing.assert_array_equal(fh["denoised/cube"][()],
                                      payloads["cube_denoised"].pixels)
        np.testing.assert_array_equal(fh["normalized/cube"][()],
                                      payloads["cube_normalized"].pixels)
        np.testing.assert_array_equal(fh["raw/Mg"][()],
                                      payloads["cube_raw"].pixels[:, :, 1])
        np.testing.assert_array_equal(fh["clusters/mineral_indices"][()],
                                      payloads["labels"].mineral_indices)
        cube = fh["denoised/cube"]
        if compression == "none":
            assert cube.compression is None and cube.chunks is None
        else:
            assert cube.compression == compression and cube.shuffle
            assert cube.chunks == (8, 8, 2)
            raw = fh["raw/Mg"]
            assert raw.compression == compression and not raw.shuffle
            if compression == "gzip":
                assert cube.compression_opts == 4


def test_export_compressions_hold_the_same_data(tmp_path, payloads):
    files = {}
    for compression in ("gzip", "lzf", "none"):
        path = str(tmp_path / f"{compression}.h5")
        get("export_h5")().run(payloads, {"path": path, "flow_json": _flow_json(),
                                          "compression": compression})
        with h5py.File(path, "r") as fh:
            files[compression] = _datasets(fh)
    assert files["gzip"].keys() == files["lzf"].keys() == files["none"].keys()
    for name, data in files["none"].items():
        np.testing.assert_array_equal(files["gzip"][name], data)
        np.testing.assert_array_equal(files["lzf"][name], data)


def test_dataset_options_chunks_and_filters():
    from karak.io.storage import dataset_options

    cube = np.zeros((6525, 3990, 19), np.float32)
    assert dataset_options(cube, "gzip", shuffle=True) == {
        "shuffle": True, "chunks": (256, 256, 19),
        "compression": "gzip", "compression_opts": 4}
    assert dataset_options(cube, "gzip", shuffle=False) == {
        "chunks": (256, 256, 19), "compression": "gzip", "compression_opts": 4}
    assert dataset_options(cube, "lzf", shuffle=True)["chunks"] == (256, 256, 19)
    assert dataset_options(cube, "none", shuffle=True) == {}
    small = np.zeros((100, 30), bool)
    assert dataset_options(small, "gzip", shuffle=False)["chunks"] == (100, 30)
    labels = np.zeros(12_495_787, np.int32)
    assert dataset_options(labels, "gzip", shuffle=False)["chunks"] == (262_144,)
    indices = np.zeros((12_495_787, 2), np.int32)
    assert dataset_options(indices, "gzip", shuffle=False)["chunks"] == (131_072, 2)
    assert dataset_options(np.zeros(5), "gzip", shuffle=False)["chunks"] == (5,)
    assert dataset_options(np.zeros((0, 2)), "gzip", shuffle=False) == {}
    with pytest.raises(ValueError, match="compression"):
        dataset_options(cube, "zstd", shuffle=False)


def test_export_writes_names_and_history(tmp_path, payloads):
    from karak.io.storage import load_mineral_names

    labels = payloads["labels"].replace(
        names={0: "Olivine", 1: "Augite"},
        history=({"stage": "split_gmm", "parent": 0, "new_labels": [1],
                  "names": {1: "Augite"}, "n_pixels": {1: 3},
                  "method": "GMM on Ca", "note": "why"},),
    )
    path = tmp_path / "out.h5"
    get("export_h5")().run(
        {"labels": labels, "stats": payloads["stats"], "features": payloads["features"]},
        {"path": str(path), "flow_json": _flow_json(), "compression": "none"},
    )
    assert load_mineral_names(path) == {0: "Olivine", 1: "Augite"}
    with h5py.File(path, "r") as fh:
        sub = fh["clusters/subclustering"]
        assert json.loads(sub.attrs["history"])[0]["new_labels"] == [1]
        assert json.loads(sub.attrs["split_00"])["parent"] == 0
        assert fh["clusters"].attrs["cluster_1_name"] == "Augite"
