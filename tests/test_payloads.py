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


def test_labels_roundtrip_names_and_history(tmp_path):
    labels = Labels(
        labels=np.array([0, 1, 2], dtype=np.int32),
        probabilities=None,
        mineral_indices=np.zeros((3, 2), dtype=np.int32),
        image_shape=(2, 2),
        state=LabelState.CLEANED,
        names={0: "Olivine", 1: "Augite", 2: "Pigeonite"},
        history=({"stage": "split_gmm", "parent": 1, "new_labels": [2],
                  "names": {2: "Pigeonite"}, "n_pixels": {2: 1},
                  "method": "GMM on Ca", "note": "why",
                  "component_means": {1: {"Ca": 0.1}, 2: {"Ca": 0.9}}},),
    )
    back = _roundtrip(labels, tmp_path)
    assert back.names == {0: "Olivine", 1: "Augite", 2: "Pigeonite"}
    assert back.history[0]["new_labels"] == [2]
    assert back.history[0]["names"] == {2: "Pigeonite"}
    assert back.history[0]["n_pixels"] == {2: 1}
    assert back.history[0]["component_means"] == {1: {"Ca": 0.1}, 2: {"Ca": 0.9}}
    # old files have neither attribute
    import h5py
    with h5py.File(tmp_path / "old.h5", "w") as fh:
        labels.to_h5(fh.create_group("p"))
        del fh["p"].attrs["names"], fh["p"].attrs["history"]
    with h5py.File(tmp_path / "old.h5", "r") as fh:
        old = payload_from_h5(fh["p"])
    assert old.names == {} and old.history == ()


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


def test_tiled_artifacts_roundtrip_deferred_fields(tmp_path):
    from karak.clustering.tiling import TileResult

    artifacts = TiledArtifacts(
        tile_results=(
            TileResult(tile_id=0, n_pixels=3, n_clusters=1, n_noise=3,
                       local_labels=np.array([0, 0, 0], dtype=np.int32),
                       merge_map={}, new_phases=[], deferred=True),
            TileResult(tile_id=1, n_pixels=2, n_clusters=2, n_noise=0,
                       local_labels=np.array([0, 1], dtype=np.int32),
                       merge_map={0: 0, 1: 1}, new_phases=[0, 1]),
        ),
        phase_registry=(),
        tile_size=32,
        deferred_pixels=np.array([0, 1, 2], dtype=np.int64),
    )
    back = _roundtrip(artifacts, tmp_path)
    assert back.tile_results[0].deferred is True
    assert back.tile_results[1].deferred is False
    assert back.deferred_tiles == (0,)
    np.testing.assert_array_equal(back.deferred_pixels, [0, 1, 2])


def test_tiled_artifacts_without_deferred_fields_loads_as_none_deferred(tmp_path):
    """Cache files written before the deferred-tile rule have no such
    attributes; they load with no deferred tiles and no deferred pixels."""
    import h5py

    from karak.clustering.tiling import TileResult

    artifacts = TiledArtifacts(
        tile_results=(TileResult(tile_id=0, n_pixels=1, n_clusters=1, n_noise=0,
                                 local_labels=np.array([0], dtype=np.int32),
                                 merge_map={0: 0}, new_phases=[0]),),
        phase_registry=(), tile_size=32,
    )
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as fh:
        artifacts.to_h5(fh.create_group("p"))
        del fh["p/tiles/0"].attrs["deferred"]        # as an old file has
        del fh["p/deferred_pixels"]
    with h5py.File(path, "r") as fh:
        back = payload_from_h5(fh["p"])
    assert back.deferred_tiles == ()
    assert back.deferred_pixels.size == 0


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
    data["names"] = {0: "A"}
    payload = Fingerprints(data=data, similar_pairs=[(0, 1, 0.97)])
    back = _roundtrip(payload, tmp_path)
    assert back.data["names"] == {0: "A"}
    assert set(back.data["fingerprints"]) == {0}  # int keys survive
    np.testing.assert_allclose(back.data["fingerprints"][0]["mean"], [0.1, 0.2])
    assert back.data["element_names"] == ["Fe", "Mg"]
    np.testing.assert_array_equal(back.data["element_order"], [1, 0])
    assert back.similar_pairs == [(0, 1, 0.97)]


@pytest.mark.parametrize("compression", ["lzf", "gzip", "none"])
def test_every_payload_roundtrips_under_each_compression(tmp_path, compression):
    """Each payload class writes with the requested filter and reads back."""
    import h5py

    cube = _cube()
    payloads = [
        cube,
        BseImage(pixels=np.zeros((4, 5), np.float32)),
        MaskSet(mineral_mask=np.ones((4, 5), bool), valid_mask=None),
        PCAFeatures(features=np.zeros((7, 3), np.float32),
                    mineral_indices=np.zeros((7, 2), np.int32),
                    image_shape=(4, 5),
                    explained_variance_ratio=np.array([0.5, 0.3, 0.2]),
                    n_kept=2),
        Labels(labels=np.zeros(7, np.int32), probabilities=None,
               mineral_indices=np.zeros((7, 2), np.int32),
               image_shape=(4, 5), state=LabelState.RAW),
        ClusterStats(stats={"n": 1}),
    ]
    for payload in payloads:
        path = tmp_path / f"{type(payload).__name__}.h5"
        with h5py.File(path, "w") as fh:
            payload.to_h5(fh.create_group("p"), compression=compression)
        with h5py.File(path, "r") as fh:
            back = payload_from_h5(fh["p"])
            pixels = fh["p"].get("pixels")
            if pixels is not None:
                expected = None if compression == "none" else compression
                assert pixels.compression == expected
                assert pixels.chunks is not None
        assert type(back) is type(payload)


def test_to_h5_defaults_to_lzf(tmp_path):
    import h5py

    with h5py.File(tmp_path / "d.h5", "w") as fh:
        _cube().to_h5(fh.create_group("p"))
    with h5py.File(tmp_path / "d.h5", "r") as fh:
        assert fh["p"]["pixels"].compression == "lzf"


def test_one_dimensional_labels_are_chunked_and_compressed(tmp_path):
    import h5py

    labels = Labels(labels=np.zeros(7, np.int32), probabilities=None,
                    mineral_indices=np.zeros((7, 2), np.int32),
                    image_shape=(4, 5), state=LabelState.RAW)
    with h5py.File(tmp_path / "l.h5", "w") as fh:
        labels.to_h5(fh.create_group("p"))
    with h5py.File(tmp_path / "l.h5", "r") as fh:
        assert fh["p"]["labels"].compression == "lzf"
        assert fh["p"]["labels"].chunks == (7,)


def test_gzip_file_written_before_the_change_still_loads(tmp_path):
    """HDF5 records the filter per dataset; the reader does not care."""
    import h5py

    cube = _cube()
    with h5py.File(tmp_path / "old.h5", "w") as fh:
        g = fh.create_group("p")
        g.attrs["payload_type"] = "element_cube"
        g.attrs["element_names"] = list(cube.element_names)
        g.attrs["space"] = "raw"
        g.attrs["downsample_factor"] = 2
        g.attrs["header_trim_px"] = 100
        g.attrs["left_trim_px"] = 0
        g.create_dataset("pixels", data=cube.pixels, compression="gzip")
    with h5py.File(tmp_path / "old.h5", "r") as fh:
        back = payload_from_h5(fh["p"])
    np.testing.assert_array_equal(back.pixels, cube.pixels)


import karak.accel as accel  # noqa: E402
from karak.stages.payloads import payload_nbytes  # noqa: E402


class _FakeDeviceArray:
    __module__ = "cupy"

    def __init__(self, host):
        self.host = np.asarray(host)
        self.shape, self.dtype, self.nbytes = self.host.shape, self.host.dtype, self.host.nbytes

    def get(self):
        return self.host


class _FakeCupy:
    """Enough of cupy for to(): asarray wraps, .get() unwraps."""

    @staticmethod
    def asarray(arr):
        return arr if isinstance(arr, _FakeDeviceArray) else _FakeDeviceArray(arr)


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: True)
    monkeypatch.setattr(accel, "get_array_module",
                        lambda device: _FakeCupy if device == "cuda" else np)


def test_host_payload_is_cpu_and_to_cpu_is_self():
    cube = _cube()
    assert cube.device == "cpu"
    assert cube.to("cpu") is cube


def test_to_cuda_moves_every_array_field_and_back(fake_cuda):
    cube = _cube().replace(means=np.ones(4, np.float32), stds=np.ones(4, np.float32))
    on_device = cube.to("cuda")
    assert on_device is not cube
    assert on_device.device == "cuda"
    assert isinstance(on_device.pixels, _FakeDeviceArray)
    assert isinstance(on_device.means, _FakeDeviceArray)
    assert on_device.element_names == cube.element_names
    assert on_device.to("cuda") is on_device
    back = on_device.to("cpu")
    assert back.device == "cpu"
    np.testing.assert_array_equal(back.pixels, cube.pixels)
    np.testing.assert_array_equal(back.means, cube.means)


def test_to_keeps_none_fields(fake_cuda):
    masks = MaskSet(mineral_mask=np.ones((2, 2), bool), valid_mask=None)
    on_device = masks.to("cuda")
    assert on_device.valid_mask is None
    assert isinstance(on_device.mineral_mask, _FakeDeviceArray)
    assert on_device.to("cpu").valid_mask is None


def test_payloads_without_arrays_are_cpu(fake_cuda):
    stats = ClusterStats(stats={"n": 1})
    assert stats.device == "cpu"
    assert stats.to("cuda") is stats


def test_payload_nbytes_splits_host_and_device(fake_cuda):
    cube = _cube()
    host, device = payload_nbytes(cube)
    assert (host, device) == (cube.pixels.nbytes, 0)
    host, device = payload_nbytes(cube.to("cuda"))
    assert (host, device) == (0, cube.pixels.nbytes)
    assert payload_nbytes(ClusterStats(stats={})) == (0, 0)
