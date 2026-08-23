"""Payload types that flow between stage ports.

Every payload is an immutable dataclass with a ``.replace()`` helper, so
``apply()`` never mutates its inputs and caching stays correct. Payloads are
self-describing: cubes carry a ``space`` tag and labels a ``state`` tag that
ports can type-check on the wire.

Geometry metadata (downsample factor, edge trims) travels inside
``ElementCube`` so downstream stages (e.g. napari polygon-mask loading) never
need a side channel to interpret coordinates.

Each payload serializes to/from an HDF5 group (``to_h5`` / ``from_h5``);
``payload_from_h5`` dispatches on the group's ``payload_type`` attribute.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Space(str, Enum):
    RAW = "raw"
    DENOISED = "denoised"
    NORMALIZED = "normalized"


class LabelState(str, Enum):
    RAW = "raw"          # may contain -1 (noise / unassigned)
    CLEANED = "cleaned"  # every mineral pixel has a phase label


_PAYLOAD_TYPES: dict = {}


def _payload(cls):
    _PAYLOAD_TYPES[cls.payload_type] = cls
    return cls


def payload_from_h5(group):
    """Reconstruct a payload from an HDF5 group written by ``to_h5``."""
    kind = group.attrs["payload_type"]
    return _PAYLOAD_TYPES[kind].from_h5(group)


class _Replaceable:
    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


@_payload
@dataclass(frozen=True)
class ElementCube(_Replaceable):
    """(H, W, C) element cube plus channel names and geometry metadata."""

    payload_type = "element_cube"

    pixels: np.ndarray                      # (H, W, C) float32
    element_names: tuple                    # length C
    space: Space
    means: np.ndarray | None = None         # (C,), set when NORMALIZED
    stds: np.ndarray | None = None          # (C,), set when NORMALIZED
    downsample_factor: int = 1
    header_trim_px: int = 0
    left_trim_px: int = 0

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["element_names"] = list(self.element_names)
        group.attrs["space"] = self.space.value
        group.attrs["downsample_factor"] = self.downsample_factor
        group.attrs["header_trim_px"] = self.header_trim_px
        group.attrs["left_trim_px"] = self.left_trim_px
        group.create_dataset("pixels", data=self.pixels, compression="gzip")
        if self.means is not None:
            group.create_dataset("means", data=self.means)
        if self.stds is not None:
            group.create_dataset("stds", data=self.stds)

    @classmethod
    def from_h5(cls, group) -> "ElementCube":
        return cls(
            pixels=group["pixels"][()],
            element_names=tuple(str(n) for n in group.attrs["element_names"]),
            space=Space(group.attrs["space"]),
            means=group["means"][()] if "means" in group else None,
            stds=group["stds"][()] if "stds" in group else None,
            downsample_factor=int(group.attrs["downsample_factor"]),
            header_trim_px=int(group.attrs["header_trim_px"]),
            left_trim_px=int(group.attrs["left_trim_px"]),
        )


@_payload
@dataclass(frozen=True)
class BseImage(_Replaceable):
    """(H, W) backscatter-electron grayscale image."""

    payload_type = "bse_image"

    pixels: np.ndarray

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.create_dataset("pixels", data=self.pixels, compression="gzip")

    @classmethod
    def from_h5(cls, group) -> "BseImage":
        return cls(pixels=group["pixels"][()])


@_payload
@dataclass(frozen=True)
class MaskSet(_Replaceable):
    """Mineral / valid-region masks plus their summary statistics."""

    payload_type = "mask_set"

    mineral_mask: np.ndarray                # (H, W) bool
    valid_mask: np.ndarray | None = None    # (H, W) bool
    stats: dict = dataclasses.field(default_factory=dict)

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["stats"] = json.dumps(self.stats)
        group.create_dataset(
            "mineral_mask", data=self.mineral_mask, compression="gzip"
        )
        if self.valid_mask is not None:
            group.create_dataset(
                "valid_mask", data=self.valid_mask, compression="gzip"
            )

    @classmethod
    def from_h5(cls, group) -> "MaskSet":
        return cls(
            mineral_mask=group["mineral_mask"][()].astype(bool),
            valid_mask=(
                group["valid_mask"][()].astype(bool)
                if "valid_mask" in group else None
            ),
            stats=json.loads(group.attrs["stats"]),
        )


@_payload
@dataclass(frozen=True)
class PCAFeatures(_Replaceable):
    """PCA-reduced mineral-pixel features plus their image coordinates."""

    payload_type = "pca_features"

    features: np.ndarray                    # (N_mineral, n_kept) float32
    mineral_indices: np.ndarray             # (N_mineral, 2) int32 (row, col)
    image_shape: tuple                      # (H, W)
    explained_variance_ratio: np.ndarray    # full EVR from the PCA fit
    n_kept: int

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["image_shape"] = list(self.image_shape)
        group.attrs["n_kept"] = self.n_kept
        group.create_dataset("features", data=self.features, compression="gzip")
        group.create_dataset(
            "mineral_indices", data=self.mineral_indices, compression="gzip"
        )
        group.create_dataset(
            "explained_variance_ratio", data=self.explained_variance_ratio
        )

    @classmethod
    def from_h5(cls, group) -> "PCAFeatures":
        return cls(
            features=group["features"][()],
            mineral_indices=group["mineral_indices"][()],
            image_shape=tuple(int(v) for v in group.attrs["image_shape"]),
            explained_variance_ratio=group["explained_variance_ratio"][()],
            n_kept=int(group.attrs["n_kept"]),
        )


@_payload
@dataclass(frozen=True)
class Labels(_Replaceable):
    """Per-mineral-pixel phase labels."""

    payload_type = "labels"

    labels: np.ndarray                      # (N_mineral,) int32
    probabilities: np.ndarray | None        # (N_mineral,) float32
    mineral_indices: np.ndarray             # (N_mineral, 2) int32
    image_shape: tuple                      # (H, W)
    state: LabelState

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["image_shape"] = list(self.image_shape)
        group.attrs["state"] = self.state.value
        group.create_dataset("labels", data=self.labels, compression="gzip")
        group.create_dataset(
            "mineral_indices", data=self.mineral_indices, compression="gzip"
        )
        if self.probabilities is not None:
            group.create_dataset(
                "probabilities", data=self.probabilities, compression="gzip"
            )

    @classmethod
    def from_h5(cls, group) -> "Labels":
        return cls(
            labels=group["labels"][()],
            probabilities=(
                group["probabilities"][()] if "probabilities" in group else None
            ),
            mineral_indices=group["mineral_indices"][()],
            image_shape=tuple(int(v) for v in group.attrs["image_shape"]),
            state=LabelState(group.attrs["state"]),
        )


@_payload
@dataclass(frozen=True)
class TiledArtifacts(_Replaceable):
    """Tile diagnostics + phase registry from tiled clustering."""

    payload_type = "tiled_artifacts"

    tile_results: tuple                     # tuple[TileResult, ...]
    phase_registry: tuple                   # tuple[PhaseEntry, ...]
    tile_size: int

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["tile_size"] = self.tile_size
        tiles = group.create_group("tiles")
        for i, tr in enumerate(self.tile_results):
            sub = tiles.create_group(str(i))
            sub.attrs["tile_id"] = tr.tile_id
            sub.attrs["n_pixels"] = tr.n_pixels
            sub.attrs["n_clusters"] = tr.n_clusters
            sub.attrs["n_noise"] = tr.n_noise
            sub.attrs["merge_map"] = json.dumps(
                {str(k): int(v) for k, v in tr.merge_map.items()}
            )
            sub.attrs["new_phases"] = [int(p) for p in tr.new_phases]
            sub.create_dataset(
                "local_labels", data=tr.local_labels, compression="gzip"
            )
        registry = group.create_group("registry")
        for i, entry in enumerate(self.phase_registry):
            sub = registry.create_group(str(i))
            sub.attrs["global_id"] = entry.global_id
            sub.attrs["n_pixels"] = entry.n_pixels
            sub.attrs["discovered_in_tile"] = entry.discovered_in_tile
            sub.attrs["tile_contributions"] = json.dumps(
                {str(k): int(v) for k, v in entry.tile_contributions.items()}
            )
            sub.create_dataset("mean_fingerprint", data=entry.mean_fingerprint)

    @classmethod
    def from_h5(cls, group) -> "TiledArtifacts":
        from karak.clustering.tiling import PhaseEntry, TileResult

        tile_results = []
        for key in sorted(group["tiles"], key=int):
            sub = group["tiles"][key]
            tile_results.append(TileResult(
                tile_id=int(sub.attrs["tile_id"]),
                n_pixels=int(sub.attrs["n_pixels"]),
                n_clusters=int(sub.attrs["n_clusters"]),
                n_noise=int(sub.attrs["n_noise"]),
                local_labels=sub["local_labels"][()],
                merge_map={
                    int(k): v
                    for k, v in json.loads(sub.attrs["merge_map"]).items()
                },
                new_phases=[int(p) for p in sub.attrs["new_phases"]],
            ))
        phase_registry = []
        for key in sorted(group["registry"], key=int):
            sub = group["registry"][key]
            phase_registry.append(PhaseEntry(
                global_id=int(sub.attrs["global_id"]),
                mean_fingerprint=sub["mean_fingerprint"][()],
                n_pixels=int(sub.attrs["n_pixels"]),
                discovered_in_tile=int(sub.attrs["discovered_in_tile"]),
                tile_contributions={
                    int(k): v
                    for k, v in json.loads(
                        sub.attrs["tile_contributions"]
                    ).items()
                },
            ))
        return cls(
            tile_results=tuple(tile_results),
            phase_registry=tuple(phase_registry),
            tile_size=int(group.attrs["tile_size"]),
        )


def _jsonable(value):
    """Recursively convert numpy values so json.dumps accepts them."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@_payload
@dataclass(frozen=True)
class ClusterStats(_Replaceable):
    """Summary statistics for a clustering result."""

    payload_type = "cluster_stats"

    stats: dict

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["stats"] = json.dumps(_jsonable(self.stats))

    @classmethod
    def from_h5(cls, group) -> "ClusterStats":
        return cls(stats=json.loads(group.attrs["stats"]))


@_payload
@dataclass(frozen=True)
class Fingerprints(_Replaceable):
    """Per-cluster chemical fingerprints + flagged similar pairs."""

    payload_type = "fingerprints"

    data: dict                              # compute_fingerprints() result
    similar_pairs: list = dataclasses.field(default_factory=list)

    def to_h5(self, group) -> None:
        group.attrs["payload_type"] = self.payload_type
        group.attrs["data"] = json.dumps(_jsonable(self.data))
        group.attrs["similar_pairs"] = json.dumps(_jsonable(self.similar_pairs))

    @classmethod
    def from_h5(cls, group) -> "Fingerprints":
        data = json.loads(group.attrs["data"])
        if "fingerprints" in data:
            data["fingerprints"] = {
                int(label): {
                    "mean": np.asarray(entry["mean"]),
                    "std": np.asarray(entry["std"]),
                    "n_pixels": entry["n_pixels"],
                    "area_pct": entry["area_pct"],
                }
                for label, entry in data["fingerprints"].items()
            }
        if "element_order" in data:
            data["element_order"] = np.asarray(data["element_order"])
        pairs = [
            tuple(pair)
            for pair in json.loads(group.attrs["similar_pairs"])
        ]
        return cls(data=data, similar_pairs=pairs)
