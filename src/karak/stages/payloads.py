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
