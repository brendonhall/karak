"""What produced a result: software versions for run records and HDF5."""

from __future__ import annotations

import importlib.metadata

_LIBRARIES = [
    "numpy",
    "scikit-image",
    "scikit-learn",
    "h5py",
    "hdbscan",
    "medpy",
    "imageio",
    "pyyaml",
    "scipy",
]


def karak_version() -> str:
    try:
        return importlib.metadata.version("karak")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def get_software_versions() -> dict[str, str]:
    """Return a dict of key library versions for provenance tracking."""
    versions: dict[str, str] = {}
    for lib in _LIBRARIES:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    return versions
