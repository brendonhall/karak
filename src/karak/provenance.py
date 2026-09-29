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


def git_info() -> dict | None:
    """Commit and dirty flag of the karak source tree, or None outside git."""
    import subprocess
    from pathlib import Path

    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(here), *args],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty}


def host_info() -> dict[str, str]:
    import platform
    import socket

    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
