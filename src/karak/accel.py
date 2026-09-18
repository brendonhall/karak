"""Optional acceleration helpers: CPU worker resolution and CUDA access.

The whole GPU surface of karak lives here. CuPy is imported lazily so a
plain install never touches it.
"""

from __future__ import annotations

import os

import numpy as np

from karak.stages.base import StageError

_INSTALL_HINT = (
    "device='cuda' requested but no usable GPU stack was found. "
    "Install the extra with: pip install 'karak[cuda]' "
    "(needs an NVIDIA driver and CUDA 12)."
)


def resolve_workers(workers: int | None) -> int:
    """None -> 1 (serial), 0 -> all cores, N -> N."""
    if workers is None:
        return 1
    if workers < 0:
        raise ValueError(f"workers must be >= 0, got {workers}")
    if workers == 0:
        return os.cpu_count() or 1
    return workers


def cuda_available() -> bool:
    """True when CuPy imports and sees at least one device."""
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_name() -> str | None:
    """Name of GPU 0, or None without a usable CUDA stack."""
    if not cuda_available():
        return None
    import cupy

    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def get_array_module(device: str):
    """numpy for 'cpu', cupy for 'cuda'; StageError otherwise."""
    if device == "cpu":
        return np
    if device == "cuda":
        if not cuda_available():
            raise StageError(_INSTALL_HINT)
        import cupy

        return cupy
    raise StageError(f"unknown device {device!r}; expected 'cpu' or 'cuda'")


def to_device(arr, device: str):
    """Move an array to the device. No-op on CPU."""
    if device == "cpu":
        return arr
    return get_array_module(device).asarray(arr)


def to_numpy(arr) -> np.ndarray:
    """Bring an array back to host memory. No-op for numpy arrays."""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "get"):  # cupy
        return arr.get()
    return np.asarray(arr)
