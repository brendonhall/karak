"""Host memory figures (standard library), for the numeric core and the
flow layer alike."""

from __future__ import annotations

import os

MEMINFO = "/proc/meminfo"


def available_host_memory() -> int | None:
    """``MemAvailable`` in bytes, or None where /proc/meminfo is absent."""
    try:
        with open(MEMINFO) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def physical_memory() -> int | None:
    """Physical memory in bytes from ``os.sysconf``, or None where the
    platform does not report it."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
