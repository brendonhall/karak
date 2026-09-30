"""Host memory budget for payloads held between nodes (standard library)."""

from __future__ import annotations

import os

MEMINFO = "/proc/meminfo"
FALLBACK_BUDGET = 8 * 2**30


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


def default_ram_budget() -> int:
    """Half of the available memory; without /proc/meminfo, half of the
    physical memory; 8 GB when neither can be read."""
    available = available_host_memory()
    if available is None:
        available = physical_memory()
    return FALLBACK_BUDGET if available is None else available // 2
