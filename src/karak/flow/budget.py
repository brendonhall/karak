"""Host memory budget for payloads held between nodes (standard library)."""

from __future__ import annotations

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


def default_ram_budget() -> int:
    """Half of the available memory, or 8 GB when it cannot be read."""
    available = available_host_memory()
    return FALLBACK_BUDGET if available is None else available // 2
