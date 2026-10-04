"""Host memory budget for payloads held between nodes (standard library)."""

from __future__ import annotations

from karak.memory import available_host_memory, physical_memory

FALLBACK_BUDGET = 8 * 2**30


def default_ram_budget() -> int:
    """Half of the available memory; without /proc/meminfo, half of the
    physical memory; 8 GB when neither can be read."""
    available = available_host_memory()
    if available is None:
        available = physical_memory()
    return FALLBACK_BUDGET if available is None else available // 2
