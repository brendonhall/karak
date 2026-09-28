"""Process memory sampling for the run dashboard (standard library only).

With --workers, pool processes are not counted; the dashboard labels the
figure "main process" in that case.
"""

from __future__ import annotations

import os
import sys
import threading


def current_rss() -> int | None:
    """Resident set size of this process in bytes, or None without /proc."""
    try:
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def peak_rss() -> int:
    """Peak resident set size of this process in bytes (0 if unknown).

    ``ru_maxrss`` and ``/proc/self/statm`` count pages slightly differently,
    so the peak is never reported below the current RSS.
    """
    current = current_rss() or 0
    try:
        import resource
    except ImportError:
        return current
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = peak if sys.platform == "darwin" else peak * 1024
    return max(peak, current)


class MemorySampler:
    """Samples current and peak RSS on a daemon thread."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.current: int | None = current_rss()
        self.peak: int = peak_rss()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample(self) -> None:
        self.current = current_rss()
        self.peak = max(self.peak, peak_rss())

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="karak-memory", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.sample()
