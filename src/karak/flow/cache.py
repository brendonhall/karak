"""Content-addressed payload cache.

A node's recipe hash folds its stage type, coerced params, upstream hashes,
and (for source nodes) a source signature. Identical recipes reuse the
cached output written under ``<cache_dir>/<hash>__<port>.h5``. Files are
written to a temp path and atomically renamed, so a cache entry either
exists complete or not at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

import h5py

from karak.stages.payloads import payload_from_h5


def recipe_hash(
    stage_type: str,
    params: dict,
    upstream_hashes: dict,
    source_sig: str | None,
) -> str:
    recipe = {
        "type": stage_type,
        "params": params,
        "upstream": dict(sorted(upstream_hashes.items())),
        "source": source_sig,
    }
    canonical = json.dumps(recipe, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


def payload_path(recipe: str, port: str, cache_dir: str | Path) -> Path:
    """Where a cached payload lives (whether or not it exists yet)."""
    return Path(cache_dir) / f"{recipe}__{port}.h5"



def store_payload(
    recipe: str, port: str, payload, cache_dir: str | Path,
    upstream: dict | None = None, compression: str = "lzf",
) -> Path:
    """Write a payload to the cache. ``upstream`` maps the producing
    node's input ports to the recipe hashes they consumed; it is stored
    as an attribute so a cache scan can tell which outputs belong together.
    ``compression`` is the HDF5 filter: "lzf" (default), "gzip" or "none"."""
    path = payload_path(recipe, port, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with h5py.File(tmp, "w") as fh:
        group = fh.create_group("payload")
        payload.to_h5(group, compression=compression)
        if upstream:
            group.attrs["upstream"] = json.dumps(dict(sorted(upstream.items())))
    os.replace(tmp, path)
    return path


def _tmp_path(path: Path) -> Path:
    """A tmp name unique to this process, so two runs that race on the same
    entry never write the same file."""
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def load_upstream(path: str | Path) -> dict:
    """The upstream recipes stored with a cached payload; {} when the file
    predates the attribute or has none (a source node's output)."""
    with h5py.File(path, "r") as fh:
        raw = fh["payload"].attrs.get("upstream")
    return json.loads(str(raw)) if raw is not None else {}


def load_payload(recipe: str, port: str, cache_dir: str | Path):
    """Load a cached payload, or None if it is not in the cache."""
    path = payload_path(recipe, port, cache_dir)
    if not path.exists():
        return None
    with h5py.File(path, "r") as fh:
        return payload_from_h5(fh["payload"])


def has_payload(recipe: str, port: str, cache_dir: str | Path) -> bool:
    return payload_path(recipe, port, cache_dir).exists()


def _summary_path(recipe: str, port: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{recipe}__{port}.summary.txt"


def store_summary(
    recipe: str, port: str, text: str, cache_dir: str | Path
) -> None:
    """Store a payload's one-line summary next to its cached payload.

    A cached step whose output nobody consumes is never loaded, so the
    run dashboard reads this text instead of the payload.
    """
    path = _summary_path(recipe, port, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_summary(recipe: str, port: str, cache_dir: str | Path) -> str | None:
    """The stored summary text, or None for entries cached without one."""
    path = _summary_path(recipe, port, cache_dir)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


class CacheWriter:
    """Writes payloads to the cache on one background thread, FIFO.

    ``submit`` enqueues a host payload; the thread calls ``store_payload``
    and ``store_summary``. ``wait_for`` and ``wait`` block on the disk state
    and re-raise the first error the thread hit. ``close`` never raises. The
    thread never talks to a reporter: it appends log lines that ``drain_log``
    hands back to the caller's thread.
    """

    def __init__(self, cache_dir: str | Path, compression: str = "lzf"):
        self.cache_dir = Path(cache_dir)
        self.compression = compression
        self.written: set[tuple[str, str]] = set()
        self.seconds = 0.0
        self.per_label: dict[str, float] = {}   # "dn.cube" -> seconds
        self._queue: queue.Queue = queue.Queue()
        self._done = threading.Condition()
        self._error: BaseException | None = None
        self._pending: set[tuple[str, str]] = set()   # submitted, not yet done
        self._log: list[str] = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="karak-cache-writer", daemon=True)
        self._thread.start()

    def submit(self, recipe: str, port: str, payload, summary: str,
               upstream: dict | None, label: str | None = None) -> None:
        if self._closed:
            raise RuntimeError("CacheWriter is closed")
        with self._done:
            self._pending.add((recipe, port))
        self._queue.put((recipe, port, payload, summary, upstream,
                         label or f"{recipe[:8]}.{port}"))

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            recipe, port, payload, summary, upstream, label = item
            started = time.monotonic()
            try:
                if self._error is None:
                    store_payload(recipe, port, payload, self.cache_dir,
                                  upstream=upstream,
                                  compression=self.compression)
                    store_summary(recipe, port, summary, self.cache_dir)
            except BaseException as exc:   # surfaces at the next wait
                with self._done:
                    self._error = exc
            else:
                if self._error is None:
                    elapsed = time.monotonic() - started
                    with self._done:
                        self.seconds += elapsed
                        self.written.add((recipe, port))
                        self.per_label[label] = (
                            self.per_label.get(label, 0.0) + elapsed)
                        self._log.append(
                            f"cache: {label} written in {elapsed:.1f} s")
            finally:
                with self._done:
                    self._pending.discard((recipe, port))
                    self._queue.task_done()
                    self._done.notify_all()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def wait_for(self, recipe: str, port: str) -> None:
        """Block until (recipe, port), if submitted, is on disk.

        Returns at once for a key never submitted. Raises a writer error.
        """
        with self._done:
            while (recipe, port) in self._pending and self._error is None:
                self._done.wait(0.05)
        self._raise_if_failed()

    def wait(self) -> None:
        if self._thread.is_alive():
            self._queue.join()
        self._raise_if_failed()

    def drain_log(self) -> list[str]:
        with self._done:
            lines, self._log = self._log, []
        return lines

    def close(self) -> None:
        """Drain the queue and stop the thread. Never raises; idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.join()
        finally:
            self._queue.put(None)
            self._thread.join()
