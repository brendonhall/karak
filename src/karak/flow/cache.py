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
