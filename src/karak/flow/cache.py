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


def _payload_path(recipe: str, port: str, cache_dir: str | Path) -> Path:
    return Path(cache_dir) / f"{recipe}__{port}.h5"


def store_payload(
    recipe: str, port: str, payload, cache_dir: str | Path
) -> Path:
    path = _payload_path(recipe, port, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as fh:
        payload.to_h5(fh.create_group("payload"))
    os.replace(tmp, path)
    return path


def load_payload(recipe: str, port: str, cache_dir: str | Path):
    """Load a cached payload, or None if it is not in the cache."""
    path = _payload_path(recipe, port, cache_dir)
    if not path.exists():
        return None
    with h5py.File(path, "r") as fh:
        return payload_from_h5(fh["payload"])


def has_payload(recipe: str, port: str, cache_dir: str | Path) -> bool:
    return _payload_path(recipe, port, cache_dir).exists()
