"""Tests for recipe hashing and the payload cache store."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from karak.flow.cache import (
    CacheWriter,
    has_payload,
    load_payload,
    load_summary,
    recipe_hash,
    store_payload,
)
from karak.stages.payloads import ClusterStats


def test_same_inputs_same_hash():
    a = recipe_hash("denoise", {"sigma": 1.0}, {"cube": "abc"}, None)
    b = recipe_hash("denoise", {"sigma": 1.0}, {"cube": "abc"}, None)
    assert a == b


def test_param_change_changes_hash():
    a = recipe_hash("denoise", {"sigma": 1.0}, {"cube": "abc"}, None)
    b = recipe_hash("denoise", {"sigma": 2.0}, {"cube": "abc"}, None)
    assert a != b


def test_upstream_hash_change_propagates():
    a = recipe_hash("denoise", {"sigma": 1.0}, {"cube": "abc"}, None)
    b = recipe_hash("denoise", {"sigma": 1.0}, {"cube": "xyz"}, None)
    assert a != b


def test_source_signature_changes_hash():
    a = recipe_hash("load", {}, {}, "sig1")
    b = recipe_hash("load", {}, {}, "sig2")
    assert a != b


def test_store_and_load_roundtrip(tmp_path):
    payload = ClusterStats(stats={"n_clusters": 4})
    store_payload("deadbeef", "stats", payload, tmp_path)
    back = load_payload("deadbeef", "stats", tmp_path)
    assert back.stats == {"n_clusters": 4}


def test_store_records_the_upstream_recipes(tmp_path):
    from karak.flow.cache import load_upstream

    payload = ClusterStats(stats={})
    path = store_payload("abc", "stats", payload, tmp_path,
                         upstream={"labels": "111", "cube": "222"})
    assert load_upstream(path) == {"labels": "111", "cube": "222"}
    bare = store_payload("def", "stats", payload, tmp_path)
    assert load_upstream(bare) == {}


def test_load_missing_returns_none(tmp_path):
    assert load_payload("no_such", "stats", tmp_path) is None


def test_summary_sidecar_roundtrip(tmp_path):
    from karak.flow.cache import load_summary, store_summary

    store_summary("deadbeef", "cube", "ElementCube 2×2×1 float32 16 B space=raw", tmp_path)
    assert load_summary("deadbeef", "cube", tmp_path) == (
        "ElementCube 2×2×1 float32 16 B space=raw"
    )
    assert (tmp_path / "deadbeef__cube.summary.txt").exists()


def test_missing_summary_returns_none(tmp_path):
    from karak.flow.cache import load_summary

    assert load_summary("nope", "cube", tmp_path) is None


def test_store_payload_passes_the_compression_through(tmp_path):
    import h5py

    from karak.stages.payloads import BseImage

    payload = BseImage(pixels=np.zeros((4, 4), np.float32))
    path = store_payload("r1", "bse", payload, tmp_path, compression="none")
    with h5py.File(path) as fh:
        assert fh["payload"]["pixels"].compression is None
    path = store_payload("r2", "bse", payload, tmp_path)
    with h5py.File(path) as fh:
        assert fh["payload"]["pixels"].compression == "lzf"


def _bse(n=4):
    from karak.stages.payloads import BseImage

    return BseImage(pixels=np.zeros((n, n), np.float32))


def test_writer_writes_entries_complete_or_not_at_all(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    gate = threading.Event()
    real = cache.store_payload

    def slow_store(*args, **kwargs):
        gate.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "BseImage 4×4", {"cube": "abc"})
    time.sleep(0.05)
    assert not has_payload("r1", "bse", tmp_path)   # still in the queue
    assert not list(tmp_path.glob("*__bse.h5"))
    gate.set()
    writer.wait_for("r1", "bse")
    assert has_payload("r1", "bse", tmp_path)
    assert load_summary("r1", "bse", tmp_path) == "BseImage 4×4"
    assert ("r1", "bse") in writer.written
    writer.close()


def test_writer_wait_drains_in_fifo_order(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    order = []
    real = cache.store_payload

    def recording_store(recipe, port, *args, **kwargs):
        order.append((recipe, port))
        return real(recipe, port, *args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", recording_store)
    writer = CacheWriter(tmp_path)
    for i in range(5):
        writer.submit(f"r{i}", "bse", _bse(), "", {})
    writer.wait()
    assert order == [(f"r{i}", "bse") for i in range(5)]
    assert writer.seconds > 0
    writer.close()


def test_writer_exception_surfaces_at_wait(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "", {})
    with pytest.raises(OSError, match="disk full"):
        writer.wait()
    with pytest.raises(OSError, match="disk full"):   # sticky until close
        writer.wait_for("r1", "bse")
    writer.close()


def test_writer_close_never_raises_after_an_error(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "", {})
    writer.close()          # must not raise
    writer.close()
    with pytest.raises(OSError, match="disk full"):
        writer.wait()


def test_writer_log_lines_name_the_entry_and_the_time(tmp_path):
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "cube", _bse(), "", {}, label="dn.cube")
    writer.wait()
    lines = writer.drain_log()
    assert len(lines) == 1
    assert lines[0].startswith("cache: dn.cube written in ")
    assert lines[0].endswith(" s")
    assert writer.per_label["dn.cube"] > 0
    assert writer.drain_log() == []
    writer.close()


def test_writer_close_is_idempotent_and_wait_after_close_is_a_noop(tmp_path):
    writer = CacheWriter(tmp_path)
    writer.submit("r1", "bse", _bse(), "", {})
    writer.close()
    writer.close()
    writer.wait()
    assert has_payload("r1", "bse", tmp_path)


def test_tmp_name_is_unique_per_process(tmp_path):
    import os

    from karak.flow.cache import _tmp_path, payload_path

    tmp = _tmp_path(payload_path("r1", "bse", tmp_path))
    assert tmp.name == f"r1__bse.h5.{os.getpid()}.tmp"
