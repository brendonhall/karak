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


def _gated_writer(tmp_path, monkeypatch, n=5):
    import karak.flow.cache as cache

    gate = threading.Event()
    real = cache.store_payload

    def slow_store(*args, **kwargs):
        gate.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)
    writer = CacheWriter(tmp_path)
    for i in range(n):
        writer.submit(f"r{i}", "bse", _bse(), "", {})
    return writer, gate


def test_wait_for_unknown_key_after_close_returns(tmp_path):
    writer = CacheWriter(tmp_path)
    writer.close()
    t = threading.Thread(target=writer.wait_for, args=("never", "x"))
    t.start()
    t.join(2)
    assert not t.is_alive()


def test_wait_for_never_submitted_key_does_not_wait_for_the_queue(
        tmp_path, monkeypatch):
    writer, gate = _gated_writer(tmp_path, monkeypatch)
    t = threading.Thread(target=writer.wait_for, args=("never", "bse"))
    t.start()
    t.join(1)
    finished = not t.is_alive()
    gate.set()
    writer.close()
    assert finished


def test_wait_for_blocks_until_that_entry_is_written(tmp_path, monkeypatch):
    writer, gate = _gated_writer(tmp_path, monkeypatch)
    t = threading.Thread(target=writer.wait_for, args=("r4", "bse"))
    t.start()
    t.join(0.2)
    assert t.is_alive()
    gate.set()
    t.join(5)
    assert not t.is_alive()
    assert has_payload("r4", "bse", tmp_path)
    writer.close()


class _BrokenPayload:
    """A payload whose write fails partway (a full disk)."""

    def to_h5(self, group, compression="lzf"):
        group.create_dataset("partial", data=np.zeros(4))
        raise OSError("disk full")


def test_failed_write_leaves_no_tmp_file(tmp_path):
    with pytest.raises(OSError, match="disk full"):
        store_payload("r1", "bse", _BrokenPayload(), tmp_path)
    assert list(tmp_path.iterdir()) == []


def _dead_pid() -> int:
    import subprocess
    import sys

    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_sweep_removes_dead_tmp_files_and_keeps_live_ones(tmp_path):
    import os

    from karak.flow.cache import sweep_stale_tmp

    dead = tmp_path / f"r1__bse.h5.{_dead_pid()}.tmp"
    live = tmp_path / f"r2__bse.h5.{os.getpid()}.tmp"
    odd = tmp_path / "r3__bse.h5.notapid.tmp"
    for path in (dead, live, odd):
        path.write_bytes(b"x")
    sweep_stale_tmp(tmp_path)
    assert not dead.exists()
    assert live.exists()
    assert odd.exists()


def test_sweep_of_a_missing_cache_dir_is_a_noop(tmp_path):
    from karak.flow.cache import sweep_stale_tmp

    sweep_stale_tmp(tmp_path / "nope")


def test_store_summary_writes_through_a_tmp_file(tmp_path, monkeypatch):
    import os

    import karak.flow.cache as cache

    replaced = []
    real = os.replace

    def spy(src, dst):
        replaced.append((str(src), str(dst)))
        return real(src, dst)

    monkeypatch.setattr(cache.os, "replace", spy)
    cache.store_summary("r1", "bse", "text", tmp_path)
    assert replaced and replaced[0][0].endswith(f".{os.getpid()}.tmp")
    assert load_summary("r1", "bse", tmp_path) == "text"
    assert [p.name for p in tmp_path.iterdir()] == ["r1__bse.summary.txt"]


def test_check_raises_the_sticky_error_and_passes_otherwise(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    writer = CacheWriter(tmp_path)
    writer.check()                          # no error yet: returns

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    writer.submit("r1", "bse", _bse(), "", {})
    deadline = time.monotonic() + 5
    while writer._error is None and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(OSError, match="disk full"):
        writer.check()
    writer.close()


def test_interrupt_during_close_discards_the_queue(tmp_path, monkeypatch):
    writer, gate = _gated_writer(tmp_path, monkeypatch, n=5)
    real_join = writer._queue.join
    calls = []

    def interrupted_join():
        if not calls:
            calls.append(1)
            raise KeyboardInterrupt
        return real_join()

    monkeypatch.setattr(writer._queue, "join", interrupted_join)
    timer = threading.Timer(0.3, gate.set)   # the item in flight finishes
    timer.start()
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        writer.close()
    assert time.monotonic() - started < 6
    timer.join()
    assert not writer._thread.is_alive()
    assert writer._pending == set()
    written = {p.name.split("__")[0] for p in tmp_path.glob("*__bse.h5")}
    assert written <= {"r0"}                  # r1..r4 were discarded
    assert not list(tmp_path.glob("*.tmp"))


def test_dataset_treats_compression_none_as_no_filter(tmp_path):
    import h5py

    from karak.stages.payloads import _dataset

    with h5py.File(tmp_path / "f.h5", "w") as fh:
        ds = _dataset(fh, "a", np.zeros((4, 4)), compression=None)
        assert ds.compression is None


# --- review fixes: release finished payloads, bound queued bytes ------------

def test_idle_writer_does_not_retain_the_last_written_payload(tmp_path):
    import gc
    import weakref

    writer = CacheWriter(tmp_path)
    payload = _bse(64)
    ref = weakref.ref(payload)
    writer.submit("r1", "bse", payload, "", {})
    del payload
    writer.wait()
    gc.collect()
    assert ref() is None            # collected while the writer is still open
    assert writer._thread.is_alive()
    writer.close()


def test_submit_blocks_while_queued_bytes_exceed_the_bound(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    gate = threading.Event()
    real = cache.store_payload

    def slow_store(*args, **kwargs):
        gate.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "store_payload", slow_store)
    one = _bse(16).pixels.nbytes                      # 1 KiB each
    writer = CacheWriter(tmp_path, max_pending_bytes=2 * one)
    accepted = []

    def produce():
        for i in range(5):
            writer.submit(f"r{i}", "bse", _bse(16), "", {})
            accepted.append(i)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    time.sleep(0.3)
    assert accepted == [0, 1]                         # third submit waits
    assert writer.pending_bytes == 2 * one
    gate.set()
    producer.join(10)
    writer.wait()
    assert accepted == [0, 1, 2, 3, 4]
    assert writer.pending_bytes == 0
    assert all(has_payload(f"r{i}", "bse", tmp_path) for i in range(5))
    writer.close()


def test_a_payload_larger_than_the_bound_is_admitted_alone(tmp_path):
    writer = CacheWriter(tmp_path, max_pending_bytes=10)
    writer.submit("big", "bse", _bse(64), "", {})     # 16 KiB > 10 bytes
    writer.wait()
    assert has_payload("big", "bse", tmp_path)
    writer.close()


def test_submit_does_not_block_after_a_writer_error(tmp_path, monkeypatch):
    import karak.flow.cache as cache

    def failing_store(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cache, "store_payload", failing_store)
    writer = CacheWriter(tmp_path, max_pending_bytes=1)
    for i in range(3):
        writer.submit(f"r{i}", "bse", _bse(16), "", {})
    with pytest.raises(OSError):
        writer.wait()
    writer.close()
