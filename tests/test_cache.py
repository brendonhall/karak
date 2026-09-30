"""Tests for recipe hashing and the payload cache store."""

from __future__ import annotations

import numpy as np

from karak.flow.cache import load_payload, recipe_hash, store_payload
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
