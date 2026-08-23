"""Tests for recipe hashing and the payload cache store."""

from __future__ import annotations

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


def test_load_missing_returns_none(tmp_path):
    assert load_payload("no_such", "stats", tmp_path) is None
