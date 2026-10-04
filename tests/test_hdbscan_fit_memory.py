"""The cpu HDBSCAN fit refuses to start when its memory estimate exceeds
80 % of the available memory."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import hdbscan_cfg
import karak.memory as memory
from karak.clustering.hdbscan_cluster import (
    check_fit_memory,
    fit_memory_bytes,
    run_hdbscan,
)
from karak.errors import StageError


def test_estimate_matches_the_measured_fits():
    # measured peak RSS above the start: 24.1 GB at 782,419 x 1000 and
    # 12.7 GB at 400,000 x 1000 on NWA 4587 features
    assert fit_memory_bytes(782_419, 1000) == pytest.approx(24.1e9, rel=0.05)
    assert fit_memory_bytes(400_000, 1000) == pytest.approx(12.7e9, rel=0.05)


def test_check_passes_under_the_limit_and_without_a_figure():
    check_fit_memory(50_000, 12_495_787, 1000, available=30e9)   # 1.6 GB
    check_fit_memory(10**9, 10**9, 1000, available=None)


def test_check_refuses_an_oversized_fit_with_the_remedy():
    with pytest.raises(StageError) as err:
        check_fit_memory(12_495_787, 12_495_787, 1000, available=30e9)
    text = str(err.value)
    assert "400 GB" in text and "12,495,787 of 12,495,787" in text
    assert "subsample_n" in text and "full-scale" in text


def test_run_hdbscan_checks_the_fitted_count(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    features = rng.normal(size=(3000, 3)).astype(np.float32)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   100 kB\n")   # 102 kB; the fits need 960 kB and 160 kB
    monkeypatch.setattr(memory, "MEMINFO", str(meminfo))
    with pytest.raises(StageError, match="3,000 of 3,000"):
        run_hdbscan(features, hdbscan_cfg(min_cluster_size=50, min_samples=10),
                    device="cpu")
    with pytest.raises(StageError, match="500 of 3,000"):
        run_hdbscan(features, hdbscan_cfg(min_cluster_size=50, min_samples=10,
                                          subsample_n=500), device="cpu")
