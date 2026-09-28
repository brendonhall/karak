"""Process memory sampling for the dashboard."""

from __future__ import annotations

import sys

import pytest

from karak.cli.memory import MemorySampler, current_rss, peak_rss


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc is Linux-only")
def test_current_rss_is_positive_on_linux():
    assert current_rss() > 0


def test_peak_is_at_least_current():
    current = current_rss() or 0
    assert peak_rss() >= current > -1


def test_sampler_start_stop_is_idempotent():
    sampler = MemorySampler(interval=0.01)
    sampler.start()
    sampler.start()
    sampler.stop()
    sampler.stop()
    assert sampler.peak > 0
    assert sampler._thread is None
