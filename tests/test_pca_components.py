"""Tests for the auto component-count heuristic (moved out of the runner)."""

from __future__ import annotations

import numpy as np

from karak.clustering.pca import auto_n_components


def test_picks_first_component_reaching_threshold():
    # cumvar: 0.5, 0.75, 0.87, 0.93, 0.955, 0.97, ...
    evr = np.array([0.5, 0.25, 0.12, 0.06, 0.025, 0.015, 0.01, 0.008,
                    0.007, 0.005])
    assert auto_n_components(evr, variance_threshold=0.95, min_components=1) == 5


def test_enforces_minimum():
    evr = np.array([0.96, 0.02, 0.01, 0.005, 0.003, 0.002])
    assert auto_n_components(evr) == 5  # threshold hit at 1, floor is 5


def test_threshold_never_reached_keeps_all():
    evr = np.array([0.3, 0.3, 0.3])  # cum 0.9 < 0.95
    assert auto_n_components(evr, min_components=1) == 3
