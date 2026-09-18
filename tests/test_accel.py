"""Tests for karak.accel: worker resolution and CUDA access helpers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from karak import accel
from karak.stages.base import StageError


def test_resolve_workers_none_is_serial():
    assert accel.resolve_workers(None) == 1


def test_resolve_workers_zero_is_all_cores():
    assert accel.resolve_workers(0) == (os.cpu_count() or 1)


def test_resolve_workers_positive_passthrough():
    assert accel.resolve_workers(3) == 3


def test_resolve_workers_negative_raises():
    with pytest.raises(ValueError):
        accel.resolve_workers(-1)


def test_get_array_module_cpu_is_numpy():
    assert accel.get_array_module("cpu") is np


def test_get_array_module_unknown_device_raises():
    with pytest.raises(StageError):
        accel.get_array_module("tpu")


def test_get_array_module_cuda_without_gpu_raises(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    with pytest.raises(StageError, match="karak\\[cuda\\]"):
        accel.get_array_module("cuda")


def test_to_device_cpu_is_identity():
    arr = np.arange(4)
    assert accel.to_device(arr, "cpu") is arr


def test_to_numpy_passthrough():
    arr = np.arange(4)
    assert accel.to_numpy(arr) is arr


def test_cuda_available_is_bool():
    assert isinstance(accel.cuda_available(), bool)


def test_gpu_name_none_without_gpu(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    assert accel.gpu_name() is None
