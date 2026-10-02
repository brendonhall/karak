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


class _FakeDeviceArray:
    """Stands in for cupy.ndarray: the module name is what accel checks."""

    __module__ = "cupy"

    def __init__(self, host):
        self.host = np.asarray(host)
        self.shape, self.dtype, self.nbytes = self.host.shape, self.host.dtype, self.host.nbytes

    def get(self):
        return self.host


def test_is_device_array_by_module_name():
    assert not accel.is_device_array(np.zeros(3))
    assert accel.is_device_array(_FakeDeviceArray(np.zeros(3)))
    assert not accel.is_device_array(None)
    assert not accel.is_device_array([1, 2])


def test_device_of():
    assert accel.device_of(np.zeros(3)) == "cpu"
    assert accel.device_of(_FakeDeviceArray(np.zeros(3))) == "cuda"


def test_xp_is_numpy_for_host_arrays():
    assert accel.xp(np.zeros(3)) is np


def test_xp_for_a_device_array_uses_get_array_module(monkeypatch):
    fake_module = object()
    monkeypatch.setattr(accel, "get_array_module", lambda device: fake_module)
    assert accel.xp(_FakeDeviceArray(np.zeros(3))) is fake_module


def test_to_host_brings_a_device_array_back():
    arr = _FakeDeviceArray(np.arange(3))
    np.testing.assert_array_equal(accel.to_host(arr), np.arange(3))


def test_device_memory_info_none_without_cuda(monkeypatch):
    monkeypatch.setattr(accel, "cuda_available", lambda: False)
    assert accel.device_memory_info() is None
    accel.free_device_memory()   # no-op, no error


@pytest.mark.skipif(not accel.cuda_available(), reason="no CUDA")
def test_device_memory_info_on_a_gpu():
    free, total = accel.device_memory_info()
    assert 0 < free <= total
