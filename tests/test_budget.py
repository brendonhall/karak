"""RAM budget helpers."""

from __future__ import annotations

from karak.flow import budget


def test_available_host_memory_reads_meminfo(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\nMemAvailable:   20000000 kB\n")
    monkeypatch.setattr(budget, "MEMINFO", str(meminfo))
    assert budget.available_host_memory() == 20000000 * 1024


def test_default_budget_is_half_of_available(tmp_path, monkeypatch):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   20000000 kB\n")
    monkeypatch.setattr(budget, "MEMINFO", str(meminfo))
    assert budget.default_ram_budget() == 20000000 * 1024 // 2


def test_default_budget_without_proc_meminfo(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "MEMINFO", str(tmp_path / "missing"))
    assert budget.available_host_memory() is None
    assert budget.default_ram_budget() == 8 * 2**30
