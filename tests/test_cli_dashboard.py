"""The live `karak run` dashboard, rendered from recorded events."""

from __future__ import annotations

import io

from rich.console import Console

from karak.cli.dashboard import DashboardReporter
from karak.flow.events import ParamValue, RunInfo


class FakeSampler:
    current = 2_400_000_000
    peak = 4_100_000_000

    def start(self):
        pass

    def stop(self):
        pass


def _info(workers=None):
    return RunInfo(
        flow="stepwise", input_path="/data/NWA_4587_data", out_base="out/nwa",
        work_dir="out/work", cache=True, workers=workers, device="cpu",
        version="0.2.0", nodes=(("src", "load_elements"),),
    )


def _dashboard():
    return DashboardReporter(Console(width=120, color_system=None),
                             live=False, sampler=FakeSampler())


def _text(reporter):
    console = Console(record=True, width=120, color_system=None)
    console.print(reporter.render())
    return console.export_text()


def _params():
    return (
        [ParamValue("colormap", "tima:jet", True),
         ParamValue("header_trim_px", 100, False)]
        + [ParamValue(f"p{i}", i, True) for i in range(10)]
    )


def test_running_view_shows_context_params_progress_and_log():
    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", _params())
    d.node_cache("src", "abcdef1234567890", False, "out/work/cache")
    d.node_started("src", "Load elements")
    d.progress("src", 14, 20, "Si")
    d.log("info", "Loaded element 'Si'")
    text = _text(d)
    for expected in [
        "stepwise", "karak 0.2.0", "/data/NWA_4587_data", "out/nwa",
        "device cpu", "cache on", "mem 2.4 GB (peak 4.1 GB)",
        "load_elements", "14/20", "header_trim_px", "Si",
        "Loaded element 'Si'", "(4 more at defaults)",
    ]:
        assert expected in text, expected
    assert text.index("header_trim_px") < text.index("colormap")


def test_memory_label_with_workers():
    d = _dashboard()
    d.run_started(_info(workers=16))
    assert "mem (main process)" in _text(d)


def test_finished_view_shows_outputs_and_summary():
    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", _params())
    d.node_cache("src", "abcdef1234567890", False, "out/work/cache")
    d.node_started("src", "Load elements")
    d.node_outputs("src", {"cube": "ElementCube 6525×3990×19 float32 1.98 GB space=raw"})
    d.node_finished("src", 58.2, False)
    d.run_finished({"src": {"cached": False, "seconds": 58.2}}, 58.4)
    text = _text(d)
    for expected in [
        "src.cube", "ElementCube 6525×3990×19", "58.2s",
        "finished in 58.4s", "1 ran, 0 cached", "peak memory 4.1 GB",
        "out/work/cache",
    ]:
        assert expected in text, expected
    assert "params:" not in text


def test_cached_step_shows_short_hash():
    d = _dashboard()
    d.run_started(_info())
    d.node_cache("src", "abcdef1234567890", True, "out/work/cache")
    d.node_outputs("src", {"cube": "ElementCube 2×2×1 float32 16 B space=raw"})
    d.node_finished("src", 0.1, True)
    d.run_finished({"src": {"cached": True, "seconds": 0.1}}, 0.2)
    assert "cached abcdef12" in _text(d)


def test_failure_view():
    d = _dashboard()
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.node_failed("src", "No files matching '*.png' in /nowhere")
    text = _text(d)
    assert "failed" in text
    assert "No files matching '*.png' in /nowhere" in text


def test_interrupt_marks_running_step():
    d = _dashboard()
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.close("interrupted")
    assert "interrupted" in _text(d)


def test_long_param_values_are_truncated():
    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", [ParamValue("flow", "{" + "a" * 500 + "}", False)])
    d.node_started("src", "Load elements")
    assert "a" * 100 not in _text(d)


def test_close_is_idempotent():
    d = _dashboard()
    d.run_started(_info())
    d.close("interrupted")
    d.close()
    d.close("interrupted")
    assert d.status == "interrupted"


def test_live_mode_leaves_final_summary_on_screen():
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=True, color_system=None)
    d = DashboardReporter(console, sampler=FakeSampler())
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.node_outputs("src", {"cube": "ElementCube 2×2×1 float32 16 B space=raw"})
    d.node_finished("src", 0.5, False)
    d.run_finished({"src": {"cached": False, "seconds": 0.5}}, 0.6)
    assert "finished in 0.6s" in buf.getvalue()


def test_render_is_safe_while_the_main_thread_updates_state():
    import threading

    d = _dashboard()
    d.run_started(_info())
    d.node_params("src", _params())
    d.node_started("src", "Load elements")
    errors = []
    stop = threading.Event()

    def refresher():
        while not stop.is_set():
            try:
                d.render()
            except Exception as exc:  # the Live refresh thread would die here
                errors.append(exc)

    thread = threading.Thread(target=refresher)
    thread.start()
    try:
        for i in range(20_000):
            d.log("info", f"line {i}")
            d.progress("src", i % 20 + 1, 20, "Si")
            if i % 50 == 0:
                d.node_finished("src", 1.0, False)
                d.node_started("src", "Load elements")
    finally:
        stop.set()
        thread.join()
    assert errors == []


def test_close_failed_marks_running_step_failed():
    d = _dashboard()
    d.run_started(_info())
    d.node_started("src", "Load elements")
    d.close("failed")
    assert d.status == "failed"
    assert d.steps["src"].status == "failed"
    assert "running" not in _text(d)


def test_finished_view_shows_the_run_record():
    from dataclasses import replace

    d = _dashboard()
    d.run_started(replace(_info(), record="out/nwa/runs/2026-09-29T14-05-12Z"))
    d.node_started("src", "Load elements")
    d.node_finished("src", 1.0, False)
    d.run_finished({"src": {"cached": False, "seconds": 1.0}}, 1.1)
    assert "record out/nwa/runs/2026-09-29T14-05-12Z" in _text(d)
