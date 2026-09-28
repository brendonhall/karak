"""Line-based output of `karak run --plain`."""

from __future__ import annotations

from rich.console import Console

from karak.cli.reporter import RichReporter, short
from karak.flow.events import ParamValue, RunInfo

INFO = RunInfo(
    flow="stepwise", input_path="/data/NWA", out_base="out/nwa",
    work_dir="out/work", cache=True, workers=None, device="cpu",
    version="0.2.0", nodes=(("src", "load_elements"),),
)


def _reporter():
    console = Console(record=True, width=120, color_system=None)
    return RichReporter(console), console


def test_plain_run_prints_context_params_outputs_and_summary():
    reporter, console = _reporter()
    reporter.run_started(INFO)
    reporter.node_params("src", [
        ParamValue("header_trim_px", 100, False),
        ParamValue("colormap", "tima:jet", True),
    ])
    reporter.node_cache("src", "abcdef1234567890", False, "out/work/cache")
    reporter.node_started("src", "Load elements")
    reporter.progress("src", 3, 20, "Si")
    reporter.node_outputs("src", {"cube": "ElementCube 2×2×1 float32 16 B space=raw"})
    reporter.node_finished("src", 1.5, False)
    reporter.run_finished({"src": {"cached": False, "seconds": 1.5}}, 1.6)
    text = console.export_text()
    for expected in [
        "karak run", "flow=stepwise", "karak 0.2.0", "/data/NWA", "out/nwa",
        "workers serial", "device cpu", "cache on",
        "src.header_trim_px = 100", "src: 1 params at defaults",
        "src: 3/20 - Si", "src.cube -> ElementCube 2×2×1",
        "src: done in 1.5s", "run finished in 1.6s (1 ran, 0 cached)",
    ]:
        assert expected in text
    assert "colormap" not in text


def test_plain_cached_step_shows_hash():
    reporter, console = _reporter()
    reporter.run_started(INFO)
    reporter.node_cache("src", "abcdef1234567890", True, "out/work/cache")
    reporter.node_finished("src", 0.1, True)
    assert "src: cached (abcdef12)" in console.export_text()


def test_plain_failure_escapes_markup():
    reporter, console = _reporter()
    reporter.node_failed("src", "bad value [red]x[/red]")
    assert "src: failed: bad value [red]x[/red]" in console.export_text()


def test_long_param_values_are_truncated():
    assert short("x" * 100, width=10) == "xxxxxxxxx…"
    assert short("abc", width=10) == "abc"
    reporter, console = _reporter()
    reporter.node_params("exp", [ParamValue("flow_json", "{" + "a" * 500 + "}", False)])
    line = [l for l in console.export_text().splitlines() if "flow_json" in l][0]
    assert len(line) < 100


def test_close_is_idempotent():
    reporter, console = _reporter()
    reporter.close("interrupted")
    reporter.close("interrupted")
    reporter.close()
    assert console.export_text().count("interrupted") == 1


def test_plain_default_console_does_not_wrap_long_lines(capsys):
    reporter = RichReporter()
    long_line = "Loaded element 'Si' (inverted via tima:jet): " + "x" * 150
    reporter.log("info", long_line)
    assert long_line in capsys.readouterr().out
