"""Live Rich dashboard for `karak run` (the default on a terminal).

The executor and stages send events through the Reporter protocol; this
class keeps a small model of the run and renders it with Rich ``Live``.
When the run ends the last frame stays on screen as the run summary.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from karak.cli.memory import MemorySampler
from karak.cli.reporter import short

LOG_LINES = 5
PARAM_ROWS = 8


@dataclass
class _Step:
    node_id: str
    stage: str
    status: str = "pending"  # pending|running|done|cached|failed|interrupted
    started: float | None = None
    seconds: float | None = None
    recipe_hash: str = ""
    progress: tuple | None = None  # (done, total, msg)
    params: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    error: str = ""


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def _gb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e9:.1f} GB"


class DashboardReporter:
    """Event-driven model of a run, drawn by Rich ``Live``.

    ``Live`` calls ``render()`` from its refresh thread while events arrive
    on the main thread, so every state change and every render holds
    ``_lock``. ``_stop()`` never holds it while joining the refresh thread.
    """

    def __init__(self, console: Console | None = None, *, live: bool = True,
                 sampler=None):
        self.console = console or Console()
        self.sampler = sampler or MemorySampler()
        self._use_live = live
        self._live: Live | None = None
        self._stopped = False
        self.info = None
        self.steps: dict[str, _Step] = {}
        self.logs: deque = deque(maxlen=LOG_LINES)
        self.current: str | None = None
        self.status = "running"  # running|done|failed|interrupted
        self.total_seconds: float | None = None
        self._lock = threading.RLock()

    # -- reporter protocol --------------------------------------------------

    def run_started(self, info) -> None:
        with self._lock:
            self.info = info
            self.steps = {nid: _Step(nid, stage) for nid, stage in info.nodes}
        self.sampler.start()
        if self._use_live:
            self._live = Live(
                console=self.console, get_renderable=self.render,
                refresh_per_second=4, transient=False,
            )
            self._live.start()

    def node_params(self, node_id: str, params) -> None:
        with self._lock:
            self._step(node_id).params = list(params)
            self.current = node_id

    def node_cache(self, node_id: str, recipe_hash: str, cached: bool,
                   cache_dir: str) -> None:
        with self._lock:
            step = self._step(node_id)
            step.recipe_hash = recipe_hash
            if cached:
                step.status = "cached"

    def node_started(self, node_id: str, label: str) -> None:
        with self._lock:
            step = self._step(node_id)
            step.status = "running"
            step.started = time.monotonic()
            self.current = node_id

    def progress(self, node_id: str, done: int, total: int, msg: str = "") -> None:
        with self._lock:
            self._step(node_id).progress = (done, total, msg)

    def log(self, level: str, msg: str) -> None:
        with self._lock:
            self.logs.append((level, msg))

    def node_outputs(self, node_id: str, summaries: dict) -> None:
        with self._lock:
            self._step(node_id).outputs = dict(summaries)

    def node_finished(self, node_id: str, seconds: float, cached: bool) -> None:
        with self._lock:
            step = self._step(node_id)
            step.seconds = seconds
            step.status = "cached" if cached else "done"
            step.progress = None
            if self.current == node_id:
                self.current = None

    def node_failed(self, node_id: str, message: str) -> None:
        with self._lock:
            step = self._step(node_id)
            step.status = "failed"
            step.error = message
            self.status = "failed"
        self._stop()

    def run_finished(self, summary: dict, seconds: float) -> None:
        with self._lock:
            self.total_seconds = seconds
            self.status = "done"
        self._stop()

    def close(self, status: str | None = None) -> None:
        """Stop the display; ``status`` is "interrupted" or "failed" when the
        run ended without ``run_finished`` or ``node_failed``."""
        with self._lock:
            if status in ("interrupted", "failed") and self.status == "running":
                self.status = status
                for step in self.steps.values():
                    if step.status == "running":
                        step.status = status
        self._stop()

    # -- internals ----------------------------------------------------------

    def _step(self, node_id: str) -> _Step:
        return self.steps.setdefault(node_id, _Step(node_id, "?"))

    def _stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self.sampler.stop()
        if self._live is not None:
            self._live.update(self.render(), refresh=True)
            self._live.stop()
            self._live = None

    # -- rendering ----------------------------------------------------------

    def render(self) -> Panel:
        with self._lock:
            return self._render()

    def _render(self) -> Panel:
        parts = [self._header(), self._steps_table()]
        outputs = self._outputs()
        if outputs is not None:
            parts.append(outputs)
        if self.status == "running":
            step = self.steps.get(self.current) if self.current else None
            if step is not None and step.params:
                parts.append(self._params_panel(step))
            if step is not None and step.progress:
                parts.append(self._progress_line(step))
            if self.logs:
                parts.append(self._log_panel())
        else:
            parts.append(self._footer())
        title = "karak run"
        if self.info is not None:
            title += f"  {self.info.flow}  karak {self.info.version}"
        return Panel(Group(*parts), title=title, title_align="left")

    def _memory_text(self) -> str:
        info = self.info
        label = "mem" if info is None or info.workers in (None, 1) else "mem (main process)"
        if self.sampler.current is None:
            return f"{label} peak {_gb(self.sampler.peak)}"
        return f"{label} {_gb(self.sampler.current)} (peak {_gb(self.sampler.peak)})"

    def _header(self) -> Table:
        grid = Table.grid(padding=(0, 3))
        grid.add_column()
        grid.add_column()
        if self.info is not None:
            info = self.info
            workers = "serial" if info.workers is None else str(info.workers)
            grid.add_row(f"input {short(info.input_path or '-', 70)}",
                         f"out {short(info.out_base, 40)}")
            grid.add_row(
                f"workers {workers}   device {info.device}   "
                f"cache {'on' if info.cache else 'off'}",
                self._memory_text(),
            )
        return grid

    def _status_text(self, step: _Step) -> Text:
        if step.status == "running":
            label = "▶ running"
            if step.progress:
                label = f"▶ {step.progress[0]}/{step.progress[1]}"
            return Text(label, style="cyan")
        if step.status == "cached":
            return Text(f"cached {step.recipe_hash[:8]}", style="green")
        style = {"done": "green", "failed": "bold red",
                 "interrupted": "yellow", "pending": "dim"}.get(step.status, "")
        return Text(step.status, style=style)

    def _time_text(self, step: _Step) -> str:
        if step.status == "running" and step.started is not None:
            return _clock(time.monotonic() - step.started)
        if step.seconds is not None:
            return f"{step.seconds:.1f}s"
        return ""

    def _steps_table(self) -> Table:
        table = Table(box=None, pad_edge=False, header_style="bold")
        for name in ("step", "stage", "status"):
            table.add_column(name)
        table.add_column("time", justify="right")
        for step in self.steps.values():
            table.add_row(step.node_id, step.stage, self._status_text(step),
                          self._time_text(step))
        return table

    def _outputs(self) -> Table | None:
        rows = [(f"{s.node_id}.{port}", text)
                for s in self.steps.values() for port, text in s.outputs.items()]
        if not rows:
            return None
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        for name, text in rows:
            grid.add_row(name, f"→ {text}")
        return grid

    def _params_panel(self, step: _Step) -> Panel:
        changed = [p for p in step.params if not p.is_default]
        defaults = [p for p in step.params if p.is_default]
        shown_defaults = defaults[: max(PARAM_ROWS - len(changed), 0)]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        for p in changed:
            grid.add_row(p.name, Text(short(p.value)))
        for p in shown_defaults:
            grid.add_row(p.name, Text(short(p.value), style="dim"))
        hidden = len(defaults) - len(shown_defaults)
        if hidden:
            grid.add_row("", Text(f"({hidden} more at defaults)", style="dim"))
        return Panel(grid, title=f"params: {step.node_id}", title_align="left")

    def _progress_line(self, step: _Step) -> Table:
        done, total, msg = step.progress
        grid = Table.grid(padding=(0, 2))
        grid.add_row(
            ProgressBar(total=max(total, 1), completed=done, width=50),
            f"{done}/{total}",
            short(msg, 30),
        )
        return grid

    def _log_panel(self) -> Panel:
        styles = {"warning": "yellow", "error": "red"}
        lines = [
            Text(msg, style=styles.get(level, "dim"), no_wrap=True,
                 overflow="ellipsis")
            for level, msg in self.logs
        ]
        return Panel(Group(*lines), title="log", title_align="left")

    def _footer(self) -> Group:
        lines = []
        if self.status == "done":
            ran = [s for s in self.steps.values() if s.status in ("done", "cached")]
            cached = sum(1 for s in ran if s.status == "cached")
            lines.append(Text(
                f"finished in {self.total_seconds:.1f}s · "
                f"{len(ran) - cached} ran, {cached} cached · "
                f"peak memory {_gb(self.sampler.peak)}",
                style="green",
            ))
        elif self.status == "failed":
            for step in self.steps.values():
                if step.status == "failed":
                    lines.append(Text(f"failed at {step.node_id}: {step.error}",
                                      style="bold red"))
        elif self.status == "interrupted":
            lines.append(Text("interrupted", style="yellow"))
        if self.info is not None:
            if getattr(self.info, "record", ""):
                lines.append(Text(f"record {self.info.record}", style="dim"))
            lines.append(Text(f"cache {self.info.work_dir}/cache", style="dim"))
        return Group(*lines)
