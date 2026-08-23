"""Rich terminal reporter — the only place Rich touches the pipeline."""

from __future__ import annotations

from rich.console import Console


class RichReporter:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def node_started(self, node_id: str, label: str) -> None:
        self.console.print(f"[cyan]>[/cyan] {node_id}: {label}")

    def node_finished(self, node_id: str, seconds: float, cached: bool) -> None:
        if cached:
            self.console.print(f"[green]#[/green] {node_id}: cached")
        else:
            self.console.print(
                f"[green]#[/green] {node_id}: done in {seconds:.1f}s"
            )

    def progress(self, node_id: str, done: int, total: int, msg: str = "") -> None:
        detail = f" - {msg}" if msg else ""
        self.console.print(f"  {node_id}: {done}/{total}{detail}")

    def log(self, level: str, msg: str) -> None:
        style = {"warning": "yellow", "error": "red"}.get(level, "dim")
        self.console.print(f"[{style}]{msg}[/{style}]")
