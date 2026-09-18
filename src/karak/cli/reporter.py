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


def render_bench_table(result: dict) -> None:
    """Rich table: nodes as rows, configs as columns, speedup vs first."""
    from rich.console import Console
    from rich.table import Table

    configs = result["configs"]
    meta = result.get("meta") or {}
    title = f"karak bench  ({meta.get('hostname', '?')}, {meta.get('cpus', '?')} cpus"
    if meta.get("gpu"):
        title += f", {meta['gpu']}"
    title += ")"

    table = Table(title=title)
    table.add_column("node")
    for cfg in configs:
        table.add_column(cfg["label"], justify="right")
    if len(configs) > 1:
        table.add_column("speedup", justify="right")

    node_ids = list(configs[0]["nodes"])
    baseline = configs[0]["nodes"]
    for node_id in node_ids:
        row = [node_id]
        for cfg in configs:
            row.append(f"{cfg['nodes'].get(node_id, float('nan')):.2f}s")
        if len(configs) > 1:
            last = configs[-1]["nodes"].get(node_id)
            row.append(f"{baseline[node_id] / last:.1f}x" if last else "-")
        table.add_row(*row)

    totals = ["total"] + [f"{cfg['total']:.2f}s" for cfg in configs]
    if len(configs) > 1 and configs[-1]["total"]:
        totals.append(f"{configs[0]['total'] / configs[-1]['total']:.1f}x")
    table.add_section()
    table.add_row(*totals)
    Console().print(table)
