"""Rich terminal reporter — the only place Rich touches the pipeline."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape


def short(value, width: int = 60) -> str:
    """``str(value)`` cut to ``width`` characters with a trailing ellipsis."""
    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


class RichReporter:
    """Line-based output: `karak run --plain`, or any non-terminal stdout."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()
        self._hashes: dict[str, str] = {}
        self._closed = False

    def run_started(self, info) -> None:
        workers = "serial" if info.workers is None else str(info.workers)
        self.console.print(
            f"[bold]karak run[/bold]  flow={escape(info.flow)}  karak {info.version}"
        )
        self.console.print(f"  input   {escape(info.input_path or '-')}")
        self.console.print(
            f"  out     {escape(info.out_base)}   work {escape(info.work_dir)}"
        )
        self.console.print(
            f"  workers {workers}   device {info.device}   "
            f"cache {'on' if info.cache else 'off'}"
        )

    def node_params(self, node_id: str, params) -> None:
        changed = [p for p in params if not p.is_default]
        for p in changed:
            self.console.print(
                f"  {node_id}.{p.name} = {escape(short(p.value))}"
            )
        n_default = len(params) - len(changed)
        if n_default:
            self.console.print(
                f"  {node_id}: {n_default} params at defaults", style="dim"
            )

    def node_cache(self, node_id: str, recipe_hash: str, cached: bool,
                   cache_dir: str) -> None:
        self._hashes[node_id] = recipe_hash

    def node_started(self, node_id: str, label: str) -> None:
        self.console.print(f"[cyan]>[/cyan] {node_id}: {escape(label)}")

    def progress(self, node_id: str, done: int, total: int, msg: str = "") -> None:
        detail = f" - {msg}" if msg else ""
        self.console.print(f"  {node_id}: {done}/{total}{escape(detail)}")

    def log(self, level: str, msg: str) -> None:
        style = {"warning": "yellow", "error": "red"}.get(level, "dim")
        self.console.print(escape(msg), style=style)

    def node_outputs(self, node_id: str, summaries: dict) -> None:
        for port, text in summaries.items():
            self.console.print(f"  {node_id}.{port} -> {escape(text)}")

    def node_finished(self, node_id: str, seconds: float, cached: bool) -> None:
        if cached:
            short_hash = self._hashes.get(node_id, "")[:8]
            self.console.print(f"[green]#[/green] {node_id}: cached ({short_hash})")
        else:
            self.console.print(
                f"[green]#[/green] {node_id}: done in {seconds:.1f}s"
            )

    def node_failed(self, node_id: str, message: str) -> None:
        self.console.print(
            f"[red]x {node_id}: failed: {escape(message)}[/red]"
        )

    def run_finished(self, summary: dict, seconds: float) -> None:
        ran = [n for n, e in summary.items() if not e.get("skipped")]
        cached = sum(1 for n in ran if summary[n].get("cached"))
        self.console.print(
            f"run finished in {seconds:.1f}s "
            f"({len(ran) - cached} ran, {cached} cached)"
        )

    def close(self, status: str | None = None) -> None:
        if self._closed:
            return
        if status == "interrupted":
            self.console.print("interrupted", style="yellow")
        self._closed = status is not None


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
