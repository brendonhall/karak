"""Progress/event seam between the executor and any UI.

Stages and the executor emit through a Reporter; Rich lives only in the CLI
front-end. The protocol is duck-typed — any object with these methods works.
The executor calls every method added after ``node_started``,
``node_finished``, ``progress`` and ``log`` only if the reporter defines it,
so older reporters keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunInfo:
    """Run context, sent once before the first node."""

    flow: str
    input_path: str
    out_base: str
    work_dir: str
    cache: bool
    workers: int | None
    device: str
    version: str
    nodes: tuple  # ((node_id, stage_type), ...) in execution order


@dataclass(frozen=True)
class ParamValue:
    """One resolved node parameter; ``is_default`` compares to the stage default."""

    name: str
    value: object
    is_default: bool


class Reporter(Protocol):
    def run_started(self, info: RunInfo) -> None: ...

    def node_params(self, node_id: str, params: list[ParamValue]) -> None: ...

    def node_cache(
        self, node_id: str, recipe_hash: str, cached: bool, cache_dir: str
    ) -> None: ...

    def node_started(self, node_id: str, label: str) -> None: ...

    def progress(
        self, node_id: str, done: int, total: int, msg: str = ""
    ) -> None: ...

    def log(self, level: str, msg: str) -> None: ...

    def node_outputs(self, node_id: str, summaries: dict[str, str]) -> None: ...

    def node_finished(
        self, node_id: str, seconds: float, cached: bool
    ) -> None: ...

    def node_failed(self, node_id: str, message: str) -> None: ...

    def run_finished(self, summary: dict, seconds: float) -> None: ...


class NullReporter:
    def run_started(self, info: RunInfo) -> None:
        pass

    def node_params(self, node_id: str, params: list[ParamValue]) -> None:
        pass

    def node_cache(
        self, node_id: str, recipe_hash: str, cached: bool, cache_dir: str
    ) -> None:
        pass

    def node_started(self, node_id: str, label: str) -> None:
        pass

    def progress(
        self, node_id: str, done: int, total: int, msg: str = ""
    ) -> None:
        pass

    def log(self, level: str, msg: str) -> None:
        pass

    def node_outputs(self, node_id: str, summaries: dict[str, str]) -> None:
        pass

    def node_finished(
        self, node_id: str, seconds: float, cached: bool
    ) -> None:
        pass

    def node_failed(self, node_id: str, message: str) -> None:
        pass

    def run_finished(self, summary: dict, seconds: float) -> None:
        pass
