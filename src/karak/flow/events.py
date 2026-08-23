"""Progress/event seam between the executor and any UI.

Stages and the executor emit through a Reporter; Rich lives only in the CLI
front-end. The protocol is duck-typed — any object with these methods works.
"""

from __future__ import annotations

from typing import Protocol


class Reporter(Protocol):
    def node_started(self, node_id: str, label: str) -> None: ...

    def node_finished(
        self, node_id: str, seconds: float, cached: bool
    ) -> None: ...

    def progress(
        self, node_id: str, done: int, total: int, msg: str = ""
    ) -> None: ...

    def log(self, level: str, msg: str) -> None: ...


class NullReporter:
    def node_started(self, node_id: str, label: str) -> None:
        pass

    def node_finished(
        self, node_id: str, seconds: float, cached: bool
    ) -> None:
        pass

    def progress(
        self, node_id: str, done: int, total: int, msg: str = ""
    ) -> None:
        pass

    def log(self, level: str, msg: str) -> None:
        pass
