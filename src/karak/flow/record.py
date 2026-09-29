"""Run records: each run keeps the complete flow it executed, and its outcome.

``{out}/runs/<UTC time>/`` holds

- ``flow.json``: the complete flow exactly as executed (after ``--set`` and
  ``--device``), with run tokens unresolved, so it can be run again;
- ``run.json``: status, times, argv, token values, runtime settings,
  overrides, software versions (karak, git commit, libraries, host), and
  per node the resolved params, recipe hash, status, time, and output files
  with their summaries.

``run.json`` is written when the run starts and rewritten as nodes finish,
so a failed or interrupted run leaves a record too. ``{out}/runs/latest``
points at the newest run (a symlink, or a ``LATEST`` text file where
symlinks are unavailable).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from karak.flow.graph import Graph
from karak.provenance import get_software_versions, git_info, host_info, karak_version

RECORD_VERSION = 1


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def runs_dir(out_base: str) -> Path:
    return Path(out_base) / "runs"


def latest_run_dir(out_base: str) -> Path | None:
    """The newest run record directory for an ``--out`` base, if any."""
    runs = runs_dir(out_base)
    link = runs / "latest"
    if link.is_symlink() or link.is_dir():
        return link.resolve()
    marker = runs / "LATEST"
    if marker.is_file():
        return runs / marker.read_text().strip()
    return None


class RunRecord:
    """Writes one run's record; the executor calls ``node`` and ``finish``."""

    def __init__(self, out_base: str, flow: Graph, *, argv: list[str],
                 source: str, tokens: dict, settings: dict, overrides: dict):
        self.out_base = out_base
        self.flow = flow
        self.path: Path | None = None
        self._flow_json = json.dumps(flow.to_json())
        self._started = time.monotonic()
        host = host_info()
        self.data = {
            "record_version": RECORD_VERSION,
            "status": "running",
            "error": None,
            "started": None,
            "finished": None,
            "seconds": None,
            "flow": {"name": flow.name, "source": source, "file": "flow.json"},
            "argv": list(argv),
            "cwd": os.getcwd(),
            "tokens": dict(tokens),
            "settings": dict(settings),
            "overrides": dict(overrides),
            "karak": {"version": karak_version(), "git": git_info()},
            "python": host["python"],
            "platform": host["platform"],
            "host": host["host"],
            "libraries": get_software_versions(),
            "nodes": {n.id: {"type": n.type, "status": "pending"}
                      for n in flow.nodes},
        }

    def _new_dir(self) -> Path:
        base = runs_dir(self.out_base)
        stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
        for n in range(1, 1000):
            path = base / (stamp if n == 1 else f"{stamp}-{n}")
            try:
                path.mkdir(parents=True)
                return path
            except FileExistsError:
                continue
        raise RuntimeError(f"no free run directory under {base}")

    def _point_latest(self) -> None:
        runs = self.path.parent
        link = runs / "latest"
        tmp = runs / f".latest-{os.getpid()}"
        try:
            if tmp.is_symlink():
                tmp.unlink()
            tmp.symlink_to(self.path.name)
            os.replace(tmp, link)
        except OSError:
            (runs / "LATEST").write_text(self.path.name + "\n")

    def _save(self) -> None:
        if self.path is not None:
            _write_json(self.path / "run.json", self.data)

    def start(self) -> Path:
        self.path = self._new_dir()
        _write_json(self.path / "flow.json", self.flow.to_json())
        self.data["started"] = _now()
        self._started = time.monotonic()
        self._save()
        self._point_latest()
        return self.path

    def node(self, node_id: str, **fields) -> None:
        """Merge ``fields`` into a node's entry and rewrite ``run.json``."""
        params = fields.get("params")
        if params is not None:
            # the export sink's flow_json is this run's flow.json; don't repeat it
            fields["params"] = {
                k: ("{flow}" if v == self._flow_json else v)
                for k, v in params.items()
            }
        self.data["nodes"].setdefault(node_id, {}).update(fields)
        self._save()

    def finish(self, status: str, error: str | None = None) -> None:
        if status == "interrupted":
            for entry in self.data["nodes"].values():
                if entry.get("status") == "running":
                    entry["status"] = "interrupted"
        self.data.update(
            status=status,
            error=error,
            finished=_now(),
            seconds=round(time.monotonic() - self._started, 3),
        )
        self._save()
