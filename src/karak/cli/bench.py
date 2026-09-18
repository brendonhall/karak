"""karak bench: time flow nodes across worker/device configurations.

Runs the flow once per --config with the cache disabled, keeps the
per-node minimum over --repeats, and writes <out>_bench.json plus a
terminal table. --compare renders a table from two result files.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchConfig:
    label: str
    workers: int | None
    device: str | None


def parse_config(spec: str) -> BenchConfig:
    """'baseline' | comma list of workers=N and device=NAME."""
    if spec == "baseline":
        return BenchConfig("baseline", None, None)
    workers: int | None = None
    device: str | None = None
    for part in spec.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise SystemExit(f"error: bench config needs key=value, got {part!r}")
        if key == "workers":
            workers = int(value)
        elif key == "device":
            device = value
        else:
            raise SystemExit(
                f"error: unknown bench config key {key!r} in {spec!r} "
                "(expected workers=N and/or device=NAME)"
            )
    return BenchConfig(spec, workers, device)


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


def _meta(input_path: str) -> dict:
    from karak import accel

    return {
        "hostname": socket.gethostname(),
        "cpus": os.cpu_count() or 1,
        "gpu": accel.gpu_name(),
        "git": _git_revision(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input": input_path,
    }


def run_bench(
    graph,
    *,
    input_path: str,
    out_base: str,
    work_dir: str,
    configs: list[BenchConfig],
    repeats: int,
    include_qc: bool,
) -> dict:
    from karak.flow.__main__ import QC_STAGE_TYPES
    from karak.flow.builtins import apply_device
    from karak.flow.executor import run as run_flow

    skip = frozenset() if include_qc else QC_STAGE_TYPES
    results = []
    for cfg in configs:
        run_graph = apply_device(graph, cfg.device) if cfg.device else graph
        best: dict[str, float] = {}
        for _ in range(repeats):
            summary = run_flow(
                run_graph,
                input_path=input_path,
                out_base=out_base,
                work_dir=work_dir,
                cache=False,
                workers=cfg.workers,
                skip_types=skip,
            )
            for node_id, entry in summary.items():
                if entry.get("skipped"):
                    continue
                seconds = entry["seconds"]
                if node_id not in best or seconds < best[node_id]:
                    best[node_id] = seconds
        results.append({
            "label": cfg.label,
            "workers": cfg.workers,
            "device": cfg.device,
            "nodes": best,
            "total": sum(best.values()),
        })
    return {"meta": _meta(input_path), "configs": results}


def _build_parser() -> argparse.ArgumentParser:
    from karak.flow.__main__ import _add_flow_args

    parser = argparse.ArgumentParser(
        prog="karak bench",
        description="Time each flow node across configurations.",
    )
    _add_flow_args(parser)
    parser.add_argument("--input", default="", help="Input data directory")
    parser.add_argument("--out", default="output/bench",
                        help="Output basename ({out} token)")
    parser.add_argument("--work", default=None,
                        help="Work directory (default: <out dir>/work)")
    parser.add_argument("--config", action="append", default=[],
                        metavar="SPEC",
                        help="'baseline' or comma list of workers=N,"
                             "device=NAME (repeatable)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per config; per-node minimum is kept")
    parser.add_argument("--qc", action="store_true",
                        help="Include QC figure sinks (skipped by default)")
    parser.add_argument("--set", action="append", default=[],
                        metavar="NODE.PARAM=VALUE",
                        help="Override a node parameter (repeatable)")
    parser.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                        help="Render a table from two bench JSON files")
    return parser


def bench_main(argv: list[str] | None = None) -> int:
    from karak.cli.reporter import render_bench_table

    args = _build_parser().parse_args(argv)

    if args.compare:
        merged = {"meta": None, "configs": []}
        for path in args.compare:
            data = json.loads(Path(path).read_text())
            stem = Path(path).stem
            merged["meta"] = merged["meta"] or data["meta"]
            for cfg in data["configs"]:
                merged["configs"].append({**cfg, "label": f"{stem}:{cfg['label']}"})
        render_bench_table(merged)
        return 0

    from karak.flow.__main__ import _load_graph, _parse_set
    from karak.flow.builtins import override_params

    graph = _load_graph(args)
    if args.set:
        graph = override_params(graph, _parse_set(args.set))
    configs = [parse_config(s) for s in (args.config or ["baseline"])]

    if any(c.device == "cuda" for c in configs):
        from karak import accel

        if not accel.cuda_available():
            raise SystemExit(
                "error: a config requests device=cuda but no usable GPU "
                "stack was found. Install with: pip install 'karak[cuda]'"
            )

    work_dir = args.work or str(Path(args.out).parent / "work")
    result = run_bench(
        graph,
        input_path=args.input,
        out_base=args.out,
        work_dir=work_dir,
        configs=configs,
        repeats=args.repeats,
        include_qc=args.qc,
    )
    json_path = Path(f"{args.out}_bench.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {json_path}")
    render_bench_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(bench_main())
