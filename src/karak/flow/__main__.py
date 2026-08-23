"""CLI for the flow layer: ``python -m karak.flow {run|validate|schema}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from karak.flow.builtins import builtin_flow, builtin_names, override_params
from karak.flow.graph import Graph
from karak.flow.validate import validate

QC_STAGE_TYPES = frozenset({
    "qc_mask", "qc_denoise", "qc_normalize", "qc_scree", "qc_phase_map",
    "qc_cluster_summary", "qc_tiled", "qc_fingerprints", "qc_named_phase_map",
})


def _load_graph(args) -> Graph:
    if args.builtin:
        return builtin_flow(args.builtin)
    if not args.flow:
        raise SystemExit("error: provide a FLOW.json path or --builtin NAME")
    data = json.loads(Path(args.flow).read_text())
    return Graph.from_json(data)


def _parse_set(values: list[str]) -> dict:
    overrides = {}
    for item in values:
        spec, _, raw = item.partition("=")
        if not _ or "." not in spec:
            raise SystemExit(
                f"error: --set expects NODE.PARAM=VALUE, got {item!r}"
            )
        try:
            overrides[spec] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[spec] = raw
    return overrides


def _add_flow_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("flow", nargs="?", help="Path to a flow JSON file")
    parser.add_argument(
        "--builtin", choices=builtin_names(),
        help="Use a builtin flow instead of a JSON file",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karak flow",
        description="Validate and run karak flow graphs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Execute a flow")
    _add_flow_args(run_parser)
    run_parser.add_argument("--input", default="", help="Input data directory")
    run_parser.add_argument(
        "--out", default="output/karak", help="Output basename ({out} token)"
    )
    run_parser.add_argument(
        "--work", default=None,
        help="Work/cache directory (default: <out dir>/work)",
    )
    run_parser.add_argument("--no-cache", action="store_true")
    run_parser.add_argument("--no-qc", action="store_true",
                            help="Skip QC figure sinks")
    run_parser.add_argument(
        "--set", action="append", default=[], metavar="NODE.PARAM=VALUE",
        help="Override a node parameter (repeatable)",
    )

    validate_parser = sub.add_parser("validate", help="Validate a flow")
    _add_flow_args(validate_parser)

    sub.add_parser("schema", help="Print the stage palette as JSON")
    return parser


def main(argv: list[str] | None = None, reporter=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "schema":
        from karak.stages import list_stages

        print(json.dumps(list_stages(), indent=2))
        return 0

    graph = _load_graph(args)

    if args.command == "validate":
        issues = validate(graph)
        errors = [i for i in issues if i.level == "error"]
        for issue in issues:
            print(f"{issue.level:8s} [{issue.where}] {issue.message}")
        print(f"{len(errors)} errors, {len(issues) - len(errors)} warnings")
        return 1 if errors else 0

    # run
    from karak.flow.executor import run as run_flow

    if args.set:
        graph = override_params(graph, _parse_set(args.set))
    work_dir = args.work or str(Path(args.out).parent / "work")
    summary = run_flow(
        graph,
        input_path=args.input,
        out_base=args.out,
        work_dir=work_dir,
        cache=not args.no_cache,
        reporter=reporter,
        skip_types=QC_STAGE_TYPES if args.no_qc else frozenset(),
    )
    cached = sum(1 for entry in summary.values() if entry.get("cached"))
    print(f"{len(summary)} nodes: {cached} cached, "
          f"{len(summary) - cached} executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
