"""CLI for the flow layer: ``python -m karak.flow {run|validate|schema}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from karak.flow.builtins import apply_device, builtin_flow, builtin_names, override_params
from karak.flow.complete import canonicalize, complete_graph
from karak.flow.graph import Graph
from karak.flow.validate import validate

QC_STAGE_TYPES = frozenset({
    "qc_mask", "qc_denoise", "qc_normalize", "qc_scree", "qc_phase_map",
    "qc_cluster_summary", "qc_tiled", "qc_fingerprints", "qc_named_phase_map",
})


def _positive_gb(text: str) -> float:
    """argparse type for --ram-budget/--gpu-budget: a number of GB above zero."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be above 0 GB, got {text}")
    return value


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


def _write_flow(graph: Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.to_json(), indent=2) + "\n")


def _flow_command(args) -> int:
    if args.flow_command == "init":
        out = Path(args.output)
        if out.exists() and not args.force:
            raise SystemExit(f"error: {out} exists (use --force to overwrite)")
        _write_flow(builtin_flow(args.builtin), out)
        print(f"wrote {out} ({args.builtin}, every param listed)")
        return 0

    # complete
    source = Path(args.flow)
    graph, added = complete_graph(Graph.from_json(json.loads(source.read_text())))
    out = Path(args.output) if args.output else source
    _write_flow(graph, out)
    if not added:
        print(f"{out}: already complete")
    for node_id, names in added.items():
        print(f"{node_id}: added {len(names)} params from stage templates: "
              + ", ".join(names))
    return 0


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
    run_parser.add_argument(
        "--cache-compression", choices=["lzf", "gzip", "none"], default="lzf",
        help="HDF5 filter for cache files (lzf: fast; gzip: small; none: "
             "fastest, largest). Reads accept any.",
    )
    run_parser.add_argument(
        "--ram-budget", type=_positive_gb, default=None, metavar="GB",
        help="Host memory to hold stage outputs between steps before "
             "spilling them to the cache (default: half of available "
             "memory). Writes waiting for the disk are bounded by the "
             "same amount; a step waits when the queue is full.",
    )
    run_parser.add_argument(
        "--gpu-budget", type=_positive_gb, default=None, metavar="GB",
        help="Device memory to hold stage outputs between GPU steps before "
             "moving them to host memory (default: 80%% of free device "
             "memory at start). Used only when a node runs on cuda.",
    )
    run_parser.add_argument("--no-qc", action="store_true",
                            help="Skip QC figure sinks")
    run_parser.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="CPU worker processes for stages that parallelize "
             "(0 = all cores; default: serial). Never changes results.",
    )
    run_parser.add_argument(
        "--set", action="append", default=[], metavar="NODE.PARAM=VALUE",
        help="Override a node parameter (repeatable)",
    )
    run_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Apply this device to every node that declares a device "
             "param (cuda needs the karak[cuda] extra).",
    )
    run_parser.add_argument(
        "--plain", action="store_true",
        help="Line-based output instead of the live dashboard "
             "(automatic when stdout is not a terminal).",
    )

    flow_parser = sub.add_parser(
        "flow", help="Create or complete flow JSON files")
    flow_sub = flow_parser.add_subparsers(dest="flow_command", required=True)
    init_parser = flow_sub.add_parser(
        "init", help="Write a builtin flow, with every param, to a file")
    init_parser.add_argument("--builtin", required=True, choices=builtin_names())
    init_parser.add_argument("-o", "--output", required=True,
                             help="Flow JSON file to write")
    init_parser.add_argument("--force", action="store_true",
                             help="Overwrite an existing file")
    complete_parser = flow_sub.add_parser(
        "complete",
        help="Fill missing params from stage templates (upgrades v1 flows)")
    complete_parser.add_argument("flow", help="Flow JSON file")
    complete_parser.add_argument(
        "-o", "--output", default=None,
        help="Write here instead of updating the file in place")

    validate_parser = sub.add_parser("validate", help="Validate a flow")
    _add_flow_args(validate_parser)

    sub.add_parser("schema", help="Print the stage palette as JSON")
    sub.add_parser("bench", help="Time flow nodes across configurations",
                    add_help=False)
    return parser


def main(argv: list[str] | None = None, reporter=None) -> int:
    raw = sys.argv[2:] if argv is None else argv[1:]
    if (argv is not None and argv and argv[0] == "bench") or (
        argv is None and len(sys.argv) > 1 and sys.argv[1] == "bench"
    ):
        from karak.cli.bench import bench_main  # lazy: Rich stays in cli/

        return bench_main(raw)

    args = build_parser().parse_args(argv)

    if args.command == "schema":
        from karak.stages import list_stages

        print(json.dumps(list_stages(), indent=2))
        return 0

    if args.command == "flow":
        return _flow_command(args)

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
    from karak.flow.record import RunRecord

    overrides = _parse_set(args.set) if args.set else {}
    try:
        if overrides:
            graph = override_params(graph, overrides)
        if args.device:
            graph = apply_device(graph, args.device)
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"error: --set/--device: {exc}") from None
    graph = canonicalize(graph)
    if any(n.params.get("device") == "cuda" for n in graph.nodes):
        import karak.accel as accel

        if not accel.cuda_available():
            raise SystemExit(
                "error: device='cuda' requested but no usable GPU stack "
                "was found. Install with: pip install 'karak[cuda]'"
            )
    work_dir = args.work or str(Path(args.out).parent / "work")
    record = RunRecord(
        args.out, graph,
        argv=list(sys.argv[1:] if argv is None else argv),
        source=args.flow or f"builtin:{args.builtin}",
        tokens={"input": args.input, "out": args.out, "work": work_dir},
        settings={"workers": args.workers, "cache": not args.no_cache,
                  "no_qc": args.no_qc, "device": args.device,
                  "cache_compression": args.cache_compression,
                  "ram_budget_gb": args.ram_budget,
                  "gpu_budget_gb": args.gpu_budget},
        overrides=overrides,
    )
    summary = run_flow(
        graph,
        input_path=args.input,
        out_base=args.out,
        work_dir=work_dir,
        cache=not args.no_cache,
        cache_compression=args.cache_compression,
        ram_budget=(None if args.ram_budget is None
                    else int(args.ram_budget * 2**30)),
        gpu_budget=(None if args.gpu_budget is None
                    else int(args.gpu_budget * 2**30)),
        reporter=reporter,
        workers=args.workers,
        skip_types=QC_STAGE_TYPES if args.no_qc else frozenset(),
        record=record,
    )
    if reporter is None:
        print(f"run record: {record.path}")
        cached = sum(1 for entry in summary.values() if entry.get("cached"))
        print(f"{len(summary)} nodes: {cached} cached, "
              f"{len(summary) - cached} executed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
