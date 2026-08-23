"""karak CLI: flow subcommands plus the legacy YAML config path.

New style:
    karak run (FLOW.json | --builtin global|tiled|tiled-rare) --input DIR --out BASE
    karak validate (FLOW.json | --builtin NAME)
    karak schema

Legacy shim (converted to a flow and executed by the same engine):
    karak -c config.yaml [--test-mode] [--no-qc] [--clean] [--emit-flow] [-v]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

# Legacy --test-mode: 4 elements at 4x downsample for fast validation runs
TEST_ELEMENTS = ["Fe-K", "Ca", "Mg", "Si"]
TEST_DOWNSAMPLE = 4

_FLOW_COMMANDS = {"run", "validate", "schema"}


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karak",
        description=(
            "SEM-EDS mineral phase mapping pipeline. "
            "Subcommands: run, validate, schema (see `karak run --help`), "
            "or the legacy -c config.yaml mode."
        ),
    )
    parser.add_argument(
        "-c", "--config", default="config.yaml",
        help="Path to the legacy YAML config (default: config.yaml)",
    )
    parser.add_argument(
        "--from-stage", default=None,
        choices=["load", "mask", "denoise", "normalize", "cluster"],
        help="Deprecated: per-node caching supersedes stage resume",
    )
    parser.add_argument("--no-qc", action="store_true",
                        help="Skip QC figure generation")
    parser.add_argument("--test-mode", action="store_true",
                        help="Fast validation: 4 elements, 4x downsample")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the HDF5 output and flow cache first")
    parser.add_argument("--emit-flow", action="store_true",
                        help="Print the equivalent flow JSON and exit")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _legacy_main(argv: list[str]) -> int:
    from karak.config import load_config
    from karak.flow.__main__ import QC_STAGE_TYPES
    from karak.flow.builtins import flow_from_config, override_params
    from karak.flow.executor import run as run_flow

    args = _legacy_parser().parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.from_stage:
        print(
            "warning: --from-stage is deprecated; the flow cache resumes "
            "automatically from whatever changed",
            file=sys.stderr,
        )

    cfg = load_config(args.config)
    graph = flow_from_config(cfg)

    if args.test_mode:
        graph = override_params(graph, {
            "src.include_elements": ",".join(TEST_ELEMENTS),
            "src.downsample_factor": TEST_DOWNSAMPLE,
        })

    if args.emit_flow:
        print(json.dumps(graph.to_json(), indent=2))
        return 0

    h5_path = Path(cfg.hdf5_output)
    work_dir = h5_path.parent / "work"

    if args.clean:
        h5_path.unlink(missing_ok=True)
        shutil.rmtree(work_dir / "cache", ignore_errors=True)

    from karak.cli.reporter import RichReporter

    summary = run_flow(
        graph,
        input_path=cfg.input_dir,
        out_base=str(h5_path.with_suffix("")),
        work_dir=str(work_dir),
        cache=not args.no_cache,
        reporter=RichReporter(),
        skip_types=QC_STAGE_TYPES if args.no_qc else frozenset(),
    )
    cached = sum(1 for entry in summary.values() if entry.get("cached"))
    print(f"{len(summary)} nodes: {cached} cached, "
          f"{len(summary) - cached} executed")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in _FLOW_COMMANDS:
        from karak.cli.reporter import RichReporter
        from karak.flow.__main__ import main as flow_main

        reporter = RichReporter() if argv[0] == "run" else None
        return flow_main(argv, reporter=reporter)
    return _legacy_main(argv)


if __name__ == "__main__":
    sys.exit(main())
