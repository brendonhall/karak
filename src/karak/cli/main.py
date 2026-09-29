"""karak CLI.

    karak flow init --builtin NAME -o FLOW.json     start a pipeline from a builtin
    karak flow complete FLOW.json [-o OUT]          fill in missing params
    karak run (FLOW.json | --builtin NAME) --input DIR --out BASE
    karak validate (FLOW.json | --builtin NAME)
    karak schema | bench | view

A flow JSON lists every parameter of every node; nothing comes from code.
"""

from __future__ import annotations

import sys

_FLOW_COMMANDS = {"run", "validate", "schema", "flow"}

USAGE = """\
usage: karak COMMAND [...]

A pipeline is a flow JSON file that lists every parameter of every node.

  karak flow init --builtin NAME -o FLOW.json   start from a builtin flow
                                                (global, tiled, tiled-rare, stepwise)
  karak flow complete FLOW.json [-o OUT]        fill missing params from stage templates
  karak run (FLOW.json | --builtin NAME) --input DIR --out BASE [--set NODE.PARAM=VALUE]
  karak validate (FLOW.json | --builtin NAME)   check a flow without running it
  karak schema                                  stage palette (params, ports) as JSON
  karak bench ...                               time nodes across worker/device settings
  karak view BASE                               open a run's outputs in napari

Run `karak COMMAND --help` for the options of a command.
"""


def _run_reporter(plain: bool, isatty: bool):
    """The live dashboard on a terminal; line output with --plain or a pipe."""
    if plain or not isatty:
        from karak.cli.reporter import RichReporter

        return RichReporter()
    from karak.cli.dashboard import DashboardReporter

    return DashboardReporter()


def _run_command(argv: list[str]) -> int:
    from karak.cli.logs import capture_logs
    from karak.flow.__main__ import main as flow_main
    from karak.flow.executor import FlowError

    reporter = _run_reporter("--plain" in argv, sys.stdout.isatty())
    try:
        with capture_logs(reporter):
            return flow_main(argv, reporter=reporter)
    except FlowError as exc:
        reporter.close()
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        reporter.close("interrupted")
        return 130
    except Exception:
        reporter.close("failed")
        raise
    finally:
        reporter.close()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "view":
        from karak.cli.view import view_main

        return view_main(argv[1:])
    if argv and argv[0] == "bench":
        from karak.cli.bench import bench_main

        return bench_main(argv[1:])
    if argv and argv[0] == "run":
        return _run_command(argv)
    if argv and argv[0] in _FLOW_COMMANDS:
        from karak.flow.__main__ import main as flow_main

        return flow_main(argv)
    print(USAGE)
    return 0 if argv[:1] in (["-h"], ["--help"]) else 2


if __name__ == "__main__":
    sys.exit(main())
