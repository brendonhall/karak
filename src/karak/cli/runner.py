"""Entry point shim.

The orchestration that used to live here moved into the stage and flow
layers (``karak.stages`` / ``karak.flow``); the CLI lives in
``karak.cli.main``. This module only preserves the packaged entry point
``karak = "karak.cli.runner:main"``.
"""

from karak.cli.main import main  # noqa: F401
