"""Self-describing pipeline stages.

Importing this package auto-imports every stage submodule so ``@register``
fires and the registry is complete. Drop a new stage file in this directory
and it appears in the palette with no other wiring.
"""

from karak.stages.base import Param, Port, Stage, StageError  # noqa: F401
from karak.stages.registry import get, list_stages, register  # noqa: F401


def _autoload() -> None:
    import importlib
    import pkgutil

    skip = {"base", "registry", "payloads"}     # infra, not stages
    for module in pkgutil.iter_modules(__path__):
        if module.name not in skip:
            importlib.import_module(f"{__name__}.{module.name}")


_autoload()
