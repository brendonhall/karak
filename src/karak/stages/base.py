"""The stage contract: Param / Port / Stage.

A stage is one pipeline operation. It declares a stable ``id``, typed
parameters (``PARAMS``), and named input/output ports (``INPUTS`` /
``OUTPUTS``). ``run()`` is coerce params -> validate inputs -> ``apply()``;
subclasses override only ``apply(inputs, params) -> {port_name: payload}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from karak.errors import StageError

__all__ = ["StageError", "Param", "Port", "Stage"]


_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


@dataclass(frozen=True)
class Param:
    name: str
    type: str                       # "float" | "int" | "bool" | "enum" | "str"
    default: Any
    label: str = ""
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple | None = None
    unit: str | None = None

    def coerce(self, value: Any) -> Any:
        """Cast to the declared type and enforce bounds/choices.

        ``None`` (JSON null) is a real value only for params whose template
        value is None; it never stands in for the template value.
        """
        if value is None:
            if self.default is None:
                return None
            raise ValueError(
                f"{self.name}: null not allowed "
                f"(template value {self.default!r})"
            )

        if self.type == "float":
            coerced: Any = float(value)
        elif self.type == "int":
            if isinstance(value, float):
                if not value.is_integer():
                    raise ValueError(
                        f"{self.name}: expected int, got fractional {value!r}"
                    )
                coerced = int(value)
            else:
                coerced = int(value)
        elif self.type == "bool":
            if isinstance(value, bool):
                coerced = value
            elif isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in _TRUE_STRINGS:
                    coerced = True
                elif lowered in _FALSE_STRINGS:
                    coerced = False
                else:
                    raise ValueError(f"{self.name}: not a bool: {value!r}")
            else:
                coerced = bool(value)
        elif self.type in ("enum", "str"):
            coerced = str(value)
        else:
            raise ValueError(f"{self.name}: unknown param type {self.type!r}")

        if self.choices is not None and coerced not in self.choices:
            raise ValueError(
                f"{self.name}: {coerced!r} not in choices {self.choices!r}"
            )
        if self.min is not None and coerced < self.min:
            raise ValueError(f"{self.name}: {coerced!r} below min {self.min!r}")
        if self.max is not None and coerced > self.max:
            raise ValueError(f"{self.name}: {coerced!r} above max {self.max!r}")
        return coerced


@dataclass(frozen=True)
class Port:
    name: str
    space: Any = None       # optional typed constraint on what flows through
    required: bool = True
    help: str = ""


def _wire_type(payload: Any) -> Any:
    """The type tag a payload carries on the wire (space or state)."""
    tag = getattr(payload, "space", None)
    if tag is None:
        tag = getattr(payload, "state", None)
    return tag


class Stage:
    id: str = ""                    # stable, unique — the "type" a flow node references
    label: str = ""                 # human name (GUI)
    description: str = ""
    INPUTS: list = []               # list[Port]
    OUTPUTS: list = []              # list[Port]
    PARAMS: list = []               # list[Param]

    reporter: Any = None            # injected by the flow executor; duck-typed
    workers: int | None = None       # injected by the flow executor; None = serial
    node_id: str = ""               # injected by the flow executor

    @classmethod
    def schema(cls) -> dict:
        """The GUI palette entry for this stage — pure JSON."""
        def port_json(port: Port) -> dict:
            space = port.space
            return {
                "name": port.name,
                "space": None if space is None else str(space.value),
                "required": port.required,
                "help": port.help,
            }

        def param_json(param: Param) -> dict:
            return {
                "name": param.name,
                "type": param.type,
                "default": param.default,
                "label": param.label,
                "help": param.help,
                "min": param.min,
                "max": param.max,
                "step": param.step,
                "choices": None if param.choices is None else list(param.choices),
                "unit": param.unit,
            }

        return {
            "id": cls.id,
            "label": cls.label,
            "description": cls.description,
            "inputs": [port_json(p) for p in cls.INPUTS],
            "outputs": [port_json(p) for p in cls.OUTPUTS],
            "params": [param_json(p) for p in cls.PARAMS],
        }

    @classmethod
    def source_signature(cls, params: dict) -> str | None:
        """Cache signature for source stages (no inputs).

        Override in stages that read external data so the recipe hash
        invalidates when that data changes. None = params-only hashing.
        """
        return None

    @classmethod
    def recipe_revision(cls, params: dict) -> str | None:
        """Revision tag for recipes whose results changed under unchanged
        params (a fix in the core), so an old cache entry is not reused.

        Return a tag only for the affected params; None (the default)
        leaves the recipe hash exactly as it was. Bump the tag on each
        later change to the same results.
        """
        return None

    @classmethod
    def template(cls) -> dict:
        """Every param with its template value: the starting point for a
        new flow node. Flows must spell out every param; these values are
        never filled in silently during a flow run."""
        return {p.name: p.default for p in cls.PARAMS}

    @classmethod
    def coerce_params(cls, params: dict | None, *,
                      require_complete: bool = False) -> dict:
        """Coerce ``params``; missing ones take their template value unless
        ``require_complete``, in which case they are an error."""
        params = params or {}
        declared = {p.name for p in cls.PARAMS}
        unknown = set(params) - declared
        if unknown:
            raise ValueError(
                f"{cls.id}: unknown param(s) {sorted(unknown)!r}; "
                f"declared: {sorted(declared)!r}"
            )
        missing = [p.name for p in cls.PARAMS if p.name not in params]
        if missing and require_complete:
            raise ValueError(f"{cls.id}: missing param(s) {missing!r}")
        return {
            p.name: p.coerce(params[p.name] if p.name in params else p.default)
            for p in cls.PARAMS
        }

    def check(self, inputs: dict, params: dict) -> list:
        """Return a list of error strings; empty means valid."""
        errors: list[str] = []
        for port in self.INPUTS:
            if port.name not in inputs:
                if port.required:
                    errors.append(f"missing required input {port.name!r}")
                continue
            if port.space is not None:
                wire = _wire_type(inputs[port.name])
                if wire != port.space:
                    errors.append(
                        f"input {port.name!r} expects {port.space!r}, "
                        f"got {wire!r}"
                    )
        return errors

    def run(self, inputs: dict, params: dict | None = None) -> dict:
        coerced = self.coerce_params(params)
        errors = self.check(inputs, coerced)
        if errors:
            raise StageError(f"{self.id}: " + "; ".join(errors))
        return self.apply(inputs, coerced)

    def apply(self, inputs: dict, params: dict) -> dict:
        raise NotImplementedError
