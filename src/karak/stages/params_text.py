"""Parsers for the small text grammars inside string params.

Params are scalar, so lists and rules travel as strings (as
``exclude_elements`` does). These parsers turn them into data and raise
``ValueError`` messages that start with the param name, so
``Stage.check_params`` and ``karak validate`` can report them.
"""

from __future__ import annotations

from karak.clustering.features import parse_feature  # noqa: F401  (moved to the core)

_OPS = ("<=", ">=", "<", ">")


def parse_csv(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_int_list(text: str | None, name: str) -> list[int]:
    values = []
    for item in parse_csv(text):
        try:
            values.append(int(item))
        except ValueError:
            raise ValueError(f"{name}: {item!r} is not an integer label") from None
    return values


def parse_names(text: str | None, name: str) -> dict[int, str]:
    """``"0: Ilmenite; 1: Silica"`` -> {0: "Ilmenite", 1: "Silica"}."""
    names: dict[int, str] = {}
    for entry in (text or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        label, sep, value = entry.partition(":")
        if not sep or not value.strip():
            raise ValueError(f"{name}: expected 'label: name', got {entry!r}")
        try:
            key = int(label)
        except ValueError:
            raise ValueError(f"{name}: {label.strip()!r} is not an integer label") from None
        if key in names:
            raise ValueError(f"{name}: label {key} given twice")
        names[key] = value.strip()
    return names


def parse_rule(text: str | None, name: str) -> list[tuple[str, str, float]]:
    """``"Fe-K > 0.6 & Ca < 0.10"`` -> [("Fe-K", ">", 0.6), ("Ca", "<", 0.1)]."""
    if not text or not text.strip():
        raise ValueError(f"{name}: empty rule")
    rules = []
    for clause in text.split("&"):
        clause = clause.strip()
        op = next((o for o in _OPS if o in clause), None)
        if op is None:
            raise ValueError(f"{name}: {clause!r} needs one of {_OPS}")
        channel, _, value = clause.partition(op)
        channel = channel.strip()
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"{name}: {value.strip()!r} is not a number") from None
        if not channel:
            raise ValueError(f"{name}: {clause!r} has no channel name")
        rules.append((channel, op, number))
    return rules
