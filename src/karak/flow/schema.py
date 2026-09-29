"""JSON Schema for flow files, generated from the stage registry.

A visual composer (or any editor) can validate a flow against
``docs/flow.schema.json`` while it is being built: one node variant per
stage type, every declared param required, with types, choices, bounds, and
nullability; the template value is the JSON Schema ``default`` annotation,
used to prefill a new node, never applied at run time. ``flow.validate``
stays the authoritative check (graph structure, port types, cycles).

Regenerate the shipped copy with ``uv run python -m karak.flow.schema``;
a test pins it to the registry.
"""

from __future__ import annotations

import json
from pathlib import Path

from karak.flow.complete import FLOW_VERSION
from karak.stages import registry

_JSON_TYPES = {"float": "number", "int": "integer", "bool": "boolean",
               "enum": "string", "str": "string"}

SCHEMA_PATH = (Path(__file__).resolve().parent.parent.parent.parent
               / "docs" / "flow.schema.json")


def _param_schema(param: dict) -> dict:
    json_type = _JSON_TYPES[param["type"]]
    nullable = param["default"] is None
    schema: dict = {"type": [json_type, "null"] if nullable else json_type}
    if param["label"]:
        schema["title"] = param["label"]
    description = param["help"] or ""
    if param["unit"]:
        description = f"{description} [{param['unit']}]".strip()
    if description:
        schema["description"] = description
    if param["choices"] is not None:
        schema["enum"] = list(param["choices"]) + ([None] if nullable else [])
    if param["min"] is not None:
        schema["minimum"] = param["min"]
    if param["max"] is not None:
        schema["maximum"] = param["max"]
    schema["default"] = param["default"]
    return schema


def flow_json_schema() -> dict:
    stages = sorted(registry.list_stages(), key=lambda s: s["id"])
    defs: dict = {
        "endpoint": {
            "type": "object",
            "required": ["node", "port"],
            "properties": {"node": {"type": "string"},
                           "port": {"type": "string"}},
            "additionalProperties": False,
        },
        "edge": {
            "type": "object",
            "required": ["id", "from", "to"],
            "properties": {
                "id": {"type": "string"},
                "from": {"$ref": "#/$defs/endpoint",
                         "description": "an output port"},
                "to": {"$ref": "#/$defs/endpoint",
                       "description": "an input port"},
            },
            "additionalProperties": False,
        },
        "ui": {
            "type": "object",
            "description": "Editor-only metadata (e.g. canvas position); "
                           "ignored when the flow runs.",
        },
        "node": {"oneOf": [{"$ref": f"#/$defs/node_{s['id']}"} for s in stages]},
    }
    for stage in stages:
        sid = stage["id"]
        defs[f"node_{sid}"] = {
            "title": stage["label"] or sid,
            "description": stage["description"],
            "type": "object",
            "required": ["id", "type", "params"],
            "properties": {
                "id": {"type": "string"},
                "type": {"const": sid},
                "params": {"$ref": f"#/$defs/params_{sid}"},
                "ui": {"$ref": "#/$defs/ui"},
            },
            "additionalProperties": False,
        }
        defs[f"params_{sid}"] = {
            "type": "object",
            "required": [p["name"] for p in stage["params"]],
            "properties": {p["name"]: _param_schema(p) for p in stage["params"]},
            "additionalProperties": False,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "karak flow",
        "description": (
            f"A karak pipeline (flow format version {FLOW_VERSION}): stage "
            "nodes with every parameter listed, connected by edges from "
            "output ports to input ports. Generated from the stage registry."
        ),
        "type": "object",
        "required": ["version", "name", "nodes", "edges"],
        "properties": {
            "version": {"const": FLOW_VERSION},
            "name": {"type": "string"},
            "nodes": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            "edges": {"type": "array", "items": {"$ref": "#/$defs/edge"}},
        },
        "additionalProperties": False,
        "$defs": defs,
    }


def render() -> str:
    return json.dumps(flow_json_schema(), indent=2) + "\n"


def main() -> None:
    SCHEMA_PATH.write_text(render())
    print(f"wrote {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
