"""The flow-file JSON Schema: what a visual composer builds against."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from karak.flow.builtins import builtin_flow, builtin_names
from karak.flow.schema import flow_json_schema, render
from karak.stages import registry

DOC = Path(__file__).resolve().parent.parent / "docs" / "flow.schema.json"


def test_shipped_schema_matches_the_registry():
    assert DOC.read_text() == render(), (
        "docs/flow.schema.json is stale; regenerate with "
        "`uv run python -m karak.flow.schema`"
    )


def test_schema_is_valid_draft_2020_12():
    jsonschema.Draft202012Validator.check_schema(flow_json_schema())


def test_every_stage_has_a_node_variant_requiring_every_param():
    schema = flow_json_schema()
    for entry in registry.list_stages():
        params = schema["$defs"][f"params_{entry['id']}"]
        assert params["required"] == [p["name"] for p in entry["params"]]
        assert params["additionalProperties"] is False


@pytest.mark.parametrize("name", builtin_names())
def test_builtin_flows_conform(name):
    jsonschema.validate(builtin_flow(name).to_json(), flow_json_schema())


def _stepwise():
    return builtin_flow("stepwise").to_json()


def test_a_missing_param_violates_the_schema():
    flow = _stepwise()
    del flow["nodes"][0]["params"]["colormap"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(flow, flow_json_schema())


def test_null_only_where_the_template_is_null():
    flow = _stepwise()
    flow["nodes"][0]["params"]["filename_pattern"] = None     # template None
    jsonschema.validate(flow, flow_json_schema())
    flow["nodes"][0]["params"]["downsample_factor"] = None    # template 2
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(flow, flow_json_schema())


def test_bounds_choices_and_ui_are_expressed():
    schema = flow_json_schema()
    load = schema["$defs"]["params_load_elements"]["properties"]
    assert load["downsample_factor"]["minimum"] == 1
    assert load["downsample_factor"]["default"] == 2       # the template value
    denoise = schema["$defs"]["params_denoise"]["properties"]
    assert set(denoise["method"]["enum"]) == {"bilateral", "anisotropic_diffusion"}
    flow = _stepwise()
    flow["nodes"][0]["ui"] = {"x": 120, "y": 40}
    jsonschema.validate(flow, schema)


def test_version_1_flows_violate_the_schema():
    flow = _stepwise()
    flow["version"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(flow, flow_json_schema())
