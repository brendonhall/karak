"""Render the stage palette as a Markdown reference document.

The output is committed as ``docs/stage_reference.md``; a drift test pins
the file to the live registry. Regenerate after adding or changing a stage:

    uv run python -m karak.stages.reference
"""

from __future__ import annotations

from karak.stages import list_stages

_HEADER = """\
# Stage reference

Every karak processing step is a stage: one class with named input/output
ports and typed, bounded parameters. This file is generated from the live
stage registry (`karak schema` prints the same contract as JSON).

Do not edit by hand. Regenerate with:

```bash
uv run python -m karak.stages.reference
```

Port type tags: cubes carry a space (`raw`, `denoised`, `normalized`) and
label arrays a state (`raw` = may contain -1, `cleaned` = fully labeled).
The flow validator rejects connections whose tags disagree. A dash means
the port accepts its payload type without a tag constraint. Stages with no
outputs are sinks: they run for their side effect (a file or a figure) and
are never cached.
"""


def _format_default(value) -> str:
    if value is None:
        return "`None`"
    if value == "":
        return '`""`'
    return f"`{value}`"


def _bounds(param: dict) -> str:
    parts = []
    if param["min"] is not None or param["max"] is not None:
        low = param["min"] if param["min"] is not None else ""
        high = param["max"] if param["max"] is not None else ""
        parts.append(f"{low}..{high}")
    if param["choices"]:
        parts.append(" \\| ".join(str(c) for c in param["choices"]))
    if param["unit"]:
        parts.append(param["unit"])
    return ", ".join(parts) if parts else "-"


def _port_rows(ports: list[dict]) -> list[str]:
    rows = []
    for port in ports:
        tag = f"`{port['space']}`" if port["space"] else "-"
        required = "yes" if port["required"] else "no"
        rows.append(
            f"| `{port['name']}` | {tag} | {required} | {port['help'] or ''} |"
        )
    return rows


def render_markdown() -> str:
    lines = [_HEADER]
    for schema in sorted(list_stages(), key=lambda s: s["id"]):
        lines.append(f"## `{schema['id']}` — {schema['label']}")
        lines.append("")
        lines.append(schema["description"])
        lines.append("")

        if schema["inputs"]:
            lines.append("**Inputs**")
            lines.append("")
            lines.append("| port | type tag | required | notes |")
            lines.append("|------|----------|----------|-------|")
            lines.extend(_port_rows(schema["inputs"]))
        else:
            lines.append("**Inputs**: none (source stage)")
        lines.append("")

        if schema["outputs"]:
            lines.append("**Outputs**")
            lines.append("")
            lines.append("| port | type tag | notes |")
            lines.append("|------|----------|-------|")
            for port in schema["outputs"]:
                tag = f"`{port['space']}`" if port["space"] else "-"
                lines.append(f"| `{port['name']}` | {tag} | {port['help'] or ''} |")
        else:
            lines.append("**Outputs**: none (sink)")
        lines.append("")

        if schema["params"]:
            lines.append("**Parameters**")
            lines.append("")
            lines.append("| name | type | default | bounds / choices | help |")
            lines.append("|------|------|---------|------------------|------|")
            for param in schema["params"]:
                help_text = param["help"] or param["label"] or ""
                lines.append(
                    f"| `{param['name']}` | {param['type']} | "
                    f"{_format_default(param['default'])} | {_bounds(param)} | "
                    f"{help_text} |"
                )
        else:
            lines.append("**Parameters**: none")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    from pathlib import Path

    target = Path(__file__).parents[3] / "docs" / "stage_reference.md"
    target.write_text(render_markdown())
    print(f"wrote {target}")
