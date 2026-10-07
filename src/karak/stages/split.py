"""Split stages: divide one phase by a threshold rule or a GMM.

Each split names its new labels in its own params and appends a record to
``Labels.history``, so the reason for a split sits next to the parameters
that made it, in the flow file and in the exported HDF5.
"""

from __future__ import annotations

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_rule
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register

_NOTE = Param("note", "str", "", "Note", "Why this split: the observation it rests on")


def _history_record(stage: str, parent: int, new_labels: list[int], names: dict,
                    n_pixels: dict, method: str, note: str) -> dict:
    return {"stage": stage, "parent": int(parent),
            "new_labels": [int(x) for x in new_labels],
            "names": {int(k): v for k, v in names.items()},
            "n_pixels": {int(k): int(v) for k, v in n_pixels.items()},
            "method": method, "note": note}


@register
class SplitThresholdStage(Stage):
    id = "split_threshold"
    label = "Split by threshold"
    description = (
        "Move the pixels of one phase that satisfy a rule on denoised "
        "channel values to a new label (e.g. olivine out of a pyroxene "
        "phase with 'Fe-K > 0.6 & Ca < 0.10')."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="labels with the phase to split"),
        Port("cube", space=Space.DENOISED, help="denoised channel values for the rule"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="labels with the new phase; names and history extended"),
    ]
    PARAMS = [
        Param("target_phase", "int", 0, "Target phase", "Label to split", min=0),
        Param("rule", "str", "", "Rule",
              "Comparisons joined by '&': 'Fe-K > 0.6 & Ca < 0.10' "
              "(operators <, <=, >, >= on channel names)"),
        Param("new_name", "str", "", "New name", "Name of the new label"),
        _NOTE,
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        errors = []
        try:
            parse_rule(params["rule"], "rule")
        except ValueError as exc:
            errors.append(str(exc))
        if not params["new_name"].strip():
            errors.append("new_name: the new label needs a name")
        return errors

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.refinement import threshold_split

        labels, cube = inputs["labels"], inputs["cube"]
        rules = parse_rule(params["rule"], "rule")
        try:
            updated, new_label, n = threshold_split(
                labels.labels, cube.pixels, labels.mineral_indices,
                list(cube.element_names), params["target_phase"], rules,
            )
        except ValueError as exc:
            raise StageError(f"split_threshold: {exc}") from exc
        names = dict(labels.names)
        new_labels: list[int] = []
        if new_label >= 0:
            names[new_label] = params["new_name"]
            new_labels = [new_label]
        record = _history_record(
            self.id, params["target_phase"], new_labels,
            {new_label: params["new_name"]} if new_labels else {},
            {new_label: n} if new_labels else {}, params["rule"], params["note"],
        )
        return {"labels": labels.replace(labels=updated, names=names,
                                         history=labels.history + (record,))}
