"""Split stages: divide one phase by a threshold rule or a GMM.

Each split names its new labels in its own params and appends a record to
``Labels.history``, so the reason for a split sits next to the parameters
that made it, in the flow file and in the exported HDF5.
"""

from __future__ import annotations

import logging

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_csv, parse_feature, parse_rule
from karak.stages.payloads import LabelState, Space
from karak.stages.registry import register

logger = logging.getLogger(__name__)

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


@register
class SplitGmmStage(Stage):
    id = "split_gmm"
    label = "Split by GMM"
    description = (
        "Split one phase with a Gaussian mixture on z-scored features "
        "(denoised channels, BSE, or a ratio A/(A+B)). The largest component "
        "can keep the parent label; new labels are named in order."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="labels with the phase to split"),
        Port("cube", space=Space.DENOISED, help="denoised channels for the features"),
        Port("bse", required=False, help="BSE image; needed when a feature is 'BSE'"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="labels with the new phases; names and history extended"),
    ]
    PARAMS = [
        Param("target_phase", "int", 0, "Target phase", "Label to split", min=0),
        Param("features", "str", "", "Features",
              "Comma list of channel names, 'BSE', or ratios 'A/(A+B)'"),
        Param("n_components", "int", 2, "Components", min=2),
        Param("bse_weight", "float", 1.0, "BSE weight",
              "Multiplier on the z-scored BSE column", min=0.0),
        Param("subsample_n", "int", 500_000, "Subsample N",
              "Max pixels fitted; 0 = all", min=0),
        Param("random_state", "int", 42, "Random seed"),
        Param("keep_parent", "bool", True, "Keep parent",
              "The largest component keeps the parent label"),
        Param("order_by", "str", "", "Order by",
              "Feature whose component means (ascending) order the new "
              "labels; empty = by size, descending"),
        Param("new_names", "str", "", "New names",
              "Comma list, one per new label, in order"),
        _NOTE,
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        errors = []
        features = parse_csv(params["features"])
        if not features:
            errors.append("features: at least one feature is needed")
        for token in features:
            try:
                parse_feature(token)
            except ValueError as exc:
                errors.append(f"features: {exc}")
        order_by = params["order_by"].strip()
        if order_by and order_by not in features:
            errors.append(f"order_by: {order_by!r} is not one of the features")
        expected = params["n_components"] - (1 if params["keep_parent"] else 0)
        names = parse_csv(params["new_names"])
        if len(names) != expected:
            errors.append(f"new_names: {expected} name(s) expected, got {len(names)}")
        return errors

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.refinement import gmm_split

        labels, cube = inputs["labels"], inputs["cube"]
        bse = inputs.get("bse")
        try:
            updated, new_labels, info = gmm_split(
                labels.labels, cube.pixels, None if bse is None else bse.pixels,
                labels.mineral_indices, list(cube.element_names),
                params["target_phase"], parse_csv(params["features"]),
                n_components=params["n_components"], bse_weight=params["bse_weight"],
                subsample_n=params["subsample_n"] or None,
                random_state=params["random_state"], keep_parent=params["keep_parent"],
                order_by=params["order_by"],
            )
        except ValueError as exc:
            raise StageError(f"split_gmm: {exc}") from exc
        given = parse_csv(params["new_names"])
        new_names = {label: given[i] for i, label in enumerate(new_labels) if i < len(given)}
        if len(given) > len(new_labels):
            logger.warning(
                "split_gmm: %d name(s) given for %d new label(s); dropped %s",
                len(given), len(new_labels), given[len(new_labels):],
            )
        names = {**labels.names, **new_names}
        if not params["keep_parent"] and new_labels:
            names.pop(params["target_phase"], None)   # the parent emptied
        record = _history_record(
            self.id, params["target_phase"], new_labels, new_names,
            info.get("n_pixels", {}), info["method"], params["note"],
        )
        if "skipped" in info:
            record["skipped"] = info["skipped"]
        if "component_means" in info:
            record["component_means"] = {int(k): v for k, v in info["component_means"].items()}
        return {"labels": labels.replace(labels=updated, names=names,
                                         history=labels.history + (record,))}
