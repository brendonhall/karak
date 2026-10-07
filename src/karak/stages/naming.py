"""Name-phases stage: attach researcher-assigned names to base phases."""

from __future__ import annotations

import numpy as np

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_names
from karak.stages.payloads import LabelState
from karak.stages.registry import register


@register
class NamePhasesStage(Stage):
    id = "name_phases"
    label = "Name phases"
    description = (
        "Attach mineral names to phase labels. Names travel with the labels "
        "to the split stages, the fingerprints, the QC figures and the HDF5 "
        "export (clusters/mineral_names)."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="labels to name"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="the same labels with names attached"),
    ]
    PARAMS = [
        Param("names", "str", "", "Names",
              "Entries 'label: name' separated by ';', e.g. "
              "'0: Ilmenite (FeTiO₃); 1: Silica polymorph (SiO₂)'"),
        Param("note", "str", "", "Note",
              "How the names were decided (fingerprints, TIMA, references)"),
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        try:
            parse_names(params["names"], "names")
        except ValueError as exc:
            return [str(exc)]
        return []

    def apply(self, inputs: dict, params: dict) -> dict:
        labels = inputs["labels"]
        names = parse_names(params["names"], "names")
        present = set(np.unique(labels.labels).tolist())
        missing = sorted(k for k in names if k not in present)
        if missing:
            raise StageError(
                "name_phases: " + ", ".join(f"label {m}" for m in missing)
                + " not in the labels; check the names against this run's phases"
            )
        return {"labels": labels.replace(names={**labels.names, **names})}
