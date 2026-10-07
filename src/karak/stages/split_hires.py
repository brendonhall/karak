"""Split-hires stage: a GMM at full resolution inside a set of phases."""

from __future__ import annotations

from karak.stages.base import Param, Port, Stage, StageError
from karak.stages.params_text import parse_feature, parse_int_list
from karak.stages.payloads import HiresLabels, LabelState, Space
from karak.stages.registry import register
from karak.stages.split import _NOTE, _history_record, _split_names


@register
class SplitHiresStage(Stage):
    id = "split_hires"
    label = "Split at full resolution"
    description = (
        "Inside a set of phases, fit one Gaussian mixture on a channel or "
        "ratio of a higher-resolution cube and classify every pixel of that "
        "cube; the working labels take the majority of their children. The "
        "high-resolution map is a second output (e.g. exsolution lamellae "
        "in pyroxene from Ca/(Ca+Mg) at 1x)."
    )
    INPUTS = [
        Port("labels", space=LabelState.CLEANED, help="working-resolution labels"),
        Port("cube_hires", space=Space.RAW,
             help="raw cube at a higher resolution (a second load_elements "
                  "node with downsample_factor 1 and include_elements)"),
    ]
    OUTPUTS = [
        Port("labels", space=LabelState.CLEANED,
             help="working labels with the new phases; names and history extended"),
        Port("labels_hires", help="HiresLabels: the new labels at the cube's resolution"),
    ]
    PARAMS = [
        Param("target_phases", "str", "", "Target phases",
              "Comma list of labels that form the region"),
        Param("feature", "str", "", "Feature",
              "One channel of cube_hires or a ratio 'A/(A+B)'"),
        Param("n_components", "int", 2, "Components", min=2),
        Param("subsample_n", "int", 500_000, "Subsample N",
              "Max hires pixels fitted; 0 = all", min=0),
        Param("random_state", "int", 42, "Random seed"),
        Param("new_names", "str", "", "New names",
              "Names separated by ';', one per component, by ascending feature mean"),
        _NOTE,
    ]

    @classmethod
    def check_params(cls, params: dict) -> list[str]:
        errors = []
        try:
            targets = parse_int_list(params["target_phases"], "target_phases")
            if not targets:
                errors.append("target_phases: at least one label is needed")
            for label in targets:
                if label < 0:
                    errors.append(f"target_phases: labels must be >= 0, got {label}")
                    break
        except ValueError as exc:
            errors.append(str(exc))
        try:
            kind, _ = parse_feature(params["feature"])
            if kind == "bse":
                errors.append("feature: BSE is not a channel of cube_hires")
        except ValueError as exc:
            errors.append(f"feature: {exc}")
        n_names = len(_split_names(params["new_names"]))
        if n_names != params["n_components"]:
            errors.append(
                f"new_names: {params['n_components']} names expected, got {n_names}")
        return errors

    def apply(self, inputs: dict, params: dict) -> dict:
        from karak.clustering.hires import hires_split

        labels, cube_hi = inputs["labels"], inputs["cube_hires"].to("cpu")
        targets = parse_int_list(params["target_phases"], "target_phases")
        try:
            updated, image, new_labels, info = hires_split(
                labels.labels, labels.mineral_indices, labels.image_shape,
                cube_hi.pixels, list(cube_hi.element_names), targets, params["feature"],
                n_components=params["n_components"],
                subsample_n=params["subsample_n"] or None,
                random_state=params["random_state"],
            )
        except ValueError as exc:
            raise StageError(f"split_hires: {exc}") from exc
        new_names = dict(zip(new_labels, _split_names(params["new_names"])))
        names = {**labels.names, **new_names}
        for parent in targets:                      # emptied parents lose their name
            if not (updated == parent).any():
                names.pop(parent, None)
        record = _history_record(self.id, targets[0], new_labels, new_names,
                                 info["n_pixels_working"], info["method"], params["note"])
        record.update({"parents": targets, "ratio": info["ratio"],
                       "n_pixels_hires": info["n_pixels"], "n_undefined": info["n_undefined"],
                       "component_means": info["component_means"]})
        return {
            "labels": labels.replace(labels=updated, names=names,
                                     history=labels.history + (record,)),
            "labels_hires": HiresLabels(
                image=image, ratio=info["ratio"], names=new_names,
                downsample_factor=cube_hi.downsample_factor,
                header_trim_px=cube_hi.header_trim_px, left_trim_px=cube_hi.left_trim_px,
            ),
        }
