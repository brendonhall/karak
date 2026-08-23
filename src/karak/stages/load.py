"""Source stage: load element maps + BSE from disk into an ElementCube."""

from __future__ import annotations

from karak.config import DownsampleConfig, LoaderConfig
from karak.io.loaders import build_compositional_cube, load_element_maps
from karak.stages.base import Param, Port, Stage
from karak.stages.payloads import BseImage, ElementCube, Space
from karak.stages.registry import register


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@register
class LoadElementsStage(Stage):
    id = "load_elements"
    label = "Load element maps"
    description = (
        "Load false-color element map images, invert the colormap to scalar "
        "[0, 1] intensities, downsample/trim, and stack into a cube."
    )
    INPUTS: list = []
    OUTPUTS = [
        Port("cube", space=Space.RAW, help="(H, W, C) raw element cube"),
        Port("bse", help="(H, W) BSE grayscale image"),
    ]
    PARAMS = [
        Param("input_dir", "str", "{input}", "Input directory",
              "Directory containing element map files"),
        Param("file_glob", "str", "*.png", "File glob",
              "Glob pattern (relative to input_dir) for element map files"),
        Param("filename_pattern", "str", None, "Filename pattern",
              "Pattern with {element} placeholder; None = legacy TIMA heuristic"),
        Param("bse_filename", "str", None, "BSE filename",
              "Exact BSE file name when it does not match the glob"),
        Param("colormap", "str", "cmap:jet", "Colormap",
              "'cmap:NAME' or 'lut:PATH' inversion spec"),
        Param("exclude_elements", "str", "Fe-L", "Exclude elements",
              "Comma-separated element names to skip"),
        Param("include_elements", "str", None, "Include elements",
              "Comma-separated allowlist; None = load all"),
        Param("bse_channel", "str", "SEM", "BSE channel",
              "Element name of the BSE/SEM channel"),
        Param("header_trim_px", "int", 100, "Header trim", min=0, unit="px"),
        Param("bottom_trim_px", "int", 0, "Bottom trim", min=0, unit="px"),
        Param("left_trim_px", "int", 0, "Left trim", min=0, unit="px"),
        Param("right_trim_px", "int", 0, "Right trim", min=0, unit="px"),
        Param("downsample_factor", "int", 2, "Downsample factor", min=1),
    ]

    @classmethod
    def source_signature(cls, params: dict) -> str | None:
        """(relpath, size, mtime_ns) of every matched input file.

        Cheap change detection: the cache invalidates when files are added,
        removed, or modified, without hashing gigabytes of image data.
        """
        import glob
        import os

        input_dir = params["input_dir"]
        paths = sorted(glob.glob(os.path.join(input_dir, params["file_glob"])))
        if params["bse_filename"]:
            bse_path = os.path.join(input_dir, params["bse_filename"])
            if bse_path not in paths and os.path.exists(bse_path):
                paths.append(bse_path)
        entries = []
        for path in paths:
            stat = os.stat(path)
            entries.append(
                f"{os.path.relpath(path, input_dir)}:"
                f"{stat.st_size}:{stat.st_mtime_ns}"
            )
        return ";".join(entries)

    def apply(self, inputs: dict, params: dict) -> dict:
        downsample = DownsampleConfig(
            header_trim_px=params["header_trim_px"],
            bottom_trim_px=params["bottom_trim_px"],
            left_trim_px=params["left_trim_px"],
            right_trim_px=params["right_trim_px"],
            downsample_factor=params["downsample_factor"],
        )
        loader = LoaderConfig(
            file_glob=params["file_glob"],
            filename_pattern=params["filename_pattern"],
            bse_filename=params["bse_filename"],
            colormap=params["colormap"],
        )
        include = _split_csv(params["include_elements"]) or None
        elements, bse, names = load_element_maps(
            params["input_dir"],
            downsample,
            exclude_elements=_split_csv(params["exclude_elements"]),
            bse_channel=params["bse_channel"],
            include_elements=include,
            loader_config=loader,
        )
        cube = build_compositional_cube(elements, names)
        return {
            "cube": ElementCube(
                pixels=cube,
                element_names=tuple(names),
                space=Space.RAW,
                downsample_factor=params["downsample_factor"],
                header_trim_px=params["header_trim_px"],
                left_trim_px=params["left_trim_px"],
            ),
            "bse": BseImage(pixels=bse),
        }
