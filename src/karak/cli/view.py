"""karak view: open a run's cached load, mask and denoise outputs in napari.

It opens the outputs listed in the run's latest record
(``{out}/runs/latest/run.json``): the load step's ElementCube (the first
with no upstream cube), the BseImage from the same step, the first MaskSet
(the mask step's), and the denoised ElementCube computed from that cube
(the denoise step's). Without a record it scans the cache, which names
files by recipe hash, reading each file's ``payload_type`` and upstream
recipes to find the newest of each. Elements are image layers, the
denoised elements ``dn: <element>`` layers, and the mineral mask and the
valid mask labels layers. Layers are placed in full-resolution coordinates
(scale = downsample factor, offset = trims) so positions match the
original exports and the napari shapes the valid mask was drawn with.

napari is an optional dependency: ``uv sync --extra view``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

INSTALL_HINT = (
    "napari is not installed. Install the viewer extra with:\n"
    "  uv sync --extra view      (then: uv run --extra view karak view ...)"
)


@dataclass(frozen=True)
class CacheFile:
    path: Path
    recipe: str
    port: str
    payload_type: str
    mtime: float
    upstream: dict = field(default_factory=dict)  # input port -> recipe


@dataclass(frozen=True)
class LayerSpec:
    name: str
    data: np.ndarray
    scale: tuple
    translate: tuple
    visible: bool
    colormap: str = "gray"
    contrast_limits: tuple = (0.0, 1.0)
    kind: str = "image"   # "image" (add_image) or "labels" (add_labels)


@dataclass(frozen=True)
class PickedOutputs:
    """The cache files `karak view` opens: one cube, its BSE, the masks."""

    cube: CacheFile
    bse: CacheFile | None
    masks: CacheFile | None
    denoised: CacheFile | None
    other_cubes: list


def _cache_dir_for(target: Path) -> Path:
    """Map an out base, work dir, or cache dir to the cache directory."""
    if (target / "cache").is_dir():
        return target / "cache"
    if target.is_dir() and any(target.glob("*__*.h5")):
        return target
    # An --out base: `karak run` puts the cache in <out dir>/work/cache
    return target.parent / "work" / "cache"


def _read_entry(path: Path) -> CacheFile | None:
    recipe, _, port = path.stem.partition("__")
    try:
        with h5py.File(path, "r") as fh:
            payload_type = str(fh["payload"].attrs["payload_type"])
            raw = fh["payload"].attrs.get("upstream")
    except (OSError, KeyError):
        return None
    upstream = json.loads(str(raw)) if raw is not None else {}
    return CacheFile(path, recipe, port, payload_type, path.stat().st_mtime,
                     upstream)


def find_cache_files(target: str | Path) -> list[CacheFile]:
    """Cached payloads under an out base, work dir, cache dir, or one file."""
    target = Path(target)
    if target.is_file():
        paths = [target]
    else:
        cache = _cache_dir_for(target)
        paths = sorted(cache.glob("*__*.h5")) if cache.is_dir() else []
    entries = [e for e in (_read_entry(p) for p in paths) if e is not None]
    if not entries:
        raise FileNotFoundError(f"no cached outputs found for {target}")
    return entries


def find_run_outputs(target: str | Path) -> list[CacheFile] | None:
    """Output files of the latest recorded run of an ``--out`` base, in flow
    order; None when there is no record or its files are gone."""
    from karak.flow.record import latest_run_dir

    run_dir = latest_run_dir(str(target))
    if run_dir is None or not (run_dir / "run.json").is_file():
        return None
    data = json.loads((run_dir / "run.json").read_text())
    base = Path(data.get("cwd", "."))
    entries = []
    for node in data.get("nodes", {}).values():
        for output in (node.get("outputs") or {}).values():
            if not output.get("file"):
                continue
            path = Path(output["file"])
            entry = _read_entry(path if path.is_absolute() else base / path)
            if entry is None:
                return None   # stale record (cache cleared): scan instead
            entries.append(entry)
    return entries or None


def pick_outputs(entries: list[CacheFile], newest_first: bool = True) -> PickedOutputs:
    """The load step's ElementCube, the BseImage from the same step, the
    MaskSet and the denoised ElementCube computed from that cube, and the
    other cubes.

    ``newest_first`` picks the most recent files (cache scan); otherwise
    the first in list order (a run record lists outputs in flow order).
    The load step's cube is one with no upstream ``cube`` (see
    ``store_payload``); in a cache scan the masks and the denoised cube
    must name it as their upstream ``cube``, so another run's outputs are
    never overlaid. With a record, the run itself links the masks.
    """
    def newest(found):
        return sorted(found, key=lambda e: e.mtime, reverse=True) if newest_first else found

    cubes = newest([e for e in entries if e.payload_type == "element_cube"])
    if not cubes:
        raise ValueError("no ElementCube in the cache; run the load step first")
    cube = next((e for e in cubes if "cube" not in e.upstream), cubes[0])
    bses = [e for e in entries if e.payload_type == "bse_image"]
    same_step = [e for e in bses if e.recipe == cube.recipe]
    bse = (same_step or sorted(bses, key=lambda e: e.mtime, reverse=True) or [None])[0]
    mask_sets = [e for e in entries if e.payload_type == "mask_set"]
    if newest_first:
        mask_sets = newest(e for e in mask_sets if e.upstream.get("cube") == cube.recipe)
    masks = (mask_sets or [None])[0]
    derived = [e for e in cubes if e.upstream.get("cube") == cube.recipe]
    denoised = (derived or [None])[0]
    others = [e for e in cubes if e is not cube and e is not denoised]
    return PickedOutputs(cube, bse, masks, denoised, others)


def layer_specs(cube, bse, masks=None, denoised=None,
                show=("Fe-K",)) -> list[LayerSpec]:
    """BSE, one layer per element, the denoised cube's elements as
    ``dn: <element>``, then the masks as labels layers, all placed in
    full-resolution pixels. Only ``show`` elements start visible."""
    factor = cube.downsample_factor
    offset = (factor - 1) / 2  # a block's center, in full-resolution pixels
    translate = (
        (cube.header_trim_px // factor) * factor + offset,
        (cube.left_trim_px // factor) * factor + offset,
    )
    scale = (factor, factor)
    names = list(cube.element_names)
    visible = set(show) & set(names) or {names[0]}
    specs = []
    if bse is not None:
        specs.append(LayerSpec("BSE", bse.pixels, scale, translate, True))
    for i, name in enumerate(names):
        specs.append(LayerSpec(name, cube.pixels[..., i], scale, translate,
                               name in visible))
    if denoised is not None:
        for i, name in enumerate(denoised.element_names):
            specs.append(LayerSpec(f"dn: {name}", denoised.pixels[..., i],
                                   scale, translate, name in visible))
    if masks is not None:
        specs.append(LayerSpec("mineral mask", masks.mineral_mask.astype(np.uint8),
                               scale, translate, True, kind="labels"))
        if masks.valid_mask is not None:
            specs.append(LayerSpec("valid mask", masks.valid_mask.astype(np.uint8),
                                   scale, translate, False, kind="labels"))
    return specs


def _napari_available() -> bool:
    return importlib.util.find_spec("napari") is not None


def open_viewer(specs: list[LayerSpec], shapes) -> None:
    import napari

    viewer = napari.Viewer(title="karak view")
    for spec in specs:
        if spec.kind == "labels":
            viewer.add_labels(
                spec.data, name=spec.name, scale=spec.scale,
                translate=spec.translate, visible=spec.visible,
            )
            continue
        viewer.add_image(
            spec.data, name=spec.name, scale=spec.scale,
            translate=spec.translate, visible=spec.visible,
            colormap=spec.colormap, contrast_limits=spec.contrast_limits,
        )
    if shapes:
        viewer.add_shapes(
            [vertices for _, vertices in shapes],
            shape_type=[kind for kind, _ in shapes],
            name="valid mask", edge_color="yellow", face_color="transparent",
            edge_width=20,
        )
    napari.run()


def view_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="karak view",
        description="Open the cached load, mask and denoise outputs of a run in napari.",
    )
    parser.add_argument(
        "path",
        help="--out base of a run, its work dir, a cache dir, or one cached .h5",
    )
    parser.add_argument("--show", default="Fe-K",
                        help="Comma-separated elements visible at start")
    parser.add_argument("--mask", default=None,
                        help="napari shapes CSV (e.g. mask/Valid_mask.csv)")
    args = parser.parse_args(argv)

    if not _napari_available():
        print(INSTALL_HINT, file=sys.stderr)
        return 1

    from karak.flow.cache import load_summary
    from karak.io.masks import read_napari_shapes
    from karak.stages.payloads import payload_from_h5

    from karak.flow.record import latest_run_dir

    recorded = find_run_outputs(args.path)
    if recorded is not None:
        print(f"run record: {latest_run_dir(args.path)}")
        picked = pick_outputs(recorded, newest_first=False)
    else:
        picked = pick_outputs(find_cache_files(args.path))
    for other in picked.other_cubes:
        print(f"other cube: {other.path.name} "
              f"({time.strftime('%Y-%m-%d %H:%M', time.localtime(other.mtime))})")

    def load(entry):
        size_mb = entry.path.stat().st_size / 1e6
        print(f"loading {entry.path.name} ({size_mb:.0f} MB) ...", flush=True)
        with h5py.File(entry.path, "r") as fh:
            payload = payload_from_h5(fh["payload"])
        print("  " + (load_summary(entry.recipe, entry.port, entry.path.parent)
                      or payload.summary()))
        return payload

    cube = load(picked.cube)
    bse = load(picked.bse) if picked.bse is not None else None
    masks = load(picked.masks) if picked.masks is not None else None
    denoised = load(picked.denoised) if picked.denoised is not None else None
    specs = layer_specs(cube, bse, masks, denoised,
                        show=tuple(args.show.split(",")))
    shapes = read_napari_shapes(args.mask) if args.mask else []
    open_viewer(specs, shapes)
    return 0
