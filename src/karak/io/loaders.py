"""Data loading utilities for SEM-EDS element maps.

Element maps are false-color RGB images. Each map is converted to a scalar
[0, 1] intensity by inverting the colormap. The BSE/SEM channel is true
grayscale (R==G==B) and uses standard ``rgb2gray``.

The colormap is configurable per-dataset via ``LoaderConfig.colormap``:
``'cmap:NAME'`` looks up a matplotlib colormap; ``'lut:PATH'`` loads a
(N, 3) uint8 LUT from an .npy file. Default is ``'cmap:jet'`` (TIMA
convention). Filenames are parsed with a configurable
``filename_pattern`` (e.g. ``'NAW 4587-2_S3858-{element}.png'``).

All channels are uniformly downsampled and edge-trimmed.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean

if TYPE_CHECKING:
    from karak.config import DownsampleConfig, LoaderConfig

logger = logging.getLogger(__name__)

# Elements / files to always skip (not element channels)
_NON_ELEMENT_FILES = {"Phases"}

# ---------------------------------------------------------------------------
# Colormap inversion via precomputed (256, 256, 256) lookup table
# ---------------------------------------------------------------------------
# Number of palette samples for matplotlib-cmap-derived LUTs. A custom .npy
# LUT uses however many entries it contains.
_CMAP_PALETTE_N = 4096

# Cache directory for the full RGB→scalar lookup tables (one per palette)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".cache"

# Module-level cache of (256³) LUTs keyed by colormap spec, so multiple
# load_element_maps() calls in the same process don't rebuild or reload.
_FULL_LUT_CACHE: dict[str, np.ndarray] = {}


def _resolve_palette(
    colormap_spec: str,
    *,
    base_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse a colormap spec and return ``(palette_rgb, palette_scalars)``.

    ``palette_rgb`` is (N, 3) uint8, ``palette_scalars`` is (N,) float32
    in [0, 1] giving the scalar value at each palette index.

    Accepts:
      - ``'cmap:NAME'`` — matplotlib colormap (sampled at _CMAP_PALETTE_N points)
      - ``'lut:PATH'`` — (N, 3) uint8 LUT loaded from an .npy file. Path is
        resolved relative to ``base_dir`` if not absolute.
      - bare matplotlib cmap name (e.g. ``'jet'``) — equivalent to ``'cmap:jet'``
    """
    spec = colormap_spec.strip()
    if spec.startswith("cmap:"):
        name = spec[len("cmap:"):]
        scalars = np.linspace(0, 1, _CMAP_PALETTE_N, dtype=np.float32)
        rgb = (
            np.array([matplotlib.colormaps[name](s)[:3] for s in scalars]) * 255
        ).astype(np.uint8)
        return rgb, scalars
    if spec.startswith("lut:"):
        raw_path = spec[len("lut:"):]
        path = Path(raw_path)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        if not path.exists():
            raise FileNotFoundError(f"Colormap LUT not found: {path}")
        rgb = np.load(path)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(
                f"LUT at {path} must have shape (N, 3); got {rgb.shape}"
            )
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        scalars = np.linspace(0, 1, len(rgb), dtype=np.float32)
        return rgb, scalars
    if spec in matplotlib.colormaps:
        return _resolve_palette(f"cmap:{spec}", base_dir=base_dir)
    raise ValueError(
        f"Unrecognized colormap spec: {colormap_spec!r}. "
        f"Use 'cmap:NAME' or 'lut:PATH'."
    )


def _build_full_lut(
    palette_rgb: np.ndarray, palette_scalars: np.ndarray
) -> np.ndarray:
    """Build a (256, 256, 256) float32 LUT mapping every uint8 RGB triplet
    to the scalar value of its nearest palette entry.

    Uses a KD-tree for efficient nearest-neighbour search. Takes ~30-60 s
    for a 4096-entry palette, much faster for small custom LUTs.
    """
    logger.info(
        "Building RGB→scalar lookup table from %d-entry palette "
        "(256³ entries) — this is a one-time operation …",
        len(palette_rgb),
    )
    tree = cKDTree(palette_rgb.astype(np.float64))

    r, g, b = np.mgrid[0:256, 0:256, 0:256]
    all_rgb = np.column_stack(
        [r.ravel(), g.ravel(), b.ravel()]
    ).astype(np.float64)

    chunk = 1_000_000
    indices = np.empty(len(all_rgb), dtype=np.intp)
    for start in range(0, len(all_rgb), chunk):
        end = min(start + chunk, len(all_rgb))
        _, indices[start:end] = tree.query(all_rgb[start:end])

    lut = palette_scalars[indices].reshape(256, 256, 256)
    # Black [0,0,0] is always no-data, regardless of palette
    lut[0, 0, 0] = 0.0
    return lut


def get_full_lut(
    colormap_spec: str, *, base_dir: str | Path | None = None
) -> np.ndarray:
    """Return the full (256, 256, 256) RGB→scalar LUT for a colormap spec.

    Cached in memory (module-level) and on disk (.cache/ alongside the
    karak source tree). Cache key is the spec string plus an MD5
    fingerprint of the palette bytes.
    """
    if colormap_spec in _FULL_LUT_CACHE:
        return _FULL_LUT_CACHE[colormap_spec]

    palette_rgb, palette_scalars = _resolve_palette(
        colormap_spec, base_dir=base_dir
    )
    fp = hashlib.md5(palette_rgb.tobytes()).hexdigest()[:12]
    cache_file = _CACHE_DIR / f"colormap_full_lut_{fp}.npy"

    if cache_file.exists():
        logger.info("Loading cached LUT from %s", cache_file)
        full_lut = np.load(cache_file)
    else:
        full_lut = _build_full_lut(palette_rgb, palette_scalars)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, full_lut)
        logger.info("Saved LUT cache to %s", cache_file)

    _FULL_LUT_CACHE[colormap_spec] = full_lut
    return full_lut


def invert_colormap(
    rgb_image: np.ndarray,
    colormap_spec: str = "cmap:jet",
    *,
    base_dir: str | Path | None = None,
) -> np.ndarray:
    """Convert a false-color RGB image to scalar values in [0, 1].

    Parameters
    ----------
    rgb_image : np.ndarray
        (H, W, 3) uint8 RGB image.
    colormap_spec : str
        Colormap specification — ``'cmap:NAME'`` or ``'lut:PATH'``.
        See :func:`_resolve_palette`.
    base_dir : str or Path, optional
        Base directory for resolving relative ``lut:`` paths.

    Returns
    -------
    scalar : np.ndarray
        (H, W) float32 array in [0, 1]. Black pixels map to 0.0.
    """
    lut = get_full_lut(colormap_spec, base_dir=base_dir)
    rgb = rgb_image[:, :, :3]
    return lut[rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]


def invert_jet_colormap(rgb_image: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for ``invert_colormap(rgb, 'cmap:jet')``."""
    return invert_colormap(rgb_image, colormap_spec="cmap:jet")


# ---------------------------------------------------------------------------
# Filename pattern parsing
# ---------------------------------------------------------------------------

def _compile_filename_regex(pattern: str) -> re.Pattern:
    """Compile a filename pattern containing ``{element}`` to a regex.

    The ``{element}`` placeholder becomes a greedy named capture group.
    Any other ``{name}`` placeholders become non-greedy wildcards.
    All other characters in the pattern are escaped (treated as literal).

    Example: ``'Map_NWA-5218_eds_{element}.bmp'`` →
    ``r'^Map_NWA\\-5218_eds_(?P<element>.+)\\.bmp$'``
    """
    elem_sentinel = "\x00ELEMENT\x00"
    wild_sentinel = "\x00WILDCARD\x00"
    p = pattern.replace("{element}", elem_sentinel)
    p = re.sub(r"\{[^}]+\}", wild_sentinel, p)
    p = re.escape(p)
    p = p.replace(re.escape(elem_sentinel), r"(?P<element>.+)")
    p = p.replace(re.escape(wild_sentinel), r".+?")
    return re.compile(f"^{p}$")


def _legacy_extract_element(basename_no_ext: str) -> str | None:
    """Legacy element extraction: split on '-' and join parts[2:].

    Matches the original TIMA convention ``{prefix}-{sampleid}-{element}.png``
    where ``{element}`` may itself contain hyphens (e.g. ``Fe-K``).
    """
    parts = basename_no_ext.split("-")
    if len(parts) < 3:
        return None
    return "-".join(parts[2:])


def _apply_trim(
    img: np.ndarray,
    *,
    trim_top: int,
    trim_bottom: int,
    trim_left: int,
    trim_right: int,
) -> np.ndarray:
    """Crop edge strips from a 2-D or 3-D image. Zero trims pass through."""
    h, w = img.shape[:2]
    top = trim_top
    bottom = h - trim_bottom if trim_bottom > 0 else h
    left = trim_left
    right = w - trim_right if trim_right > 0 else w
    return img[top:bottom, left:right]


def load_element_maps(
    input_dir: str,
    downsample_config: DownsampleConfig,
    exclude_elements: list[str],
    bse_channel: str = "SEM",
    include_elements: list[str] | None = None,
    loader_config: LoaderConfig | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Load, invert, downsample, and trim element maps and the BSE channel.

    Element maps are false-color RGB images. Each is inverted to a scalar
    [0, 1] intensity using the colormap specified by ``loader_config``
    (default ``'cmap:jet'``). The BSE channel is true grayscale.

    Parameters
    ----------
    input_dir : str
        Directory containing element map files.
    downsample_config : DownsampleConfig
        Downsample factor and per-edge trim parameters.
    exclude_elements : list[str]
        Element names to skip (e.g. ``["Fe-L"]``).
    bse_channel : str
        Name of the BSE channel (default ``"SEM"``). Stored separately.
        If ``loader_config.bse_filename`` is set, that file is loaded as
        BSE and ``bse_channel`` is used only as the storage key.
    include_elements : list[str] or None
        If given, *only* load these elements (plus BSE).
    loader_config : LoaderConfig, optional
        File discovery, filename parsing, and colormap parameters.
        If None, defaults are used (PNG glob, legacy filename heuristic,
        jet colormap) for backward compatibility with TIMA exports.

    Returns
    -------
    elements_dict : dict[str, np.ndarray]
        Element name -> (H, W) float32 array in [0, 1].
    bse_array : np.ndarray
        (H, W) float32 BSE image.
    element_names : list[str]
        Sorted list of element names present in ``elements_dict``.
    """
    from karak.config import LoaderConfig  # local import to avoid cycle

    cfg = downsample_config
    loader = loader_config or LoaderConfig()
    factor = cfg.downsample_factor
    skip = set(exclude_elements) | _NON_ELEMENT_FILES | {bse_channel}
    include = set(include_elements) if include_elements else None

    # Edge trims, scaled by downsample factor
    trim_top = cfg.header_trim_px // factor
    trim_bottom = cfg.bottom_trim_px // factor
    trim_left = cfg.left_trim_px // factor
    trim_right = cfg.right_trim_px // factor

    # Filename → element extractor
    if loader.filename_pattern is None:
        regex = None
    else:
        regex = _compile_filename_regex(loader.filename_pattern)

    def extract_element(filename: str) -> str | None:
        if regex is not None:
            m = regex.match(filename)
            return m.group("element") if m else None
        return _legacy_extract_element(os.path.splitext(filename)[0])

    files = sorted(glob.glob(os.path.join(input_dir, loader.file_glob)))
    if not files:
        raise FileNotFoundError(
            f"No files matching {loader.file_glob!r} in {input_dir}"
        )

    elements_dict: dict[str, np.ndarray] = {}
    bse_array: np.ndarray | None = None

    # If a separate BSE filename is given, load it first
    if loader.bse_filename is not None:
        bse_path = Path(input_dir) / loader.bse_filename
        if not bse_path.exists():
            raise FileNotFoundError(f"BSE file not found: {bse_path}")
        raw = imageio.imread(bse_path)
        gray = rgb2gray(raw) if raw.ndim == 3 else raw.astype(np.float32) / 255.0
        ds = downscale_local_mean(gray, (factor, factor))
        bse_array = _apply_trim(
            ds,
            trim_top=trim_top,
            trim_bottom=trim_bottom,
            trim_left=trim_left,
            trim_right=trim_right,
        ).astype(np.float32)
        logger.info(
            "Loaded BSE channel from %s (stored as '%s'): shape %s",
            bse_path.name,
            bse_channel,
            bse_array.shape,
        )

    for fpath in files:
        filename = os.path.basename(fpath)
        # Skip the BSE file if it was already loaded separately
        if (
            loader.bse_filename is not None
            and filename == loader.bse_filename
        ):
            continue

        element = extract_element(filename)
        if element is None:
            logger.debug("Filename did not match pattern, skipping: %s", filename)
            continue

        raw = imageio.imread(fpath)

        if loader.bse_filename is None and element == bse_channel:
            # Legacy mode: BSE is in the main glob, identified by element name
            gray = rgb2gray(raw) if raw.ndim == 3 else raw.astype(np.float32) / 255.0
            ds = downscale_local_mean(gray, (factor, factor))
            bse_array = _apply_trim(
                ds,
                trim_top=trim_top,
                trim_bottom=trim_bottom,
                trim_left=trim_left,
                trim_right=trim_right,
            ).astype(np.float32)
            logger.info(
                "Loaded BSE channel '%s': shape %s", element, bse_array.shape
            )
        elif element in skip:
            logger.info("Skipping excluded channel: %s", element)
        elif include is not None and element not in include:
            logger.debug("Skipping (not in include_elements): %s", element)
        else:
            scalar = invert_colormap(
                raw, colormap_spec=loader.colormap, base_dir=input_dir
            )
            ds = downscale_local_mean(scalar, (factor, factor))
            trimmed = _apply_trim(
                ds,
                trim_top=trim_top,
                trim_bottom=trim_bottom,
                trim_left=trim_left,
                trim_right=trim_right,
            )
            elements_dict[element] = trimmed.astype(np.float32)
            logger.info(
                "Loaded element '%s' (inverted via %s): shape %s, "
                "range [%.4f, %.4f]",
                element,
                loader.colormap,
                trimmed.shape,
                trimmed.min(),
                trimmed.max(),
            )

    if bse_array is None:
        raise FileNotFoundError(
            f"BSE channel '{bse_channel}' not found in {input_dir}"
            + (
                f" (looked for filename {loader.bse_filename!r})"
                if loader.bse_filename
                else ""
            )
        )

    # Verify shape consistency
    element_names = sorted(elements_dict.keys())
    ref_shape = bse_array.shape
    for name in element_names:
        if elements_dict[name].shape != ref_shape:
            logger.warning(
                "Shape mismatch: %s is %s, expected %s",
                name,
                elements_dict[name].shape,
                ref_shape,
            )

    logger.info(
        "Loaded %d element channels + BSE (%s).  Shape: %s",
        len(element_names),
        bse_channel,
        ref_shape,
    )
    return elements_dict, bse_array, element_names


def build_compositional_cube(
    elements_dict: dict[str, np.ndarray],
    element_names: list[str],
) -> np.ndarray:
    """Stack element arrays into an (H, W, C) compositional cube.

    Parameters
    ----------
    elements_dict : dict[str, np.ndarray]
        Element name -> (H, W) array.
    element_names : list[str]
        Ordered list of element names to include (determines channel axis order).

    Returns
    -------
    cube : np.ndarray
        (H, W, C) float32 array where C = len(element_names).
    """
    arrays = [elements_dict[name] for name in element_names]
    cube = np.stack(arrays, axis=-1).astype(np.float32)
    return cube
