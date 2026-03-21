"""Data loading utilities for SEM-EDS element maps.

Refactored from ``notebooks/1 - Clean and Prep Data.ipynb``.  All PNG
element maps (and the BSE / SEM channel) are loaded, downsampled, and
header-trimmed.

v1.1: Full-image loading -- no ROI crop.  Uses DownsampleConfig instead
of CropConfig.

TIMA exports element maps as false-color (jet colormapped) RGB PNGs.
The BSE/SEM channel is true grayscale (R==G==B).  Element maps are
converted to scalar intensity by inverting the jet colormap; the BSE
channel uses standard ``rgb2gray``.

Image resolution note
---------------------
The raw PNGs exported from TIMA are all at the same pixel resolution
(BSE and EDS alike).  The existing notebook downsamples *every* image
by a factor of 2 to keep file sizes manageable.  We replicate that
behaviour here: ``downsample_factor`` in DownsampleConfig is applied to all
channels uniformly.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import matplotlib
import numpy as np
from scipy.spatial import cKDTree
from skimage.color import rgb2gray
from skimage.transform import downscale_local_mean

if TYPE_CHECKING:
    from karak.config import DownsampleConfig

logger = logging.getLogger(__name__)

# Elements / files to always skip (not element channels)
_NON_ELEMENT_FILES = {"Phases"}

# ---------------------------------------------------------------------------
# Jet colormap inversion via precomputed (256, 256, 256) lookup table
# ---------------------------------------------------------------------------
_JET_LUT_N = 4096
_JET_SCALARS = np.linspace(0, 1, _JET_LUT_N, dtype=np.float32)
_JET_RGB = (
    np.array([matplotlib.colormaps["jet"](s)[:3] for s in _JET_SCALARS]) * 255
).astype(np.uint8)  # (_JET_LUT_N, 3)

# Cache directory for the full RGB→scalar lookup table
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".cache"

# Fingerprint the LUT so cache auto-invalidates if _JET_LUT_N changes
_LUT_HASH = hashlib.md5(_JET_RGB.tobytes()).hexdigest()[:12]
_LUT_CACHE_FILE = _CACHE_DIR / f"jet_rgb_lut_{_LUT_HASH}.npy"

# Module-level singleton — lazily initialised by _get_jet_lut()
_JET_FULL_LUT: np.ndarray | None = None


def _build_jet_lut() -> np.ndarray:
    """Build a (256, 256, 256) float32 lookup table mapping every uint8 RGB
    triplet to its nearest jet scalar value in [0, 1].

    Uses a KD-tree for efficient nearest-neighbour search over the 4096-entry
    jet palette.  Takes ~30-60 s on first run, then is cached to disk.
    """
    logger.info(
        "Building jet RGB→scalar lookup table (256³ entries) — "
        "this is a one-time operation …"
    )
    tree = cKDTree(_JET_RGB.astype(np.float64))

    # All possible uint8 RGB triplets: (16 777 216, 3)
    r, g, b = np.mgrid[0:256, 0:256, 0:256]
    all_rgb = np.column_stack(
        [r.ravel(), g.ravel(), b.ravel()]
    ).astype(np.float64)

    # Query in chunks to keep peak memory reasonable (~400 MB working set)
    chunk = 1_000_000
    indices = np.empty(len(all_rgb), dtype=np.intp)
    for start in range(0, len(all_rgb), chunk):
        end = min(start + chunk, len(all_rgb))
        _, indices[start:end] = tree.query(all_rgb[start:end])

    lut = _JET_SCALARS[indices].reshape(256, 256, 256)
    # Black [0,0,0] is always no-data
    lut[0, 0, 0] = 0.0
    return lut


def _get_jet_lut() -> np.ndarray:
    """Return the full RGB→scalar LUT, loading from cache or building it."""
    global _JET_FULL_LUT  # noqa: PLW0603
    if _JET_FULL_LUT is not None:
        return _JET_FULL_LUT

    if _LUT_CACHE_FILE.exists():
        logger.info("Loading cached jet LUT from %s", _LUT_CACHE_FILE)
        _JET_FULL_LUT = np.load(_LUT_CACHE_FILE)
        return _JET_FULL_LUT

    _JET_FULL_LUT = _build_jet_lut()

    # Persist to disk so subsequent runs are instant
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_LUT_CACHE_FILE, _JET_FULL_LUT)
    logger.info("Saved jet LUT cache to %s", _LUT_CACHE_FILE)
    return _JET_FULL_LUT


def invert_jet_colormap(rgb_image: np.ndarray) -> np.ndarray:
    """Convert a jet-colormapped RGB image to scalar values in [0, 1].

    Uses a precomputed (256, 256, 256) lookup table for O(1) per-pixel
    inversion.  The table is built once (via KD-tree) and cached to disk.

    Black pixels ``[0, 0, 0]`` are mapped to 0.0 (below detection / no data).

    Parameters
    ----------
    rgb_image : np.ndarray
        (H, W, 3) uint8 RGB image colormapped with matplotlib jet.

    Returns
    -------
    scalar : np.ndarray
        (H, W) float32 array in [0, 1].
    """
    lut = _get_jet_lut()
    rgb = rgb_image[:, :, :3]
    return lut[rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]]


def load_element_maps(
    input_dir: str,
    downsample_config: DownsampleConfig,
    exclude_elements: list[str],
    bse_channel: str = "SEM",
    include_elements: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Load, downsample, and trim element maps from PNG files.

    Element maps are jet-colormapped RGB PNGs from TIMA; they are converted
    to scalar [0, 1] values by inverting the jet colormap.  The BSE channel
    is true grayscale and uses standard ``rgb2gray``.

    v1.1: Full image -- no ROI crop.

    Parameters
    ----------
    input_dir : str
        Directory containing ``{SampleID}-{Element}.png`` files.
    downsample_config : DownsampleConfig
        Downsampling and header trim parameters.
    exclude_elements : list[str]
        Element names to skip (e.g. ``["Fe-L"]``).
    bse_channel : str
        Name of the BSE channel (default ``"SEM"``).  Stored separately.
    include_elements : list[str] or None
        If given, *only* load these elements (plus BSE).  Overrides
        ``exclude_elements``.  Useful for fast testing with a subset.

    Returns
    -------
    elements_dict : dict[str, np.ndarray]
        Element name -> (H, W) float32 array (0-1 from jet inversion).
    bse_array : np.ndarray
        (H, W) float32 BSE image.
    element_names : list[str]
        Sorted list of element names present in *elements_dict*.
    """
    cfg = downsample_config
    factor = cfg.downsample_factor
    skip = set(exclude_elements) | _NON_ELEMENT_FILES | {bse_channel}
    include = set(include_elements) if include_elements else None
    trim = cfg.header_trim_px // factor  # header trim after downsampling

    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not png_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")

    elements_dict: dict[str, np.ndarray] = {}
    bse_array: np.ndarray | None = None

    for fpath in png_files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        # Element name: everything after the second hyphen
        # e.g. "SampleID-Fe-K" -> "Fe-K"
        parts = basename.split("-")
        element = "-".join(parts[2:])

        # Load raw RGB image
        raw = imageio.imread(fpath)

        if element == bse_channel:
            # BSE is true grayscale (R==G==B) -- use standard conversion
            gray = rgb2gray(raw)
            ds = downscale_local_mean(gray, (factor, factor))
            trimmed = ds[trim:, :]
            bse_array = trimmed.astype(np.float32)
            logger.info(
                "Loaded BSE channel '%s': shape %s", element, bse_array.shape
            )
        elif element in skip:
            logger.info("Skipping excluded channel: %s", element)
        elif include is not None and element not in include:
            logger.debug("Skipping (not in include_elements): %s", element)
        else:
            # Element maps are jet-colormapped -- invert to scalar
            scalar = invert_jet_colormap(raw)
            ds = downscale_local_mean(scalar, (factor, factor))
            trimmed = ds[trim:, :]
            elements_dict[element] = trimmed.astype(np.float32)
            logger.info(
                "Loaded element '%s' (jet inverted): shape %s, "
                "range [%.4f, %.4f]",
                element,
                trimmed.shape,
                trimmed.min(),
                trimmed.max(),
            )

    if bse_array is None:
        raise FileNotFoundError(
            f"BSE channel '{bse_channel}' not found in {input_dir}"
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
