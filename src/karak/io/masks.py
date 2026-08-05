"""Background / epoxy masking for SEM-EDS data.

v1.1: Uses valid-region polygon mask + element-zero detection instead of
BSE Otsu + total-counts thresholding.  The mineral mask is the intersection
of the valid polygon and non-zero element pixels.

Optionally applies a manually-drawn valid-region mask (napari shapes CSV)
to restrict processing to the area containing the mineral sample.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.measure import label
from skimage.morphology import remove_small_objects

logger = logging.getLogger(__name__)


def load_valid_mask(
    csv_path: str | Path,
    image_shape: tuple[int, int],
    downsample_factor: int = 1,
    header_trim_px: int = 0,
    left_trim_px: int = 0,
) -> np.ndarray:
    """Load a valid-region mask from a napari shapes CSV export.

    The CSV contains polygon vertices in full-resolution image coordinates.
    Vertices are scaled by the downsample factor and offset by the
    top / left trims before rasterizing to a boolean mask.

    Only ``polygon`` shapes are rasterized (``path`` shapes are skipped
    since they represent open polylines without a filled interior).

    Parameters
    ----------
    csv_path : str or Path
        Path to the napari shapes CSV (columns: index, shape-type,
        vertex-index, axis-0, axis-1).
    image_shape : tuple[int, int]
        (H, W) of the target image (after downsample and edge trims).
    downsample_factor : int
        Factor by which the image was downsampled from the original
        resolution used when drawing the mask.
    header_trim_px : int
        Pixels trimmed from the top of the image (in original resolution).
    left_trim_px : int
        Pixels trimmed from the left of the image (in original resolution).
        Bottom and right trims do not affect coordinates — they just shrink
        the target shape, so polygons drawn at the full resolution still
        align after cropping.

    Returns
    -------
    mask : np.ndarray
        (H, W) boolean array.  ``True`` = inside valid sample region.
    """
    csv_path = Path(csv_path)

    # Parse CSV into per-shape vertex lists
    shapes: dict[int, list[tuple[float, float]]] = {}
    shape_types: dict[int, str] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            shape_types[idx] = row["shape-type"]
            if idx not in shapes:
                shapes[idx] = []
            shapes[idx].append((float(row["axis-0"]), float(row["axis-1"])))

    H, W = image_shape
    mask = np.zeros((H, W), dtype=bool)
    top_offset = header_trim_px / downsample_factor
    left_offset = left_trim_px / downsample_factor

    n_rasterized = 0
    for idx in sorted(shapes):
        vertices = shapes[idx]
        if shape_types[idx] != "polygon":
            logger.info(
                "Skipping shape %d (type '%s') -- only polygons are rasterized",
                idx,
                shape_types[idx],
            )
            continue

        # Scale coordinates: full-res -> downsampled/trimmed space
        rows = np.array([v[0] / downsample_factor - top_offset for v in vertices])
        cols = np.array([v[1] / downsample_factor - left_offset for v in vertices])

        rr, cc = draw_polygon(rows, cols, shape=(H, W))
        mask[rr, cc] = True
        n_rasterized += 1
        logger.info(
            "Rasterized polygon shape %d: %d vertices, %d pixels filled",
            idx,
            len(vertices),
            len(rr),
        )

    if n_rasterized == 0:
        logger.warning("No polygon shapes found in %s -- mask is all False", csv_path)

    n_valid = int(mask.sum())
    logger.info(
        "Valid mask: %d pixels (%.1f%% of image) from %s",
        n_valid,
        100.0 * n_valid / mask.size,
        csv_path.name,
    )
    return mask


def create_mineral_mask(
    element_cube: np.ndarray,
    valid_mask: np.ndarray | None = None,
    min_object_size: int = 100,
) -> np.ndarray:
    """Create a boolean mineral-pixel mask from element maps.

    A pixel is mineral if:
    1. It is inside the valid region (if valid_mask is provided), AND
    2. It is NOT zero across all element channels.

    Parameters
    ----------
    element_cube : np.ndarray
        (H, W, C) element intensity cube (raw [0,1] values).
    valid_mask : np.ndarray or None
        (H, W) boolean mask from ``load_valid_mask``.  If None, all
        pixels are considered valid.
    min_object_size : int
        Remove connected components smaller than this (default 100).

    Returns
    -------
    mask : np.ndarray
        (H, W) boolean array.  ``True`` = mineral pixel (keep),
        ``False`` = background / epoxy / outside valid region.
    """
    H, W, C = element_cube.shape

    # Element-zero detection: pixels where ALL channels are zero
    all_zero = np.all(element_cube == 0, axis=-1)  # (H, W)
    has_signal = ~all_zero

    if valid_mask is not None:
        combined = valid_mask & has_signal
    else:
        combined = has_signal

    # Remove small isolated objects
    if min_object_size > 0:
        combined = remove_small_objects(combined, min_size=min_object_size)

    n_mineral = int(combined.sum())
    n_total = combined.size
    logger.info(
        "Mineral mask: %d pixels (%.1f%% of image)",
        n_mineral,
        100.0 * n_mineral / n_total,
    )
    if valid_mask is not None:
        n_valid = int(valid_mask.sum())
        if n_valid > 0:
            logger.info(
                "Mineral coverage within valid region: %.1f%%",
                100.0 * n_mineral / n_valid,
            )
    return combined


def compute_mask_statistics(
    mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """Compute summary statistics for a mineral mask.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) boolean mask (True = mineral).
    valid_mask : np.ndarray or None
        (H, W) boolean valid-region mask.

    Returns
    -------
    stats : dict
        Keys: coverage_pct, coverage_of_valid_pct, n_mineral_pixels,
        n_valid_pixels, n_total_pixels, n_connected_components.
    """
    n_total = mask.size
    n_mineral = int(mask.sum())

    labeled = label(mask)
    n_components = labeled.max()

    stats = {
        "coverage_pct": round(100.0 * n_mineral / n_total, 2),
        "n_mineral_pixels": n_mineral,
        "n_total_pixels": n_total,
        "n_connected_components": int(n_components),
    }

    if valid_mask is not None:
        n_valid = int(valid_mask.sum())
        stats["n_valid_pixels"] = n_valid
        stats["coverage_of_valid_pct"] = (
            round(100.0 * n_mineral / n_valid, 2) if n_valid > 0 else 0.0
        )
    else:
        stats["n_valid_pixels"] = n_total
        stats["coverage_of_valid_pct"] = stats["coverage_pct"]

    return stats
