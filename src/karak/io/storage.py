"""HDF5 storage with group structure and provenance metadata.

The pipeline HDF5 file follows a hierarchical group layout::

    /
    +-- raw/           Element maps as-loaded (float32)
    +-- bse/           BSE image and metadata
    +-- masks/         Boolean masks and statistics
    +-- denoised/      Denoised raw [0,1] data (Plan 01-03)
    +-- normalized/    Z-score normalized data (Plan 01-03)
    +-- clusters/     PCA + HDBSCAN clustering results (Plan 02)
    attrs: pipeline_config (YAML), created, versions ...

v1.1 changes:
- normalized/ group replaces clr/ group
- save_normalized_data replaces save_clr_data
- save_mask simplified (no bse_thresh/counts sub-masks)
- Removed replaced/ and clr/ groups (z-score, not CLR)

All write functions open the file in append mode so they can be called
sequentially. Large datasets take a ``compression`` of ``"gzip"`` (level 4,
readable by any HDF5 tool), ``"lzf"`` (faster, h5py and PyTables only) or
``"none"``; compressed datasets use chunks of 256 x 256 pixels with every
trailing axis whole, and the denoised and normalized cubes also use the
shuffle filter (see ``dataset_options``).  Matching ``load_*`` functions read data back for pipeline
resume.
"""

from __future__ import annotations

import datetime
import json
import logging
import platform
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import yaml

if TYPE_CHECKING:
    from karak.clustering.tiling import PhaseEntry, TileResult

from karak.provenance import get_software_versions

logger = logging.getLogger(__name__)

# Standard group layout for v1.1
_GROUPS = ["raw", "bse", "masks", "denoised", "normalized", "clusters"]

COMPRESSION = ("gzip", "lzf", "none")
_IMAGE_CHUNK = 256          # pixels per side of an image or cube chunk
_ROW_CHUNK_BYTES = 1 << 20  # about 1 MB per chunk of a 1-D array or table


def _chunks(shape: tuple, itemsize: int) -> tuple:
    """Chunk shape: 256 x 256 pixels with every trailing axis whole for an
    image or cube, about 1 MB of rows for a 1-D array or a narrow table."""
    if len(shape) >= 2 and shape[1] > 16:   # (H, W) or (H, W, C)
        return (min(_IMAGE_CHUNK, shape[0]), min(_IMAGE_CHUNK, shape[1]),
                *shape[2:])
    row_bytes = itemsize * int(np.prod(shape[1:], dtype=np.int64))
    rows = max(1, _ROW_CHUNK_BYTES // max(1, row_bytes))
    return (min(rows, shape[0]), *shape[1:])


def dataset_options(data: np.ndarray, compression: str, *, shuffle: bool) -> dict:
    """``create_dataset`` keyword arguments for ``compression``: gzip
    (level 4) or lzf with chunks from ``_chunks``, or nothing for "none".

    ``shuffle`` adds HDF5's shuffle filter, which groups the bytes of each
    value before compression; it is part of the HDF5 library, so any reader
    that opens gzip opens it. It suits continuous floats: the NWA 4587
    denoised cube (2 GB) writes with gzip in 19 s instead of 32 s, at 0.74
    GB instead of 0.84 GB. It hurts data with few distinct values: the raw
    element maps (1,249 values in a map, a third of them zeros) grow from
    451 MB to 798 MB and take 36 s instead of 21 s.
    """
    if compression not in COMPRESSION:
        raise ValueError(f"compression must be one of {COMPRESSION}, got {compression!r}")
    if compression == "none" or data.size == 0:
        return {}
    options = {"chunks": _chunks(data.shape, data.dtype.itemsize)}
    if shuffle:
        options["shuffle"] = True
    if compression == "gzip":
        options.update(compression="gzip", compression_opts=4)
    else:
        options.update(compression="lzf")
    return options


def create_pipeline_hdf5(path: str | Path, config_dict: dict) -> None:
    """Create an HDF5 file with standard group structure and provenance.

    Parameters
    ----------
    path : str or Path
        Path for the new HDF5 file.
    config_dict : dict
        Pipeline configuration (or flow definition) to embed as a YAML
        attribute for provenance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        # Create group hierarchy
        for grp in _GROUPS:
            f.create_group(grp)

        # Top-level provenance attributes
        f.attrs["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        f.attrs["pipeline_version"] = "1.1.0"
        f.attrs["python_version"] = platform.python_version()
        f.attrs["platform"] = platform.platform()

        # Library versions
        versions = get_software_versions()
        f.attrs["library_versions"] = json.dumps(versions)

        # Embed full pipeline config as YAML
        config_yaml = yaml.dump(
            config_dict, default_flow_style=False, sort_keys=False
        )
        f.attrs["pipeline_config"] = config_yaml

    logger.info("Created HDF5: %s with groups %s", path, _GROUPS)


def save_raw_data(
    h5_path: str | Path,
    elements_dict: dict[str, np.ndarray],
    element_names: list[str],
    *,
    compression: str,
) -> None:
    """Write element arrays to the ``raw/`` group.

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    elements_dict : dict[str, np.ndarray]
        Element name -> (H, W) array.
    element_names : list[str]
        Ordered element names (stored as an attribute for channel ordering).
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["raw"]
        for name in element_names:
            if name in grp:
                del grp[name]
            data = elements_dict[name].astype(np.float32)
            grp.create_dataset(
                name, data=data, **dataset_options(data, compression, shuffle=False))

        # Store ordered element list
        grp.attrs["element_names"] = json.dumps(element_names)
        grp.attrs["n_elements"] = len(element_names)
        shape = elements_dict[element_names[0]].shape
        grp.attrs["height"] = shape[0]
        grp.attrs["width"] = shape[1]

    logger.info("Saved %d element maps to raw/ group", len(element_names))


def save_bse(
    h5_path: str | Path,
    bse: np.ndarray,
    original_shape: tuple[int, ...],
    downsample_factor: int,
    *,
    compression: str,
) -> None:
    """Write BSE array to the ``bse/`` group with metadata.

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    bse : np.ndarray
        (H, W) BSE image (full image, no crop).
    original_shape : tuple
        Shape of the BSE image before processing (for provenance).
    downsample_factor : int
        Downsample factor applied.
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["bse"]
        if "image" in grp:
            del grp["image"]
        image = bse.astype(np.float32)
        grp.create_dataset(
            "image", data=image, **dataset_options(image, compression, shuffle=False))
        grp.attrs["original_shape"] = list(original_shape)
        grp.attrs["downsample_factor"] = downsample_factor
        grp.attrs["stored_shape"] = list(bse.shape)

    logger.info("Saved BSE to bse/ group, shape %s", bse.shape)


def save_mask(
    h5_path: str | Path,
    mineral_mask: np.ndarray,
    valid_mask: np.ndarray | None,
    mask_stats: dict,
    params: dict,
    *,
    compression: str,
) -> None:
    """Write mask arrays and statistics to the ``masks/`` group.

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    mineral_mask : np.ndarray
        (H, W) boolean mask (True = mineral).
    valid_mask : np.ndarray or None
        (H, W) boolean valid-region mask (optional).
    mask_stats : dict
        Statistics from ``compute_mask_statistics()``.
    params : dict
        Masking parameters used to generate the mask (provenance).
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["masks"]

        # Primary mineral mask
        if "mineral" in grp:
            del grp["mineral"]
        grp.create_dataset("mineral", data=mineral_mask,
                           **dataset_options(mineral_mask, compression, shuffle=False))

        # Valid region mask
        if valid_mask is not None:
            if "valid" in grp:
                del grp["valid"]
            grp.create_dataset("valid", data=valid_mask,
                               **dataset_options(valid_mask, compression, shuffle=False))

        # Statistics as attributes
        for key, val in mask_stats.items():
            grp.attrs[key] = val

        # Mask config as YAML
        grp.attrs["mask_config"] = yaml.dump(
            params, default_flow_style=False, sort_keys=False
        )

    logger.info(
        "Saved mask to masks/ group: %.1f%% mineral coverage",
        mask_stats.get("coverage_pct", 0),
    )


def save_denoised_data(
    h5_path: str | Path,
    denoised_cube: np.ndarray,
    element_names: list[str],
    params: dict,
    *,
    compression: str,
) -> None:
    """Write denoised raw [0,1] cube to the ``denoised/`` group in HDF5.

    Dataset written:
    - ``denoised/cube`` -- (H, W, C) float32 denoised cube

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    denoised_cube : np.ndarray
        (H, W, C) denoised raw [0,1] cube.
    element_names : list[str]
        Ordered element names.
    params : dict
        Denoising parameters used (must include ``"method"``).
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["denoised"]
        if "cube" in grp:
            del grp["cube"]
        cube = denoised_cube.astype(np.float32)
        grp.create_dataset(
            "cube", data=cube, **dataset_options(cube, compression, shuffle=True))

        # Record method and all parameters
        method = params["method"]
        grp.attrs["method"] = method
        grp.attrs["element_order"] = json.dumps(element_names)

        if method == "bilateral":
            sigma_color = params["sigma_color"]
            grp.attrs["sigma_color"] = (
                sigma_color if sigma_color is not None else "auto"
            )
            grp.attrs["sigma_spatial"] = params["sigma_spatial"]
        elif method == "anisotropic_diffusion":
            grp.attrs["niter"] = params["niter"]
            grp.attrs["kappa"] = params["kappa"]
            grp.attrs["gamma"] = params["gamma"]
            grp.attrs["option"] = params["option"]

    logger.info(
        "Saved denoised cube %s to denoised/ group (method=%s)",
        denoised_cube.shape,
        method,
    )


def save_normalized_data(
    h5_path: str | Path,
    normalized_cube: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    element_names: list[str],
    method: str,
    *,
    compression: str,
) -> None:
    """Write z-score normalized cube to the ``normalized/`` group in HDF5.

    Datasets written:
    - ``normalized/cube`` -- (H, W, C) float32 z-score normalized cube
    - ``normalized/means`` -- (C,) per-channel means
    - ``normalized/stds`` -- (C,) per-channel standard deviations

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    normalized_cube : np.ndarray
        (H, W, C) z-score normalized cube.
    means : np.ndarray
        (C,) per-channel means used for normalization.
    stds : np.ndarray
        (C,) per-channel standard deviations used for normalization.
    element_names : list[str]
        Ordered element names.
    method : str
        Normalization method used (provenance).
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["normalized"]
        for ds_name in ["cube", "means", "stds"]:
            if ds_name in grp:
                del grp[ds_name]

        cube = normalized_cube.astype(np.float32)
        grp.create_dataset(
            "cube", data=cube, **dataset_options(cube, compression, shuffle=True))
        grp.create_dataset("means", data=means.astype(np.float32))
        grp.create_dataset("stds", data=stds.astype(np.float32))

        grp.attrs["method"] = method
        grp.attrs["element_order"] = json.dumps(element_names)
        grp.attrs["n_elements"] = len(element_names)

    logger.info(
        "Saved normalized cube %s to normalized/ group (method=%s)",
        normalized_cube.shape,
        method,
    )


def save_cluster_data(
    h5_path: str | Path,
    raw_labels: np.ndarray,
    cleaned_labels: np.ndarray,
    probabilities: np.ndarray,
    pca_variance_ratio: np.ndarray,
    mineral_indices: np.ndarray,
    cluster_stats: dict,
    n_pca_components_used: int,
    params: dict,
    *,
    compression: str,
) -> None:
    """Write clustering results to the ``clusters/`` group in HDF5.

    Datasets written:
    - ``clusters/raw_labels`` -- (N_mineral,) int32 HDBSCAN labels (-1 = noise)
    - ``clusters/cleaned_labels`` -- (N_mineral,) int32 kNN-cleaned labels (no -1)
    - ``clusters/probabilities`` -- (N_mineral,) float32 membership probabilities
    - ``clusters/pca_variance_ratio`` -- (C,) float64 per-component explained variance ratio
    - ``clusters/mineral_indices`` -- (N_mineral, 2) int32 pixel coordinates

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    raw_labels : np.ndarray
        (N_mineral,) int32 HDBSCAN labels with noise as -1.
    cleaned_labels : np.ndarray
        (N_mineral,) int32 cleaned labels (noise reassigned via kNN).
    probabilities : np.ndarray
        (N_mineral,) float32 HDBSCAN membership probabilities.
    pca_variance_ratio : np.ndarray
        (C,) explained variance ratio from PCA.
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 (row, col) coordinates of mineral pixels.
    cluster_stats : dict
        Summary statistics from compute_cluster_stats().
    n_pca_components_used : int
        Number of PCA components used for clustering.
    params : dict
        Clustering parameters used (should include ``"strategy"``).
    """
    with h5py.File(h5_path, "a") as f:
        if "clusters" not in f:
            f.create_group("clusters")
        grp = f["clusters"]

        datasets = {
            "raw_labels": raw_labels.astype(np.int32),
            "cleaned_labels": cleaned_labels.astype(np.int32),
            "probabilities": probabilities.astype(np.float32),
            "pca_variance_ratio": np.asarray(pca_variance_ratio, dtype=np.float64),
            "mineral_indices": mineral_indices.astype(np.int32),
        }

        for ds_name, data in datasets.items():
            if ds_name in grp:
                del grp[ds_name]
            grp.create_dataset(ds_name, data=data,
                               **dataset_options(data, compression, shuffle=False))

        # Attributes
        grp.attrs["n_pca_components_used"] = n_pca_components_used
        grp.attrs["n_clusters"] = cluster_stats["n_clusters"]
        grp.attrs["n_noise"] = cluster_stats["n_noise"]
        grp.attrs["noise_pct"] = cluster_stats["noise_pct"]
        grp.attrs["n_mineral_pixels"] = cluster_stats["n_total"]
        grp.attrs["cluster_stats"] = json.dumps(cluster_stats)
        grp.attrs["cluster_config"] = yaml.dump(
            params, default_flow_style=False, sort_keys=False
        )
        grp.attrs["clustering_strategy"] = params["strategy"]

    logger.info(
        "Saved cluster data to clusters/ group: %d clusters, %.1f%% noise",
        cluster_stats["n_clusters"],
        cluster_stats["noise_pct"],
    )


def load_cluster_data(
    h5_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read clustering results from the ``clusters/`` group.

    Returns
    -------
    raw_labels : np.ndarray
        (N_mineral,) int32 HDBSCAN labels (-1 = noise).
    cleaned_labels : np.ndarray
        (N_mineral,) int32 kNN-cleaned labels.
    probabilities : np.ndarray
        (N_mineral,) float32 membership probabilities.
    pca_variance_ratio : np.ndarray
        (C,) explained variance ratio from PCA.
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 pixel coordinates.
    cluster_stats : dict
        Summary statistics.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["clusters"]
        raw_labels = grp["raw_labels"][:]
        cleaned_labels = grp["cleaned_labels"][:]
        probabilities = grp["probabilities"][:]
        pca_variance_ratio = grp["pca_variance_ratio"][:]
        mineral_indices = grp["mineral_indices"][:]
        cluster_stats = json.loads(grp.attrs["cluster_stats"])
    logger.info("Loaded cluster data from clusters/ group")
    return raw_labels, cleaned_labels, probabilities, pca_variance_ratio, mineral_indices, cluster_stats


def save_tiled_metadata(
    h5_path: str | Path,
    tile_results: list,
    phase_registry: list,
) -> None:
    """Write per-tile summaries and phase registry to ``clusters/tiled/``.

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file with a ``clusters/`` group.
    tile_results : list[TileResult]
        Per-tile diagnostic summaries.
    phase_registry : list[PhaseEntry]
        Final phase registry entries.
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["clusters"]
        if "tiled" in grp:
            del grp["tiled"]
        tiled_grp = grp.create_group("tiled")

        # Per-tile summary table
        n_tiles = len(tile_results)
        tile_ids = np.array([tr.tile_id for tr in tile_results], dtype=np.int32)
        tile_n_pixels = np.array([tr.n_pixels for tr in tile_results], dtype=np.int32)
        tile_n_clusters = np.array([tr.n_clusters for tr in tile_results], dtype=np.int32)
        tile_n_noise = np.array([tr.n_noise for tr in tile_results], dtype=np.int32)
        tile_n_new = np.array(
            [len(tr.new_phases) for tr in tile_results], dtype=np.int32
        )
        tile_deferred = np.array(
            [bool(getattr(tr, "deferred", False)) for tr in tile_results], dtype=bool
        )

        tiled_grp.create_dataset("tile_ids", data=tile_ids)
        tiled_grp.create_dataset("tile_n_pixels", data=tile_n_pixels)
        tiled_grp.create_dataset("tile_n_clusters", data=tile_n_clusters)
        tiled_grp.create_dataset("tile_n_noise", data=tile_n_noise)
        tiled_grp.create_dataset("tile_n_new_phases", data=tile_n_new)
        tiled_grp.create_dataset("tile_deferred", data=tile_deferred)

        tiled_grp.attrs["n_tiles"] = n_tiles

        # Phase registry
        n_phases = len(phase_registry)
        phase_ids = np.array([p.global_id for p in phase_registry], dtype=np.int32)
        phase_n_pixels = np.array([p.n_pixels for p in phase_registry], dtype=np.int32)
        phase_discovered_in = np.array(
            [p.discovered_in_tile for p in phase_registry], dtype=np.int32
        )
        fingerprint_matrix = np.array(
            [p.mean_fingerprint for p in phase_registry], dtype=np.float32
        )

        tiled_grp.create_dataset("phase_ids", data=phase_ids)
        tiled_grp.create_dataset("phase_n_pixels", data=phase_n_pixels)
        tiled_grp.create_dataset("phase_discovered_in_tile", data=phase_discovered_in)
        tiled_grp.create_dataset(
            "phase_fingerprints", data=fingerprint_matrix, compression="gzip"
        )

        tiled_grp.attrs["n_phases"] = n_phases

        # Pass 2 summary
        n_rare = sum(1 for p in phase_registry if p.discovered_in_tile == -1)
        tiled_grp.attrs["n_rare_phases"] = n_rare
        tiled_grp.attrs["n_major_phases"] = n_phases - n_rare

    logger.info(
        "Saved tiled metadata: %d tiles, %d phases (%d major, %d rare)",
        n_tiles, n_phases, n_phases - n_rare, n_rare,
    )


# ---------------------------------------------------------------------------
# Read-back functions
# ---------------------------------------------------------------------------


def load_raw_data(
    h5_path: str | Path,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read element maps from the ``raw/`` group.

    Returns
    -------
    elements_dict : dict[str, np.ndarray]
        Element name -> (H, W) float32 array.
    element_names : list[str]
        Ordered element names.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["raw"]
        element_names: list[str] = json.loads(grp.attrs["element_names"])
        elements_dict = {name: grp[name][:] for name in element_names}
    logger.info("Loaded %d element maps from raw/ group", len(element_names))
    return elements_dict, element_names


def load_bse(h5_path: str | Path) -> np.ndarray:
    """Read BSE image from the ``bse/`` group.

    Returns
    -------
    np.ndarray
        (H, W) float32 BSE image.
    """
    with h5py.File(h5_path, "r") as f:
        bse = f["bse/image"][:]
    logger.info("Loaded BSE from bse/ group, shape %s", bse.shape)
    return bse


def load_masks(
    h5_path: str | Path,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Read masks and statistics from the ``masks/`` group.

    Returns
    -------
    mineral_mask : np.ndarray
        (H, W) boolean mask.
    valid_mask : np.ndarray or None
        (H, W) boolean valid-region mask, or *None* if not stored.
    mask_stats : dict
        Mask statistics stored as group attributes.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["masks"]
        mineral_mask = grp["mineral"][:]
        valid_mask = grp["valid"][:] if "valid" in grp else None
        # Rebuild stats dict from attributes (skip non-stat attrs)
        skip = {"mask_config", "stage_completed"}
        mask_stats = {
            k: v for k, v in dict(grp.attrs).items() if k not in skip
        }
    logger.info("Loaded masks from masks/ group")
    return mineral_mask, valid_mask, mask_stats


def load_denoised_data(
    h5_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    """Read denoised cube from the ``denoised/`` group.

    Returns
    -------
    cube : np.ndarray
        (H, W, C) float32 denoised cube.
    element_names : list[str]
        Ordered element names.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["denoised"]
        cube = grp["cube"][:]
        element_names = json.loads(grp.attrs["element_order"])
    logger.info("Loaded denoised cube %s from denoised/ group", cube.shape)
    return cube, element_names


def load_normalized_data(
    h5_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read normalized cube from the ``normalized/`` group.

    Returns
    -------
    cube : np.ndarray
        (H, W, C) float32 z-score normalized cube.
    means : np.ndarray
        (C,) per-channel means.
    stds : np.ndarray
        (C,) per-channel standard deviations.
    element_names : list[str]
        Ordered element names.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["normalized"]
        cube = grp["cube"][:]
        means = grp["means"][:]
        stds = grp["stds"][:]
        element_names = json.loads(grp.attrs["element_order"])
    logger.info("Loaded normalized cube %s from normalized/ group", cube.shape)
    return cube, means, stds, element_names


def save_mineral_names(h5_path: str | Path, mineral_names: dict[int, str]) -> None:
    """Save mineral name mapping to the ``clusters/`` group attributes.

    The mapping is stored both as a single JSON-serialized attribute
    (``mineral_names``) for programmatic access and as individual
    ``cluster_N_name`` attributes for easy browsing in HDFView.

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file with a ``clusters/`` group.
    mineral_names : dict[int, str]
        Mapping of cluster label (int) -> mineral name (str).
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["clusters"]
        # JSON-serialized mapping (keys become strings in JSON)
        grp.attrs["mineral_names"] = json.dumps(
            {str(k): v for k, v in mineral_names.items()}
        )
        # Individual attributes for HDFView convenience
        for k, v in mineral_names.items():
            grp.attrs[f"cluster_{k}_name"] = v

    logger.info(
        "Saved mineral names to clusters/ group: %s",
        {k: v for k, v in mineral_names.items()},
    )


def save_hires_labels(h5_path, image, *, ratio, names, downsample_factor,
                      header_trim_px, left_trim_px, compression) -> None:
    """Write a full-resolution sub-phase map to ``clusters/hires/labels``."""
    with h5py.File(h5_path, "a") as f:
        grp = f.require_group("clusters").require_group("hires")
        if "labels" in grp:
            del grp["labels"]
        data = np.asarray(image, dtype=np.int16)
        ds = grp.create_dataset("labels", data=data,
                                **dataset_options(data, compression, shuffle=False))
        ds.attrs["ratio"] = int(ratio)
        ds.attrs["names"] = json.dumps({str(k): v for k, v in names.items()})
        ds.attrs["downsample_factor"] = int(downsample_factor)
        ds.attrs["header_trim_px"] = int(header_trim_px)
        ds.attrs["left_trim_px"] = int(left_trim_px)
    logger.info("Saved hires labels %s to clusters/hires/labels", data.shape)


def save_subclustering(h5_path: str | Path, history: list[dict]) -> None:
    """Write the split history to ``clusters/subclustering`` attributes:
    ``history`` (the JSON list) and one ``split_NN`` attribute per record.
    This is karak's record schema (``history``, ``split_NN``), one attribute
    per split; the published NWA 4587 file used hand-written ``phase_N_*``
    attributes instead."""
    with h5py.File(h5_path, "a") as f:
        grp = f["clusters"]
        if "subclustering" in grp:
            del grp["subclustering"]
        sub = grp.create_group("subclustering")
        records = [_json_record(r) for r in history]
        sub.attrs["history"] = json.dumps(records)
        for i, rec in enumerate(records):
            sub.attrs[f"split_{i:02d}"] = json.dumps(rec)
    logger.info("Saved %d split records to clusters/subclustering", len(history))


def _json_record(record: dict) -> dict:
    """Return ``record`` with numpy scalars, arrays and non-string dict
    keys converted recursively to JSON-safe Python values."""
    return _json_safe(record)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_mineral_names(h5_path: str | Path) -> dict[int, str] | None:
    """Read mineral name mapping from the ``clusters/`` group attributes.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file.

    Returns
    -------
    dict[int, str] or None
        Mapping of cluster label (int) -> mineral name (str), or *None*
        if names have not been assigned yet.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f["clusters"]
        if "mineral_names" not in grp.attrs:
            logger.info("No mineral names found in clusters/ group")
            return None
        raw = json.loads(grp.attrs["mineral_names"])

    # Parse keys back to int (JSON serialises dict keys as strings)
    mineral_names = {int(k): v for k, v in raw.items()}
    logger.info("Loaded mineral names from clusters/ group: %s", mineral_names)
    return mineral_names
