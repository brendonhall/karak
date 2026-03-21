"""HDF5 storage with group structure and provenance metadata.

The pipeline HDF5 file follows a hierarchical group layout::

    /
    +-- raw/           Element maps as-loaded (float32, gzip)
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
sequentially.  Matching ``load_*`` functions read data back for pipeline
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
    from karak.config import (
        ClusterConfig,
        DenoiseConfig,
        MaskConfig,
        NormalizeConfig,
        PipelineConfig,
    )

from karak.config import get_software_versions

logger = logging.getLogger(__name__)

# Standard group layout for v1.1
_GROUPS = ["raw", "bse", "masks", "denoised", "normalized", "clusters"]


def create_pipeline_hdf5(path: str | Path, config: PipelineConfig) -> None:
    """Create an HDF5 file with standard group structure and provenance.

    Parameters
    ----------
    path : str or Path
        Path for the new HDF5 file.
    config : PipelineConfig
        Pipeline configuration to embed as a YAML attribute.
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
            config.model_dump(), default_flow_style=False, sort_keys=False
        )
        f.attrs["pipeline_config"] = config_yaml

    logger.info("Created HDF5: %s with groups %s", path, _GROUPS)


def save_raw_data(
    h5_path: str | Path,
    elements_dict: dict[str, np.ndarray],
    element_names: list[str],
) -> None:
    """Write element arrays to the ``raw/`` group with gzip compression.

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
            grp.create_dataset(
                name, data=elements_dict[name].astype(np.float32), compression="gzip"
            )

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
        grp.create_dataset("image", data=bse.astype(np.float32), compression="gzip")
        grp.attrs["original_shape"] = list(original_shape)
        grp.attrs["downsample_factor"] = downsample_factor
        grp.attrs["stored_shape"] = list(bse.shape)

    logger.info("Saved BSE to bse/ group, shape %s", bse.shape)


def save_mask(
    h5_path: str | Path,
    mineral_mask: np.ndarray,
    valid_mask: np.ndarray | None,
    mask_stats: dict,
    config: MaskConfig,
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
    config : MaskConfig
        Mask configuration used to generate the mask.
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["masks"]

        # Primary mineral mask
        if "mineral" in grp:
            del grp["mineral"]
        grp.create_dataset("mineral", data=mineral_mask, compression="gzip")

        # Valid region mask
        if valid_mask is not None:
            if "valid" in grp:
                del grp["valid"]
            grp.create_dataset("valid", data=valid_mask, compression="gzip")

        # Statistics as attributes
        for key, val in mask_stats.items():
            grp.attrs[key] = val

        # Mask config as YAML
        grp.attrs["mask_config"] = yaml.dump(
            config.model_dump(), default_flow_style=False, sort_keys=False
        )

    logger.info(
        "Saved mask to masks/ group: %.1f%% mineral coverage",
        mask_stats.get("coverage_pct", 0),
    )


def save_denoised_data(
    h5_path: str | Path,
    denoised_cube: np.ndarray,
    element_names: list[str],
    config: DenoiseConfig,
) -> None:
    """Write denoised raw [0,1] cube to the ``denoised/`` group in HDF5.

    Dataset written:
    - ``denoised/cube`` -- (H, W, C) float32 denoised cube, gzip compressed

    Parameters
    ----------
    h5_path : str or Path
        Path to existing HDF5 file.
    denoised_cube : np.ndarray
        (H, W, C) denoised raw [0,1] cube.
    element_names : list[str]
        Ordered element names.
    config : DenoiseConfig
        Denoising configuration used (method and parameters).
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["denoised"]
        if "cube" in grp:
            del grp["cube"]
        grp.create_dataset(
            "cube",
            data=denoised_cube.astype(np.float32),
            compression="gzip",
        )

        # Record method and all parameters
        grp.attrs["method"] = config.method
        grp.attrs["element_order"] = json.dumps(element_names)

        if config.method == "bilateral":
            grp.attrs["sigma_color"] = (
                config.sigma_color if config.sigma_color is not None else "auto"
            )
            grp.attrs["sigma_spatial"] = config.sigma_spatial
        elif config.method == "anisotropic_diffusion":
            grp.attrs["niter"] = config.niter
            grp.attrs["kappa"] = config.kappa
            grp.attrs["gamma"] = config.gamma
            grp.attrs["option"] = config.option

    logger.info(
        "Saved denoised cube %s to denoised/ group (method=%s)",
        denoised_cube.shape,
        config.method,
    )


def save_normalized_data(
    h5_path: str | Path,
    normalized_cube: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    element_names: list[str],
    config: NormalizeConfig,
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
    config : NormalizeConfig
        Normalization configuration used.
    """
    with h5py.File(h5_path, "a") as f:
        grp = f["normalized"]
        for ds_name in ["cube", "means", "stds"]:
            if ds_name in grp:
                del grp[ds_name]

        grp.create_dataset(
            "cube",
            data=normalized_cube.astype(np.float32),
            compression="gzip",
        )
        grp.create_dataset("means", data=means.astype(np.float32))
        grp.create_dataset("stds", data=stds.astype(np.float32))

        grp.attrs["method"] = config.method
        grp.attrs["element_order"] = json.dumps(element_names)
        grp.attrs["n_elements"] = len(element_names)

    logger.info(
        "Saved normalized cube %s to normalized/ group (method=%s)",
        normalized_cube.shape,
        config.method,
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
    config: ClusterConfig,
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
    config : ClusterConfig
        Clustering configuration used.
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
            grp.create_dataset(ds_name, data=data, compression="gzip")

        # Attributes
        grp.attrs["n_pca_components_used"] = n_pca_components_used
        grp.attrs["n_clusters"] = cluster_stats["n_clusters"]
        grp.attrs["n_noise"] = cluster_stats["n_noise"]
        grp.attrs["noise_pct"] = cluster_stats["noise_pct"]
        grp.attrs["n_mineral_pixels"] = cluster_stats["n_total"]
        grp.attrs["cluster_stats"] = json.dumps(cluster_stats)
        grp.attrs["cluster_config"] = yaml.dump(
            config.model_dump(), default_flow_style=False, sort_keys=False
        )
        grp.attrs["clustering_strategy"] = config.strategy

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

        tiled_grp.create_dataset("tile_ids", data=tile_ids)
        tiled_grp.create_dataset("tile_n_pixels", data=tile_n_pixels)
        tiled_grp.create_dataset("tile_n_clusters", data=tile_n_clusters)
        tiled_grp.create_dataset("tile_n_noise", data=tile_n_noise)
        tiled_grp.create_dataset("tile_n_new_phases", data=tile_n_new)

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
# Read-back functions (for pipeline resume)
# ---------------------------------------------------------------------------

# Maps HDF5 group names to pipeline stage names.
_STAGE_GROUPS: dict[str, str] = {
    "load": "raw",
    "mask": "masks",
    "denoise": "denoised",
    "normalize": "normalized",
    "cluster": "clusters",
    "identification": "clusters",
}


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


def get_completed_stages(h5_path: str | Path) -> dict[str, str | None]:
    """Check which pipeline stages have been completed in the HDF5 file.

    A stage is considered complete if its HDF5 group has a
    ``stage_completed`` attribute (a UTC ISO-8601 timestamp written by
    :func:`mark_stage_complete`).

    Returns
    -------
    dict[str, str | None]
        Mapping of stage name -> completion timestamp (or *None*).
    """
    path = Path(h5_path)
    if not path.exists():
        return {stage: None for stage in _STAGE_GROUPS}

    result: dict[str, str | None] = {}
    with h5py.File(path, "r") as f:
        for stage, grp_name in _STAGE_GROUPS.items():
            if grp_name in f and "stage_completed" in f[grp_name].attrs:
                result[stage] = str(f[grp_name].attrs["stage_completed"])
            else:
                result[stage] = None
    return result


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


def mark_stage_complete(h5_path: str | Path, stage_name: str) -> None:
    """Write a UTC timestamp to mark *stage_name* as complete.

    Parameters
    ----------
    stage_name : str
        One of ``"load"``, ``"mask"``, ``"denoise"``, ``"normalize"``, ``"cluster"``.
    """
    if stage_name not in _STAGE_GROUPS:
        raise ValueError(
            f"Unknown stage {stage_name!r}; expected one of {list(_STAGE_GROUPS)}"
        )
    grp_name = _STAGE_GROUPS[stage_name]
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with h5py.File(h5_path, "a") as f:
        f[grp_name].attrs["stage_completed"] = ts
        f.attrs["last_modified"] = ts
    logger.info("Marked stage %s complete at %s", stage_name, ts)
