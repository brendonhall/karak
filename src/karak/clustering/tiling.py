"""Tiled progressive HDBSCAN with phase registry unification.

Divides the image into spatial tiles, runs HDBSCAN independently per tile,
progressively builds a phase library via cosine-similarity fingerprint
matching, and unifies labels with a final k-NN pass.  This approach:

- Keeps memory under control (no O(N^2) distance matrix on the full image)
- Discovers locally-concentrated rare phases that would be absorbed as noise
  in a global HDBSCAN run

Use ``run_tiled_hdbscan()`` as the main entry point.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier

if TYPE_CHECKING:
    from karak.config import ClusterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TileSpec:
    """Spatial bounds and mineral pixel indices for one tile."""

    tile_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    pixel_indices: np.ndarray  # indices into global mineral_indices


@dataclass
class PhaseEntry:
    """One entry in the global phase registry."""

    global_id: int
    mean_fingerprint: np.ndarray  # (C,) mean denoised element values
    n_pixels: int
    discovered_in_tile: int
    tile_contributions: dict[int, int] = field(default_factory=dict)


@dataclass
class TileResult:
    """Per-tile diagnostic summary."""

    tile_id: int
    n_pixels: int
    n_clusters: int
    n_noise: int
    local_labels: np.ndarray  # raw HDBSCAN labels for this tile
    merge_map: dict[int, int]  # local_label -> global_id
    new_phases: list[int]  # global_ids of phases discovered in this tile


# ---------------------------------------------------------------------------
# Tile grid computation
# ---------------------------------------------------------------------------


def compute_tile_grid(
    mineral_indices: np.ndarray,
    image_shape: tuple[int, int],
    tile_size: int,
    min_tile_pixels: int,
) -> list[TileSpec]:
    """Divide the image into a fixed grid and assign mineral pixels to tiles.

    Parameters
    ----------
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 (row, col) coordinates.
    image_shape : tuple[int, int]
        (H, W) of the full image.
    tile_size : int
        Side length of each square tile in pixels.
    min_tile_pixels : int
        Tiles with fewer mineral pixels are skipped.

    Returns
    -------
    list[TileSpec]
        Tile specifications in row-major order, excluding empty/small tiles.
    """
    H, W = image_shape
    rows = mineral_indices[:, 0]
    cols = mineral_indices[:, 1]

    tiles: list[TileSpec] = []
    tile_id = 0

    for r0 in range(0, H, tile_size):
        r1 = min(r0 + tile_size, H)
        for c0 in range(0, W, tile_size):
            c1 = min(c0 + tile_size, W)

            mask = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
            idx = np.where(mask)[0]

            if len(idx) < min_tile_pixels:
                continue

            tiles.append(
                TileSpec(
                    tile_id=tile_id,
                    row_start=r0,
                    row_end=r1,
                    col_start=c0,
                    col_end=c1,
                    pixel_indices=idx,
                )
            )
            tile_id += 1

    logger.info(
        "Tile grid: %d tiles of size %d (image %dx%d, %d mineral pixels, "
        "min_tile_pixels=%d)",
        len(tiles), tile_size, H, W, len(mineral_indices), min_tile_pixels,
    )
    return tiles


# ---------------------------------------------------------------------------
# Per-tile fingerprint computation
# ---------------------------------------------------------------------------


def compute_tile_fingerprints(
    denoised_cube: np.ndarray,
    mineral_indices: np.ndarray,
    tile_pixel_indices: np.ndarray,
    local_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute mean chemical fingerprint for each non-noise cluster in a tile.

    Parameters
    ----------
    denoised_cube : np.ndarray
        (H, W, C) float32 raw denoised element cube [0, 1].
    mineral_indices : np.ndarray
        (N_mineral, 2) global (row, col) coordinates.
    tile_pixel_indices : np.ndarray
        Indices into mineral_indices for this tile's pixels.
    local_labels : np.ndarray
        HDBSCAN labels for this tile's pixels (-1 = noise).

    Returns
    -------
    dict[int, np.ndarray]
        local_label -> (C,) mean fingerprint vector.
    """
    tile_coords = mineral_indices[tile_pixel_indices]
    tile_spectra = denoised_cube[tile_coords[:, 0], tile_coords[:, 1], :]

    fingerprints: dict[int, np.ndarray] = {}
    for label in np.unique(local_labels):
        if label == -1:
            continue
        mask = local_labels == label
        fingerprints[int(label)] = np.mean(tile_spectra[mask], axis=0)

    return fingerprints


# ---------------------------------------------------------------------------
# Phase registry matching
# ---------------------------------------------------------------------------


def match_clusters_to_registry(
    tile_fingerprints: dict[int, np.ndarray],
    phase_registry: list[PhaseEntry],
    merge_threshold: float,
) -> dict[int, int | None]:
    """Match tile clusters to existing registry entries via cosine similarity.

    Parameters
    ----------
    tile_fingerprints : dict[int, np.ndarray]
        local_label -> (C,) mean fingerprint from ``compute_tile_fingerprints``.
    phase_registry : list[PhaseEntry]
        Current global phase registry.
    merge_threshold : float
        Cosine similarity threshold for matching.

    Returns
    -------
    dict[int, int | None]
        local_label -> global_id (matched) or None (new phase).
    """
    match_map: dict[int, int | None] = {}

    for local_label, fp in tile_fingerprints.items():
        best_sim = -1.0
        best_id: int | None = None

        for entry in phase_registry:
            sim = 1.0 - cosine(fp, entry.mean_fingerprint)
            if sim > best_sim:
                best_sim = sim
                best_id = entry.global_id

        if best_sim >= merge_threshold:
            match_map[local_label] = best_id
        else:
            match_map[local_label] = None

    return match_map


# ---------------------------------------------------------------------------
# k-NN noise/unassigned reassignment
# ---------------------------------------------------------------------------


def final_knn_assign(
    pca_features: np.ndarray,
    raw_labels: np.ndarray,
    noise_reassign_k: int,
) -> np.ndarray:
    """Assign all unassigned (-1) pixels to nearest phase via k-NN.

    Parameters
    ----------
    pca_features : np.ndarray
        (N_mineral, n_components) PCA-transformed features.
    raw_labels : np.ndarray
        (N_mineral,) int32 labels with -1 for unassigned.
    noise_reassign_k : int
        Number of neighbors for kNN.

    Returns
    -------
    cleaned_labels : np.ndarray
        (N_mineral,) int32 labels with no -1 values.
    """
    assigned_mask = raw_labels >= 0
    n_assigned = int(np.sum(assigned_mask))
    n_mineral = len(raw_labels)
    n_unassigned = n_mineral - n_assigned

    logger.info(
        "k-NN assignment: %d assigned, %d unassigned (%.1f%%)",
        n_assigned, n_unassigned, 100.0 * n_unassigned / n_mineral,
    )

    if n_assigned > 0 and n_unassigned > 0:
        knn = KNeighborsClassifier(
            n_neighbors=noise_reassign_k,
            weights="distance",
            n_jobs=-1,
        )
        knn.fit(pca_features[assigned_mask], raw_labels[assigned_mask])

        cleaned_labels = raw_labels.copy()
        unassigned_preds = knn.predict(pca_features[~assigned_mask])
        cleaned_labels[~assigned_mask] = unassigned_preds.astype(np.int32)
    elif n_assigned == n_mineral:
        cleaned_labels = raw_labels.copy()
    else:
        cleaned_labels = np.zeros(n_mineral, dtype=np.int32)

    return cleaned_labels


# ---------------------------------------------------------------------------
# Pass 2: rare phase reclustering
# ---------------------------------------------------------------------------


def recluster_unassigned(
    pca_features: np.ndarray,
    raw_labels: np.ndarray,
    denoised_cube: np.ndarray | str,
    mineral_indices: np.ndarray,
    phase_registry: list[PhaseEntry],
    config: ClusterConfig,
) -> tuple[np.ndarray, list[PhaseEntry], int, int]:
    """Recluster unassigned pixels to discover rare phases (Pass 2).

    Collects all pixels with label == -1, runs HDBSCAN with more sensitive
    parameters (lower min_cluster_size), and matches discovered clusters
    against the existing phase registry. Novel clusters become new phases.

    The denoised cube is only needed for fingerprinting after HDBSCAN
    completes.  Pass an HDF5 path string instead of an array to defer
    loading until needed (saves ~1.9 GB during HDBSCAN fitting).

    Parameters
    ----------
    pca_features : np.ndarray
        (N_mineral, n_components) PCA-transformed features.
    raw_labels : np.ndarray
        (N_mineral,) int32 labels from Pass 1 (-1 = unassigned).
    denoised_cube : np.ndarray or str
        (H, W, C) float32 denoised element cube, OR path to HDF5 file
        to load it on-demand (memory-efficient for large images).
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 (row, col) coordinates.
    phase_registry : list[PhaseEntry]
        Phase registry from Pass 1 (will be extended in-place).
    config : ClusterConfig
        Full clustering config (uses rare_phase sub-config).

    Returns
    -------
    updated_labels : np.ndarray
        (N_mineral,) int32 labels with rare phases assigned (still has -1 for
        remaining noise).
    phase_registry : list[PhaseEntry]
        Updated phase registry with any new rare phases appended.
    n_rare_phases : int
        Number of new rare phases discovered.
    n_still_unassigned : int
        Number of pixels still unassigned after Pass 2.
    """
    import gc

    from karak.clustering.hdbscan_cluster import run_hdbscan
    from karak.config import HDBSCANConfig

    rare_cfg = config.rare_phase
    unassigned_mask = raw_labels == -1
    n_unassigned = int(np.sum(unassigned_mask))

    if n_unassigned == 0:
        logger.info("No unassigned pixels for Pass 2")
        return raw_labels.copy(), phase_registry, 0, 0

    logger.info(
        "Pass 2: reclustering %d unassigned pixels (min_cluster_size=%d)",
        n_unassigned, rare_cfg.min_cluster_size,
    )

    # Build HDBSCAN config for Pass 2
    pass2_hdb = HDBSCANConfig(
        min_cluster_size=rare_cfg.min_cluster_size,
        min_samples=rare_cfg.min_samples,
        subsample_n=rare_cfg.subsample_n,
        noise_reassign_k=config.hdbscan.noise_reassign_k,
        random_state=config.hdbscan.random_state,
    )

    # Extract unassigned pixel features
    unassigned_indices = np.where(unassigned_mask)[0]
    unassigned_features = pca_features[unassigned_indices]

    # Run HDBSCAN on unassigned pixels
    pass2_labels, pass2_probs, _ = run_hdbscan(unassigned_features, pass2_hdb)
    del unassigned_features
    gc.collect()

    n_pass2_clusters = len(set(pass2_labels.tolist()) - {-1})
    n_pass2_noise = int(np.sum(pass2_labels == -1))
    logger.info(
        "Pass 2 HDBSCAN: %d clusters, %d noise from %d pixels",
        n_pass2_clusters, n_pass2_noise, n_unassigned,
    )

    if n_pass2_clusters == 0:
        logger.info("Pass 2: no clusters found in unassigned pixels")
        return raw_labels.copy(), phase_registry, 0, n_unassigned

    # Load denoised cube on-demand if an HDF5 path was given
    if isinstance(denoised_cube, (str, Path)):
        from karak.io.storage import load_denoised_data

        logger.info("Loading denoised cube from %s for fingerprinting", denoised_cube)
        denoised_cube, _ = load_denoised_data(denoised_cube)

    # Compute fingerprints for Pass 2 clusters — extract only unassigned spectra
    unassigned_coords = mineral_indices[unassigned_indices]
    unassigned_spectra = denoised_cube[unassigned_coords[:, 0], unassigned_coords[:, 1], :]
    del denoised_cube
    gc.collect()

    pass2_fingerprints: dict[int, np.ndarray] = {}
    pass2_pixel_counts: dict[int, int] = {}
    for label in np.unique(pass2_labels):
        if label == -1:
            continue
        mask = pass2_labels == label
        pass2_fingerprints[int(label)] = np.mean(unassigned_spectra[mask], axis=0)
        pass2_pixel_counts[int(label)] = int(np.sum(mask))

    # Match against existing registry
    merge_threshold = rare_cfg.merge_threshold
    if merge_threshold is None:
        merge_threshold = config.tiled.merge_threshold

    match_map = match_clusters_to_registry(
        pass2_fingerprints, phase_registry, merge_threshold,
    )

    # Assign labels: matched -> existing global_id, novel -> new global_id
    next_global_id = max(e.global_id for e in phase_registry) + 1 if phase_registry else 0
    updated_labels = raw_labels.copy()
    n_rare_phases = 0

    for local_label, global_id in match_map.items():
        local_mask = pass2_labels == local_label
        global_idx = unassigned_indices[local_mask]
        n_pixels_local = pass2_pixel_counts[local_label]

        if global_id is not None:
            # Matched existing phase
            updated_labels[global_idx] = global_id
            entry = next(e for e in phase_registry if e.global_id == global_id)
            total_pixels = entry.n_pixels + n_pixels_local
            entry.mean_fingerprint = (
                entry.mean_fingerprint * entry.n_pixels
                + pass2_fingerprints[local_label] * n_pixels_local
            ) / total_pixels
            entry.n_pixels = total_pixels
            logger.info(
                "Pass 2: cluster %d matched existing phase %d (%d pixels)",
                local_label, global_id, n_pixels_local,
            )
        else:
            # Novel rare phase
            gid = next_global_id
            next_global_id += 1
            updated_labels[global_idx] = gid
            n_rare_phases += 1

            phase_registry.append(
                PhaseEntry(
                    global_id=gid,
                    mean_fingerprint=pass2_fingerprints[local_label].copy(),
                    n_pixels=n_pixels_local,
                    discovered_in_tile=-1,  # -1 = discovered in Pass 2
                    tile_contributions={-1: n_pixels_local},
                )
            )
            logger.info(
                "Pass 2: cluster %d -> new rare phase %d (%d pixels)",
                local_label, gid, n_pixels_local,
            )

    n_still_unassigned = int(np.sum(updated_labels == -1))
    logger.info(
        "Pass 2 complete: %d rare phases, %d matched existing, %d still unassigned",
        n_rare_phases, n_pass2_clusters - n_rare_phases, n_still_unassigned,
    )

    return updated_labels, phase_registry, n_rare_phases, n_still_unassigned


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_tiled_hdbscan(
    pca_features: np.ndarray,
    mineral_indices: np.ndarray,
    image_shape: tuple[int, int],
    denoised_cube: np.ndarray,
    config: ClusterConfig,
    progress_callback: Callable[[int, int, TileResult], None] | None = None,
    skip_knn: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[TileResult], list[PhaseEntry]]:
    """Run tiled progressive HDBSCAN with phase registry unification.

    Parameters
    ----------
    pca_features : np.ndarray
        (N_mineral, n_components) PCA-transformed features.
    mineral_indices : np.ndarray
        (N_mineral, 2) int32 (row, col) coordinates.
    image_shape : tuple[int, int]
        (H, W) of the full image.
    denoised_cube : np.ndarray
        (H, W, C) float32 denoised element cube [0, 1].
    config : ClusterConfig
        Full clustering config (pca, hdbscan, tiled sub-configs).

    Returns
    -------
    raw_labels : np.ndarray
        (N_mineral,) int32 global labels (-1 for noise pixels).
    cleaned_labels : np.ndarray
        (N_mineral,) int32 k-NN unified labels (no -1).
    probabilities : np.ndarray
        (N_mineral,) float32 HDBSCAN probabilities (0.0 for k-NN-only pixels).
    tile_results : list[TileResult]
        Per-tile diagnostic summaries.
    phase_registry : list[PhaseEntry]
        Final phase registry.
    """
    from karak.clustering.hdbscan_cluster import run_hdbscan

    tiled_cfg = config.tiled
    hdb_cfg = config.hdbscan
    n_mineral = len(mineral_indices)

    # Resolve min_tile_pixels
    min_tile_pixels = tiled_cfg.min_tile_pixels
    if min_tile_pixels is None:
        min_tile_pixels = 2 * hdb_cfg.min_cluster_size

    # Step 1: Compute tile grid
    tiles = compute_tile_grid(
        mineral_indices, image_shape, tiled_cfg.tile_size, min_tile_pixels,
    )

    # Initialise output arrays
    raw_labels = np.full(n_mineral, -1, dtype=np.int32)
    probabilities = np.zeros(n_mineral, dtype=np.float32)

    phase_registry: list[PhaseEntry] = []
    tile_results: list[TileResult] = []
    next_global_id = 0

    # Step 2: Process each tile
    min_clusters = tiled_cfg.min_clusters_per_tile
    n_deferred_tiles = 0

    for tile in tiles:
        tile_features = pca_features[tile.pixel_indices]

        # Run per-tile HDBSCAN (reuse existing function)
        tile_labels, tile_probs, _ = run_hdbscan(tile_features, hdb_cfg)

        n_noise = int(np.sum(tile_labels == -1))
        n_clusters = len(set(tile_labels.tolist()) - {-1})

        # Hybrid fallback: if tile produced too few clusters, defer its
        # pixels to the final k-NN pass rather than forcing a label.
        # This prevents single-class tile artifacts.
        if n_clusters < min_clusters:
            logger.info(
                "Tile %d/%d: %d pixels, only %d clusters (< %d) — "
                "deferring to k-NN",
                tile.tile_id + 1, len(tiles), len(tile.pixel_indices),
                n_clusters, min_clusters,
            )
            n_deferred_tiles += 1

            tile_results.append(
                TileResult(
                    tile_id=tile.tile_id,
                    n_pixels=len(tile.pixel_indices),
                    n_clusters=n_clusters,
                    n_noise=len(tile.pixel_indices),  # all deferred
                    local_labels=tile_labels,
                    merge_map={},
                    new_phases=[],
                )
            )

            if progress_callback is not None:
                progress_callback(tile.tile_id, len(tiles), tile_results[-1])
            continue

        # Compute fingerprints for non-noise clusters
        tile_fps = compute_tile_fingerprints(
            denoised_cube, mineral_indices, tile.pixel_indices, tile_labels,
        )

        # Match to phase registry
        match_map = match_clusters_to_registry(
            tile_fps, phase_registry, tiled_cfg.merge_threshold,
        )

        # Build local->global label map and update registry
        merge_map: dict[int, int] = {}
        new_phases: list[int] = []

        for local_label, global_id in match_map.items():
            n_pixels_local = int(np.sum(tile_labels == local_label))

            if global_id is not None:
                # Matched existing phase — update registry entry
                merge_map[local_label] = global_id
                entry = next(e for e in phase_registry if e.global_id == global_id)

                # Update running mean fingerprint
                total_pixels = entry.n_pixels + n_pixels_local
                entry.mean_fingerprint = (
                    entry.mean_fingerprint * entry.n_pixels
                    + tile_fps[local_label] * n_pixels_local
                ) / total_pixels
                entry.n_pixels = total_pixels
                entry.tile_contributions[tile.tile_id] = n_pixels_local
            else:
                # New phase
                gid = next_global_id
                next_global_id += 1
                merge_map[local_label] = gid
                new_phases.append(gid)

                phase_registry.append(
                    PhaseEntry(
                        global_id=gid,
                        mean_fingerprint=tile_fps[local_label].copy(),
                        n_pixels=n_pixels_local,
                        discovered_in_tile=tile.tile_id,
                        tile_contributions={tile.tile_id: n_pixels_local},
                    )
                )

        # Assign global labels and probabilities for this tile
        for local_label, global_id in merge_map.items():
            local_mask = tile_labels == local_label
            global_idx = tile.pixel_indices[local_mask]
            raw_labels[global_idx] = global_id
            probabilities[global_idx] = tile_probs[local_mask]

        tile_results.append(
            TileResult(
                tile_id=tile.tile_id,
                n_pixels=len(tile.pixel_indices),
                n_clusters=n_clusters,
                n_noise=n_noise,
                local_labels=tile_labels,
                merge_map=merge_map,
                new_phases=new_phases,
            )
        )

        logger.info(
            "Tile %d/%d: %d pixels, %d clusters, %d noise, "
            "%d new phases, %d matched",
            tile.tile_id + 1, len(tiles), len(tile.pixel_indices),
            n_clusters, n_noise, len(new_phases),
            n_clusters - len(new_phases),
        )

        if progress_callback is not None:
            progress_callback(tile.tile_id, len(tiles), tile_results[-1])

    # Step 3: Final k-NN unification (unless caller wants to do it later)
    if skip_knn:
        logger.info(
            "Skipping k-NN unification (skip_knn=True): %d assigned, "
            "%d unassigned (%.1f%%)",
            int(np.sum(raw_labels >= 0)),
            int(np.sum(raw_labels == -1)),
            100.0 * np.sum(raw_labels == -1) / n_mineral,
        )
        cleaned_labels = raw_labels.copy()
    else:
        cleaned_labels = final_knn_assign(
            pca_features, raw_labels, hdb_cfg.noise_reassign_k,
        )

    logger.info(
        "Tiled HDBSCAN complete: %d phases, %d tiles processed",
        len(phase_registry), len(tile_results),
    )

    return raw_labels, cleaned_labels, probabilities, tile_results, phase_registry
