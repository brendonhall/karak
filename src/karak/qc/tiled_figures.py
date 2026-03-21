"""QC figure generation for tiled HDBSCAN clustering.

All functions use the ``Agg`` backend so they work headlessly in the CLI
runner.  Each returns the path to the saved PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from karak.clustering.tiling import PhaseEntry, TileResult, TileSpec  # noqa: E402

logger = logging.getLogger(__name__)


def generate_tile_grid_overlay(
    bse: np.ndarray,
    tiles: list[TileSpec],
    tile_results: list[TileResult],
    figure_dir: str | Path,
) -> str:
    """BSE image with tile grid lines and per-tile cluster count annotations.

    Parameters
    ----------
    bse : np.ndarray
        (H, W) float32 BSE image.
    tiles : list[TileSpec]
        Tile specifications from ``compute_tile_grid``.
    tile_results : list[TileResult]
        Per-tile results from ``run_tiled_hdbscan``.
    figure_dir : str or Path
        Output directory.

    Returns
    -------
    str
        Path to the saved ``tile_grid_overlay.png``.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Build tile_id -> TileResult lookup
    results_by_id = {tr.tile_id: tr for tr in tile_results}

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(bse, cmap="gray", interpolation="none")

    for tile in tiles:
        # Draw tile rectangle
        r0, r1 = tile.row_start, tile.row_end
        c0, c1 = tile.col_start, tile.col_end
        rect = plt.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=0.8, edgecolor="cyan", facecolor="none", alpha=0.7,
        )
        ax.add_patch(rect)

        # Annotate with cluster count
        tr = results_by_id.get(tile.tile_id)
        if tr is not None:
            cx = (c0 + c1) / 2
            cy = (r0 + r1) / 2
            ax.text(
                cx, cy, str(tr.n_clusters),
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5),
            )

    ax.set_title(
        f"Tile Grid Overlay ({len(tiles)} tiles, "
        f"numbers = clusters per tile)"
    )
    ax.axis("off")

    plt.tight_layout()
    out_path = str(figure_dir / "tile_grid_overlay.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tile grid overlay: %s", out_path)
    return out_path


def generate_phase_discovery_chart(
    tile_results: list[TileResult],
    phase_registry: list[PhaseEntry],
    figure_dir: str | Path,
) -> str:
    """Cumulative phase count vs tile number with discovery event markers.

    Parameters
    ----------
    tile_results : list[TileResult]
        Per-tile results from ``run_tiled_hdbscan``.
    phase_registry : list[PhaseEntry]
        Final phase registry.
    figure_dir : str or Path
        Output directory.

    Returns
    -------
    str
        Path to the saved ``phase_discovery_chart.png``.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Sort tile results by tile_id (should already be ordered)
    sorted_results = sorted(tile_results, key=lambda tr: tr.tile_id)

    tile_ids = [tr.tile_id for tr in sorted_results]
    cumulative_phases = []
    running_total = 0
    discovery_tiles = []
    discovery_counts = []

    for tr in sorted_results:
        running_total += len(tr.new_phases)
        cumulative_phases.append(running_total)
        if tr.new_phases:
            discovery_tiles.append(tr.tile_id)
            discovery_counts.append(running_total)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Cumulative line
    ax.plot(tile_ids, cumulative_phases, "o-", color="steelblue", markersize=4)
    ax.fill_between(tile_ids, cumulative_phases, alpha=0.15, color="steelblue")

    # Highlight discovery events
    if discovery_tiles:
        ax.scatter(
            discovery_tiles, discovery_counts,
            color="red", s=60, zorder=5, label="New phase discovered",
        )

    ax.set_xlabel("Tile ID")
    ax.set_ylabel("Cumulative Phase Count")
    ax.set_title(
        f"Phase Discovery Progress ({len(phase_registry)} phases from "
        f"{len(tile_results)} tiles)"
    )
    ax.legend(fontsize=9)

    # Integer y ticks
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    out_path = str(figure_dir / "phase_discovery_chart.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved phase discovery chart: %s", out_path)
    return out_path
