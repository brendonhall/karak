"""QC figure generation for each pipeline stage.

All functions use the ``Agg`` backend so they work headlessly in the CLI
runner.  Each returns the path to the saved PNG.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats as scipy_stats  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mask QC (notebook cells 9-10)
# ---------------------------------------------------------------------------


def generate_mask_qc(
    bse: np.ndarray,
    mineral_mask: np.ndarray,
    valid_mask: np.ndarray | None,
    mask_stats: dict,
    tima_path: str | None,
    downsample_factor: int,
    header_trim_px: int,
    figure_dir: str | Path,
) -> str:
    """Three-panel mask QC: BSE+mask overlay, TIMA reference, boundaries.

    Returns the path to the saved PNG.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_panels = 3 if tima_path and os.path.isfile(tima_path) else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8))

    # Panel 1: BSE with mineral mask overlay (red = background)
    overlay = np.stack([bse] * 3, axis=-1)
    bse_max = overlay.max()
    if bse_max > 0:
        overlay = overlay / bse_max
    overlay[~mineral_mask, 0] = np.clip(overlay[~mineral_mask, 0] + 0.4, 0, 1)
    overlay[~mineral_mask, 1] = overlay[~mineral_mask, 1] * 0.5
    overlay[~mineral_mask, 2] = overlay[~mineral_mask, 2] * 0.5

    axes[0].imshow(overlay, interpolation="none")
    coverage = mask_stats.get("coverage_pct", 0)
    axes[0].set_title(f"BSE + Mineral Mask\n(red = background, {coverage:.1f}% mineral)")
    axes[0].axis("off")

    # Panel 2: TIMA phase map (if available)
    if n_panels == 3:
        import imageio.v2 as imageio_v2
        from skimage.transform import downscale_local_mean

        tima_raw = imageio_v2.imread(tima_path)
        factor = downsample_factor
        trim = header_trim_px // factor
        if tima_raw.ndim == 3:
            tima_ds = np.stack(
                [
                    downscale_local_mean(
                        tima_raw[:, :, c].astype(np.float32), (factor, factor)
                    )
                    for c in range(tima_raw.shape[2])
                ],
                axis=-1,
            )
        else:
            tima_ds = downscale_local_mean(
                tima_raw.astype(np.float32), (factor, factor)
            )
        tima_trimmed = tima_ds[trim:, :]
        h, w = bse.shape
        tima_display = tima_trimmed[:h, :w]
        if tima_display.max() > 1.0:
            tima_display = tima_display / 255.0

        axes[1].imshow(tima_display, interpolation="none")
        axes[1].set_title("TIMA Phase Map (reference)")
        axes[1].axis("off")

    # Last panel: BSE with boundaries
    ax_boundary = axes[-1]
    ax_boundary.imshow(bse, cmap="gray", interpolation="none")
    if valid_mask is not None:
        ax_boundary.contour(
            valid_mask.astype(float), levels=[0.5], colors="lime", linewidths=1.0
        )
    ax_boundary.contour(
        mineral_mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.5
    )
    ax_boundary.set_title("BSE + Valid Polygon (green) + Mineral Boundary (cyan)")
    ax_boundary.axis("off")

    plt.tight_layout()
    out_path = str(figure_dir / "mask_qc.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved mask QC figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Denoise QC (notebook cells 18-19)
# ---------------------------------------------------------------------------


def generate_denoise_qc(
    cube: np.ndarray,
    denoised: np.ndarray,
    bse: np.ndarray,
    mineral_mask: np.ndarray,
    element_names: list[str],
    denoise_method: str,
    figure_dir: str | Path,
) -> str:
    """Before/after comparison with BSE edge overlay for key elements.

    Returns the path to the saved PNG.
    """
    from skimage.filters import sobel

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Pick QC elements
    preferred = ["Fe-K", "Ca", "Mg", "Si"]
    qc_indices = [element_names.index(e) for e in preferred if e in element_names]
    if not qc_indices:
        qc_indices = list(range(min(4, len(element_names))))

    # Find zoom region centred on mineral area
    H, W = bse.shape
    ys, xs = np.where(mineral_mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    zy0 = max(0, cy - 150)
    zy1 = min(H, cy + 150)
    zx0 = max(0, cx - 200)
    zx1 = min(W, cx + 200)

    # BSE edges
    bse_region = bse[zy0:zy1, zx0:zx1]
    bse_norm = (bse_region - bse_region.min()) / (bse_region.max() - bse_region.min() + 1e-10)
    bse_edges = sobel(bse_norm)
    edge_binary = bse_edges > np.percentile(bse_edges, 90)

    n_cols = len(qc_indices)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    region_mask = mineral_mask[zy0:zy1, zx0:zx1]
    for col, ch_idx in enumerate(qc_indices):
        ch_name = element_names[ch_idx]
        ch_vals = cube[zy0:zy1, zx0:zx1, ch_idx][region_mask]
        if len(ch_vals) > 0:
            vmin = np.percentile(ch_vals, 2)
            vmax = np.percentile(ch_vals, 98)
        else:
            vmin, vmax = 0, 1

        axes[0, col].imshow(
            cube[zy0:zy1, zx0:zx1, ch_idx], cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[0, col].set_title(f"{ch_name} - Raw", fontsize=10)
        axes[0, col].axis("off")

        axes[1, col].imshow(
            denoised[zy0:zy1, zx0:zx1, ch_idx], cmap="viridis", vmin=vmin, vmax=vmax
        )
        axes[1, col].contour(edge_binary, levels=[0.5], colors="red", linewidths=0.5)
        axes[1, col].set_title(f"{ch_name} - Denoised ({denoise_method})", fontsize=10)
        axes[1, col].axis("off")

    plt.suptitle(
        "Denoise QC: Before/After with BSE Boundary Overlay (red)", fontsize=13
    )
    plt.tight_layout()
    out_path = str(figure_dir / "denoise_qc.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved denoise QC figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Z-score histograms (notebook cell 24)
# ---------------------------------------------------------------------------


def generate_zscore_histograms(
    normalized: np.ndarray,
    mineral_mask: np.ndarray,
    element_names: list[str],
    figure_dir: str | Path,
) -> str:
    """Per-channel z-score distribution grid with N(0,1) overlay.

    Returns the path to the saved PNG.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_elements = len(element_names)
    n_cols = min(5, n_elements)
    n_rows = int(np.ceil(n_elements / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    mineral_normalized = normalized[mineral_mask]

    for i, name in enumerate(element_names):
        ax = axes_flat[i]
        ch_data = mineral_normalized[:, i]

        ax.hist(
            ch_data, bins=100, density=True, alpha=0.7,
            color="steelblue", edgecolor="none",
        )
        x_norm = np.linspace(ch_data.min(), ch_data.max(), 200)
        ax.plot(
            x_norm, scipy_stats.norm.pdf(x_norm, 0, 1),
            "r-", linewidth=1.5, label="N(0,1)",
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("z-score", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_elements, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.suptitle("Z-Score Distributions per Element (mineral pixels)", fontsize=14)
    plt.tight_layout()
    out_path = str(figure_dir / "zscore_histograms.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved z-score histogram figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Correlation matrix (notebook cell 26)
# ---------------------------------------------------------------------------


def generate_correlation_matrix(
    normalized: np.ndarray,
    mineral_mask: np.ndarray,
    element_names: list[str],
    figure_dir: str | Path,
) -> str:
    """Element-element Pearson *r* heatmap.

    Returns the path to the saved PNG.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    mineral_normalized = normalized[mineral_mask]
    n_elements = len(element_names)
    corr_matrix = np.corrcoef(mineral_normalized.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="none")

    ax.set_xticks(range(n_elements))
    ax.set_yticks(range(n_elements))
    ax.set_xticklabels(element_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(element_names, fontsize=9)

    for i in range(n_elements):
        for j in range(n_elements):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_title("Element-Element Correlation Matrix (z-score normalized, mineral pixels)")

    plt.tight_layout()
    out_path = str(figure_dir / "zscore_correlation.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation matrix figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# PCA Scree Plot (Phase 2)
# ---------------------------------------------------------------------------


def generate_scree_plot(
    explained_variance_ratio: np.ndarray,
    n_selected: int | None,
    figure_dir: str | Path,
) -> str:
    """PCA scree plot with cumulative explained variance.

    Parameters
    ----------
    explained_variance_ratio : np.ndarray
        Per-component explained variance ratio from PCA.
    n_selected : int or None
        Number of components selected (vertical line). None = no line.
    figure_dir : str or Path
        Output directory.

    Returns the path to the saved PNG.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    cumvar = np.cumsum(explained_variance_ratio) * 100
    n_components = len(explained_variance_ratio)
    x = np.arange(1, n_components + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: individual variance
    ax1.bar(x, explained_variance_ratio * 100, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Individual Component Variance")
    ax1.set_xticks(x)

    # Right: cumulative variance
    ax2.plot(x, cumvar, "o-", color="steelblue", markersize=5)
    ax2.fill_between(x, cumvar, alpha=0.2, color="steelblue")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xticks(x)
    ax2.set_ylim(0, 105)
    ax2.axhline(y=95, color="red", linestyle="--", alpha=0.5, label="95%")
    ax2.axhline(y=90, color="orange", linestyle="--", alpha=0.5, label="90%")

    if n_selected is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=n_selected, color="green", linestyle="-", linewidth=2, alpha=0.7)
        ax2.annotate(
            f"Selected: {n_selected} ({cumvar[n_selected-1]:.1f}%)",
            xy=(n_selected, cumvar[n_selected - 1]),
            xytext=(n_selected + 1, cumvar[n_selected - 1] - 10),
            arrowprops=dict(arrowstyle="->", color="green"),
            fontsize=10, color="green",
        )

    ax2.legend(fontsize=9)
    plt.suptitle("PCA Scree Analysis", fontsize=14)
    plt.tight_layout()
    out_path = str(figure_dir / "pca_scree.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PCA scree plot: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Phase Map (Phase 2)
# ---------------------------------------------------------------------------


def generate_phase_map(
    raw_label_image: np.ndarray,
    cleaned_label_image: np.ndarray,
    bse: np.ndarray,
    cluster_stats: dict,
    figure_dir: str | Path,
) -> str:
    """Side-by-side phase maps: raw HDBSCAN (with noise) and cleaned (kNN-assigned).

    Parameters
    ----------
    raw_label_image : np.ndarray
        (H, W) int32 label image. -1 = background/noise.
    cleaned_label_image : np.ndarray
        (H, W) int32 label image. -1 = background only (no noise).
    bse : np.ndarray
        (H, W) float32 BSE image for context.
    cluster_stats : dict
        Summary statistics from compute_cluster_stats().
    figure_dir : str or Path
        Output directory.

    Returns the path to the saved PNG.
    """
    from matplotlib.colors import ListedColormap

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = cluster_stats["n_clusters"]

    # Build a categorical colormap: distinct colors for each cluster + gray for background
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    # Map: -1 -> gray (background/noise), 0..n-1 -> distinct colors
    # Shift labels so -1 maps to index 0 in colormap
    cmap_colors = [[0.85, 0.85, 0.85, 1.0]]  # gray for -1 / background
    for i in range(n_clusters):
        cmap_colors.append(list(base_colors[i % 20]))
    cmap = ListedColormap(cmap_colors)

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    # Panel 1: BSE reference
    axes[0].imshow(bse, cmap="gray", interpolation="none")
    axes[0].set_title("BSE Reference")
    axes[0].axis("off")

    # Panel 2: Raw HDBSCAN labels (noise as gray)
    display_raw = raw_label_image + 1  # shift so -1 -> 0 (gray), 0 -> 1, etc.
    axes[1].imshow(display_raw, cmap=cmap, vmin=0, vmax=n_clusters, interpolation="none")
    noise_pct = cluster_stats["noise_pct"]
    axes[1].set_title(f"HDBSCAN Raw ({n_clusters} clusters, {noise_pct:.1f}% noise)")
    axes[1].axis("off")

    # Panel 3: Cleaned labels (noise reassigned)
    display_clean = cleaned_label_image + 1
    axes[2].imshow(display_clean, cmap=cmap, vmin=0, vmax=n_clusters, interpolation="none")
    axes[2].set_title(f"Cleaned (noise reassigned via kNN)")
    axes[2].axis("off")

    plt.suptitle("Mineral Phase Maps", fontsize=14)
    plt.tight_layout()
    out_path = str(figure_dir / "phase_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved phase map figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Cluster Summary Table (Phase 2)
# ---------------------------------------------------------------------------


def generate_cluster_summary(
    cluster_stats: dict,
    figure_dir: str | Path,
) -> str:
    """Cluster size bar chart with noise fraction.

    Parameters
    ----------
    cluster_stats : dict
        Summary statistics from compute_cluster_stats().
    figure_dir : str or Path
        Output directory.

    Returns the path to the saved PNG.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    clusters = cluster_stats.get("clusters", {})
    noise_pct = cluster_stats.get("noise_pct", 0)
    n_total = cluster_stats.get("n_total", 1)

    # Sort clusters by pixel count descending
    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]["n_pixels"], reverse=True)
    labels_list = [f"Cluster {k}" for k, _ in sorted_clusters]
    counts = [v["n_pixels"] for _, v in sorted_clusters]
    pcts = [v["pct"] for _, v in sorted_clusters]

    # Add noise bar
    labels_list.append("Noise")
    counts.append(cluster_stats.get("n_noise", 0))
    pcts.append(noise_pct)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["steelblue"] * len(sorted_clusters) + ["gray"]
    bars = ax.barh(range(len(labels_list)), counts, color=colors, edgecolor="white")

    ax.set_yticks(range(len(labels_list)))
    ax.set_yticklabels(labels_list)
    ax.set_xlabel("Number of Pixels")
    ax.set_title(f"Cluster Sizes ({cluster_stats['n_clusters']} clusters, {noise_pct:.1f}% noise)")
    ax.invert_yaxis()

    # Annotate with percentages
    for i, (bar, pct) in enumerate(zip(bars, pcts)):
        ax.text(
            bar.get_width() + n_total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center", fontsize=9,
        )

    plt.tight_layout()
    out_path = str(figure_dir / "cluster_summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cluster summary figure: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Fingerprint Chart (Phase 3)
# ---------------------------------------------------------------------------


def generate_fingerprint_chart(
    fingerprints_dict: dict,
    figure_dir: str | Path,
    mineral_names: dict[int, str] | None = None,
) -> str:
    """Small-multiples bar chart of per-cluster chemical fingerprints.

    Each subplot shows the mean element intensity with +/- 1 std error bars
    for one cluster.  Elements are ordered by cross-cluster variance
    (most discriminating first).

    Parameters
    ----------
    fingerprints_dict : dict
        Output from :func:`~karak.identification.fingerprint.compute_fingerprints`.
    figure_dir : str or Path
        Output directory for the saved figure.
    mineral_names : dict[int, str] or None, optional
        Mapping of cluster label -> mineral name for subplot titles.
        If *None*, titles use "Cluster N".

    Returns
    -------
    str
        Path to the saved ``fingerprint_chart.png``.
    """
    from matplotlib.colors import ListedColormap

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fp = fingerprints_dict["fingerprints"]
    element_names = fingerprints_dict["element_names"]
    element_order = fingerprints_dict["element_order"]
    n_clusters = fingerprints_dict["n_clusters"]

    # Ordered element names for x-axis
    ordered_names = [element_names[i] for i in element_order]
    n_elements = len(ordered_names)
    x = np.arange(n_elements)

    # Grid layout
    cols = min(n_clusters, 4)
    rows = int(np.ceil(n_clusters / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False,
    )
    axes_flat = axes.flatten()

    # Consistent y-axis: 0 to max(mean + std), capped at 1.0
    y_max = 0.0
    for label, data in fp.items():
        ordered_means = data["mean"][element_order]
        ordered_stds = data["std"][element_order]
        local_max = np.max(ordered_means + ordered_stds)
        if local_max > y_max:
            y_max = local_max
    y_max = min(float(y_max) * 1.05, 1.0)

    # Tab20 colors matching phase map
    tab20 = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    sorted_labels = sorted(fp.keys())
    for idx, label in enumerate(sorted_labels):
        ax = axes_flat[idx]
        data = fp[label]
        ordered_means = data["mean"][element_order]
        ordered_stds = data["std"][element_order]

        ax.bar(
            x, ordered_means, yerr=ordered_stds,
            color=tab20[label % 20],
            edgecolor="white", linewidth=0.5,
            capsize=2, error_kw={"linewidth": 0.8},
        )
        ax.set_ylim(0, y_max)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Relative Intensity [0, 1]", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

        # Title: mineral name or cluster label
        if mineral_names and label in mineral_names:
            title = mineral_names[label]
        else:
            title = f"Cluster {label}"
        subtitle = f"n={data['n_pixels']:,}  ({data['area_pct']:.1f}%)"
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)

    # Hide unused subplots
    for idx in range(n_clusters, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Chemical Fingerprints per Phase", fontsize=13)
    plt.tight_layout()
    out_path = str(figure_dir / "fingerprint_chart.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved fingerprint chart: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Named Phase Map (Phase 3)
# ---------------------------------------------------------------------------


def generate_named_phase_map(
    cleaned_label_image: np.ndarray,
    bse: np.ndarray,
    mineral_names: dict[int, str],
    figure_dir: str | Path,
) -> str:
    """BSE + cleaned phase map with mineral name legend.

    Parameters
    ----------
    cleaned_label_image : np.ndarray
        (H, W) int32 cleaned label image. -1 = background.
    bse : np.ndarray
        (H, W) float32 BSE image.
    mineral_names : dict[int, str]
        Mapping of cluster label (int) -> mineral name (str).
    figure_dir : str or Path
        Output directory for the saved figure.

    Returns
    -------
    str
        Path to the saved ``named_phase_map.png``.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    n_clusters = max(mineral_names.keys()) + 1 if mineral_names else 1

    # Build colormap: -1 -> gray (background), 0..n-1 -> tab20 colors
    base_colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
    cmap_colors = [[0.85, 0.85, 0.85, 1.0]]  # gray for -1 / background
    for i in range(n_clusters):
        cmap_colors.append(list(base_colors[i % 20]))
    cmap = ListedColormap(cmap_colors)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: BSE reference
    axes[0].imshow(bse, cmap="gray", interpolation="none")
    axes[0].set_title("BSE Reference")
    axes[0].axis("off")

    # Panel 2: Named phase map
    display = cleaned_label_image + 1  # shift so -1 -> 0 (gray)
    axes[1].imshow(display, cmap=cmap, vmin=0, vmax=n_clusters, interpolation="none")
    axes[1].set_title("Mineral Phase Map")
    axes[1].axis("off")

    # Legend with mineral names
    legend_patches = []
    for label in sorted(mineral_names.keys()):
        name = mineral_names[label]
        color = base_colors[label % 20]
        legend_patches.append(Patch(facecolor=color, edgecolor="black", label=name))
    legend_patches.append(
        Patch(facecolor=[0.85, 0.85, 0.85], edgecolor="black", label="Background")
    )

    axes[1].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
        frameon=True,
        framealpha=0.9,
    )

    plt.suptitle("Mineral Phase Identification", fontsize=14)
    plt.tight_layout()
    out_path = str(figure_dir / "named_phase_map.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved named phase map: %s", out_path)
    return out_path
