"""Edge-aware denoising for CLR-transformed compositional cubes.

Implements bilateral filtering (scikit-image) and Perona-Malik anisotropic
diffusion (medpy) as channel-by-channel denoisers.  Both methods handle the
mineral mask boundary by filling masked pixels with the mineral-pixel mean
before denoising, then re-applying the mask after (see RESEARCH.md Pitfall 3).

The ``compare_denoisers`` function produces a multi-panel diagnostic figure
and quantitative metrics (RMSE, edge preservation) for choosing between the
two methods.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.restoration import denoise_bilateral

if TYPE_CHECKING:
    from karak.config import DenoiseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core denoisers
# ---------------------------------------------------------------------------


def bilateral_denoise_cube(
    cube: np.ndarray,
    mask: np.ndarray,
    sigma_color: float | None = None,
    sigma_spatial: float = 1.0,
) -> np.ndarray:
    """Apply bilateral filter independently to each channel.

    Before denoising each channel, masked (non-mineral) pixels are filled
    with the mean of mineral pixels to prevent mask-boundary artifacts.
    After denoising, the mask is re-applied (masked pixels set to 0).

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) CLR-transformed compositional cube.
    mask : np.ndarray
        (H, W) boolean mask, True = mineral pixel.
    sigma_color : float or None
        Radiometric similarity sigma.  ``None`` lets scikit-image
        auto-calculate from the image standard deviation.
    sigma_spatial : float
        Spatial distance sigma (default 1.0).

    Returns
    -------
    denoised : np.ndarray
        (H, W, C) denoised cube with masked pixels set to 0.
    """
    H, W, C = cube.shape
    denoised = np.zeros_like(cube)

    for i in range(C):
        channel = cube[:, :, i].copy()

        # Fill masked pixels with mineral-pixel mean (Pitfall 3)
        channel[~mask] = np.nanmean(channel[mask])

        denoised[:, :, i] = denoise_bilateral(
            channel,
            sigma_color=sigma_color,
            sigma_spatial=sigma_spatial,
        )

        logger.info("Bilateral denoise: channel %d/%d done", i + 1, C)

    # Re-apply mask
    denoised[~mask] = 0.0

    logger.info(
        "Bilateral denoise complete: shape %s, sigma_color=%s, sigma_spatial=%s",
        denoised.shape,
        sigma_color,
        sigma_spatial,
    )
    return denoised


def anisotropic_denoise_cube(
    cube: np.ndarray,
    mask: np.ndarray,
    niter: int = 10,
    kappa: float = 50,
    gamma: float = 0.1,
    option: int = 2,
) -> np.ndarray:
    """Apply Perona-Malik anisotropic diffusion independently per channel.

    Each channel is scaled to [0, 1] (using mineral-pixel min/max) before
    diffusion, then scaled back.  Masked pixels are filled with the mineral
    mean before denoising and zeroed after.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) CLR-transformed compositional cube.
    mask : np.ndarray
        (H, W) boolean mask, True = mineral pixel.
    niter : int
        Number of diffusion iterations (default 10).
    kappa : float
        Conductance coefficient (default 50).
    gamma : float
        Diffusion speed, must be <= 0.25 for stability (default 0.1).
    option : int
        Perona-Malik option: 1 = favours high contrast edges,
        2 = favours wide regions over smaller ones (default 2).

    Returns
    -------
    denoised : np.ndarray
        (H, W, C) denoised cube with masked pixels set to 0.
    """
    H, W, C = cube.shape
    denoised = np.zeros_like(cube)

    for i in range(C):
        channel = cube[:, :, i].copy()

        # Fill masked pixels with mineral-pixel mean
        channel[~mask] = np.nanmean(channel[mask])

        # Scale to [0, 1] using mineral-pixel range for medpy
        cmin = channel[mask].min()
        cmax = channel[mask].max()

        if cmax > cmin:
            channel_scaled = (channel - cmin) / (cmax - cmin)
        else:
            channel_scaled = np.zeros_like(channel)

        diffused = anisotropic_diffusion(
            channel_scaled,
            niter=niter,
            kappa=kappa,
            gamma=gamma,
            option=option,
        )

        # Scale back to original range
        if cmax > cmin:
            denoised[:, :, i] = diffused * (cmax - cmin) + cmin
        else:
            denoised[:, :, i] = channel

        logger.info("Anisotropic denoise: channel %d/%d done", i + 1, C)

    # Re-apply mask
    denoised[~mask] = 0.0

    logger.info(
        "Anisotropic denoise complete: shape %s, niter=%d, kappa=%.1f",
        denoised.shape,
        niter,
        kappa,
    )
    return denoised


# ---------------------------------------------------------------------------
# Comparison framework
# ---------------------------------------------------------------------------


def compare_denoisers(
    clr_cube: np.ndarray,
    mask: np.ndarray,
    bse: np.ndarray,
    element_names: list[str],
    denoise_config: DenoiseConfig,
    figure_dir: str,
    test_region: tuple[int, int, int, int] | None = None,
) -> dict:
    """Run both denoisers and produce a comparison figure with metrics.

    Parameters
    ----------
    clr_cube : np.ndarray
        (H, W, C) CLR-transformed compositional cube.
    mask : np.ndarray
        (H, W) boolean mineral mask.
    bse : np.ndarray
        (H, W) BSE image (for edge overlay).
    element_names : list[str]
        Ordered element names matching the C axis.
    denoise_config : DenoiseConfig
        Denoising configuration (parameters for both methods).
    figure_dir : str
        Directory to save the comparison figure.
    test_region : tuple or None
        (y0, y1, x0, x1) sub-region for faster comparison.
        If None, use the full image.

    Returns
    -------
    metrics : dict
        ``{method: {rmse_per_channel, edge_preservation, runtime_seconds}}``
    """
    import os

    import matplotlib.pyplot as plt
    from skimage.filters import sobel

    os.makedirs(figure_dir, exist_ok=True)

    # Crop to test region if specified
    if test_region is not None:
        y0, y1, x0, x1 = test_region
        clr_sub = clr_cube[y0:y1, x0:x1]
        mask_sub = mask[y0:y1, x0:x1]
        bse_sub = bse[y0:y1, x0:x1]
    else:
        clr_sub = clr_cube
        mask_sub = mask
        bse_sub = bse

    # Pick 3 high-variance channels for display
    mineral_data = clr_sub[mask_sub]  # (N, C)
    variances = np.var(mineral_data, axis=0)
    top3_idx = np.argsort(variances)[-3:][::-1]
    top3_names = [element_names[i] for i in top3_idx]
    logger.info("Top-3 variance channels for comparison: %s", top3_names)

    # Run bilateral
    t0 = time.time()
    bilateral_result = bilateral_denoise_cube(
        clr_sub,
        mask_sub,
        sigma_color=denoise_config.sigma_color,
        sigma_spatial=denoise_config.sigma_spatial,
    )
    bilateral_time = time.time() - t0

    # Run anisotropic
    t0 = time.time()
    aniso_result = anisotropic_denoise_cube(
        clr_sub,
        mask_sub,
        niter=denoise_config.niter,
        kappa=denoise_config.kappa,
        gamma=denoise_config.gamma,
        option=denoise_config.option,
    )
    aniso_time = time.time() - t0

    # Compute BSE edges for overlay
    bse_norm = (bse_sub - bse_sub.min()) / (bse_sub.max() - bse_sub.min() + 1e-10)
    bse_edges = sobel(bse_norm)
    edge_threshold = np.percentile(bse_edges[mask_sub], 90)
    edge_binary = bse_edges > edge_threshold

    # --- Comparison figure: 4 rows x 3 columns ---
    fig, axes = plt.subplots(4, 3, figsize=(14, 16))

    for col, ch_idx in enumerate(top3_idx):
        ch_name = element_names[ch_idx]

        # Determine consistent vmin/vmax from original mineral pixels
        ch_mineral = clr_sub[:, :, ch_idx][mask_sub]
        vmin = np.percentile(ch_mineral, 2)
        vmax = np.percentile(ch_mineral, 98)

        # Row 0: Original CLR
        ax = axes[0, col]
        im = ax.imshow(clr_sub[:, :, ch_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{ch_name} - Original CLR")
        ax.axis("off")

        # Row 1: Bilateral
        ax = axes[1, col]
        ax.imshow(bilateral_result[:, :, ch_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        # Overlay BSE edges as thin red contour
        ax.contour(edge_binary, levels=[0.5], colors="red", linewidths=0.5)
        ax.set_title(f"{ch_name} - Bilateral")
        ax.axis("off")

        # Row 2: Anisotropic
        ax = axes[2, col]
        ax.imshow(aniso_result[:, :, ch_idx], cmap="viridis", vmin=vmin, vmax=vmax)
        ax.contour(edge_binary, levels=[0.5], colors="red", linewidths=0.5)
        ax.set_title(f"{ch_name} - Anisotropic")
        ax.axis("off")

        # Row 3: BSE with edge overlay
        ax = axes[3, col]
        ax.imshow(bse_sub, cmap="gray")
        ax.contour(edge_binary, levels=[0.5], colors="cyan", linewidths=0.5)
        ax.set_title(f"BSE + edges ({ch_name} context)")
        ax.axis("off")

    plt.suptitle("Denoiser Comparison (red = BSE grain boundaries)", fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "denoise_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison figure to %s", fig_path)

    # --- Compute metrics ---
    metrics: dict = {}

    for name, result, runtime in [
        ("bilateral", bilateral_result, bilateral_time),
        ("anisotropic", aniso_result, aniso_time),
    ]:
        # Per-channel RMSE on mineral pixels
        orig_mineral = clr_sub[mask_sub]  # (N, C)
        den_mineral = result[mask_sub]  # (N, C)
        rmse = np.sqrt(np.mean((orig_mineral - den_mineral) ** 2, axis=0))

        # Edge preservation: correlation of gradient magnitudes at mineral pixels
        bse_grad = sobel(bse_norm)
        edge_corrs = []
        for ch_idx in range(clr_sub.shape[-1]):
            den_grad = sobel(result[:, :, ch_idx])
            # Correlation at mineral pixels
            bse_g = bse_grad[mask_sub]
            den_g = den_grad[mask_sub]
            if bse_g.std() > 0 and den_g.std() > 0:
                corr = np.corrcoef(bse_g, den_g)[0, 1]
            else:
                corr = 0.0
            edge_corrs.append(corr)

        metrics[name] = {
            "rmse_per_channel": rmse.tolist(),
            "rmse_mean": float(rmse.mean()),
            "edge_preservation": edge_corrs,
            "edge_preservation_mean": float(np.mean(edge_corrs)),
            "runtime_seconds": runtime,
        }

    logger.info(
        "Denoiser metrics - bilateral: RMSE=%.4f, edge=%.4f (%.1fs) | "
        "anisotropic: RMSE=%.4f, edge=%.4f (%.1fs)",
        metrics["bilateral"]["rmse_mean"],
        metrics["bilateral"]["edge_preservation_mean"],
        metrics["bilateral"]["runtime_seconds"],
        metrics["anisotropic"]["rmse_mean"],
        metrics["anisotropic"]["edge_preservation_mean"],
        metrics["anisotropic"]["runtime_seconds"],
    )

    return metrics


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------


def denoise_cube(
    cube: np.ndarray,
    mask: np.ndarray,
    config: DenoiseConfig,
) -> np.ndarray:
    """Dispatch to bilateral or anisotropic denoiser based on config.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) CLR-transformed compositional cube.
    mask : np.ndarray
        (H, W) boolean mineral mask.
    config : DenoiseConfig
        Denoising configuration with ``method`` field.

    Returns
    -------
    denoised : np.ndarray
        (H, W, C) denoised cube.
    """
    if config.method == "bilateral":
        return bilateral_denoise_cube(
            cube,
            mask,
            sigma_color=config.sigma_color,
            sigma_spatial=config.sigma_spatial,
        )
    elif config.method == "anisotropic_diffusion":
        return anisotropic_denoise_cube(
            cube,
            mask,
            niter=config.niter,
            kappa=config.kappa,
            gamma=config.gamma,
            option=config.option,
        )
    else:
        raise ValueError(
            f"Unknown denoise method: {config.method!r}. "
            "Use 'bilateral' or 'anisotropic_diffusion'."
        )
