"""Edge-aware denoising for raw [0, 1] element cubes.

Channel-by-channel denoisers: the bilateral filter (scikit-image for
``bilateral``, karak's own numpy reference and CuPy kernel in
``karak.preprocessing.bilateral`` for ``bilateral_sym`` and the joint
methods) and Perona-Malik anisotropic diffusion (medpy). All handle the
mineral mask boundary by filling masked pixels with the mineral-pixel mean
before denoising, then re-applying the mask after (see RESEARCH.md Pitfall
3).

The ``compare_denoisers`` function produces a multi-panel diagnostic figure
and quantitative metrics (RMSE, edge preservation) for choosing between the
methods.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable

import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.restoration import denoise_bilateral

if TYPE_CHECKING:
    from karak.core_params import DenoiseConfig

logger = logging.getLogger(__name__)

BILATERAL_METHODS = ("bilateral", "bilateral_sym",
                     "joint_bilateral_total", "joint_bilateral_bse")
JOINT_METHODS = ("joint_bilateral_total", "joint_bilateral_bse")


# ---------------------------------------------------------------------------
# Core denoisers
# ---------------------------------------------------------------------------


def _bilateral_channel(channel, mask, sigma_color, sigma_spatial,
                       method="bilateral", guide=None):
    """One channel of bilateral_denoise_cube. Pool-safe.

    ``bilateral`` is scikit-image's filter (the published baseline); the
    other methods use karak's reference implementation, with ``guide``
    (already mask-filled) as the range image for the joint methods.
    """
    channel = channel.copy()
    channel[~mask] = np.nanmean(channel[mask])
    if method == "bilateral":
        return denoise_bilateral(
            channel, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
        )
    from karak.preprocessing.bilateral import bilateral_numpy

    return bilateral_numpy(
        channel, channel if guide is None else guide,
        sigma_color=sigma_color, sigma_spatial=sigma_spatial,
    )


def _joint_guide(cube, mask, method, guide):
    """The mask-filled guide image for a joint method, or None. Runs on
    whatever array module ``cube`` lives on."""
    from karak.accel import xp as _xp

    if method not in JOINT_METHODS:
        return None
    if method == "joint_bilateral_total":
        guide = cube.sum(axis=-1)
    elif guide is None:
        from karak.errors import StageError
        raise StageError(
            "method='joint_bilateral_bse' needs the BSE image: connect the "
            "edge src.bse -> dn.bse (the denoise stage's optional 'bse' port)"
        )
    xp = _xp(cube)
    guide = xp.ascontiguousarray(xp.asarray(guide), dtype=xp.float32).copy()
    guide[~mask] = guide[mask].mean()
    return guide


def _anisotropic_channel(channel, mask, niter, kappa, gamma, option):
    """One channel of anisotropic_denoise_cube. Pool-safe."""
    channel = channel.copy()
    channel[~mask] = np.nanmean(channel[mask])
    cmin = channel[mask].min()
    cmax = channel[mask].max()
    if cmax > cmin:
        channel_scaled = (channel - cmin) / (cmax - cmin)
    else:
        return channel
    diffused = anisotropic_diffusion(
        channel_scaled, niter=niter, kappa=kappa, gamma=gamma, option=option,
    )
    return diffused * (cmax - cmin) + cmin


def _run_channels(fn, args, workers, on_channel):
    """Apply ``fn(*args[i])`` to every channel, serially or in a process
    pool, and return the results in channel order. ``on_channel(done,
    total, index)`` fires as each channel completes (completion order in
    the pool), so callers can report progress before all channels are in.
    """
    total = len(args)
    results = [None] * total
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed

        mp_context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp_context,
        ) as pool:
            futures = {pool.submit(fn, *a): i for i, a in enumerate(args)}
            for done, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                results[index] = future.result()
                if on_channel is not None:
                    on_channel(done, total, index)
    else:
        for index, a in enumerate(args):
            results[index] = fn(*a)
            if on_channel is not None:
                on_channel(index + 1, total, index)
    return results


def bilateral_denoise_cube(
    cube: np.ndarray,
    mask: np.ndarray,
    *,
    sigma_color: float | None,
    sigma_spatial: float,
    workers: int = 1,
    device: str = "cpu",
    on_channel: Callable[[int, int, int], None] | None = None,
    method: str = "bilateral",
    guide: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a bilateral filter independently to each channel.

    Masked (non-mineral) pixels are filled with the channel's mineral mean
    before filtering and zeroed after, so the epoxy zeros do not darken
    grain boundaries.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) raw [0, 1] element cube.
    mask : np.ndarray
        (H, W) boolean mask, True = mineral pixel.
    sigma_color : float or None
        Range sigma; None = the standard deviation of the filtered channel
        (single-image methods) or of the guide (joint methods).
    sigma_spatial : float
        Spatial sigma in pixels; the window is ``max(5, 2*ceil(3*sigma)+1)``.
    workers : int
        Number of parallel workers (default 1).
    device : str
        "cpu" or "cuda". On CUDA every method runs karak's CuPy kernel;
        ``bilateral`` reproduces scikit-image to float32 precision. CuPy
        inputs are accepted and the result is a CuPy array; host inputs are
        moved to the device. ``cpu`` with a CuPy input raises StageError.
    on_channel : callable, optional
        Called as ``on_channel(done, total, index)`` after each channel
        completes; with ``workers > 1`` in completion order.
    method : str
        ``bilateral`` (scikit-image, including its off-centre spatial
        table; see ``karak.preprocessing.bilateral``), ``bilateral_sym``
        (symmetric kernel), ``joint_bilateral_total`` (range weight from
        the sum of the channels) or ``joint_bilateral_bse`` (range weight
        from ``guide``).
    guide : np.ndarray, optional
        (H, W) BSE image for ``joint_bilateral_bse``.

    Returns
    -------
    denoised : np.ndarray
        (H, W, C) denoised cube with masked pixels set to 0.
    """
    if method not in BILATERAL_METHODS:
        raise ValueError(f"unknown bilateral method {method!r}")
    from karak.accel import is_device_array

    if device == "cpu" and (is_device_array(cube) or is_device_array(mask)):
        from karak.errors import StageError
        raise StageError(
            "device='cpu' received a device array; the executor places "
            "inputs on the stage's device, so run this stage with device='cuda'"
        )
    guide = _joint_guide(cube, mask, method, guide)
    H, W, C = cube.shape

    if device == "cuda":
        from karak.accel import get_array_module
        from karak.preprocessing.bilateral import bilateral_cupy

        cp = get_array_module(device)  # raises StageError without CUDA
        gpu_cube = cp.asarray(cube)  # no-op for a device array
        gpu_mask = cp.asarray(mask)
        gpu_guide = None if guide is None else cp.asarray(guide)
        out = cp.zeros_like(gpu_cube)
        for i in range(C):
            channel = gpu_cube[:, :, i].copy()
            channel[~gpu_mask] = cp.nanmean(channel[gpu_mask])
            out[:, :, i] = bilateral_cupy(
                channel, channel if gpu_guide is None else gpu_guide,
                sigma_color=sigma_color, sigma_spatial=sigma_spatial,
                exact_skimage=(method == "bilateral"),
            )
            if on_channel is not None:
                on_channel(i + 1, C, i)
        out[~gpu_mask] = 0.0
        logger.info(
            "Bilateral denoise complete (GPU, %s): shape %s, sigma_color=%s, "
            "sigma_spatial=%s", method, out.shape, sigma_color, sigma_spatial,
        )
        return out.astype(cube.dtype)  # stays on the device

    denoised = np.zeros_like(cube)
    args = [(cube[:, :, i].copy(), mask, sigma_color, sigma_spatial, method, guide)
            for i in range(C)]
    channels = _run_channels(_bilateral_channel, args, workers, on_channel)
    for i, ch in enumerate(channels):
        denoised[:, :, i] = ch
        logger.info("Bilateral denoise: channel %d/%d done", i + 1, C)

    # Re-apply mask
    denoised[~mask] = 0.0

    logger.info(
        "Bilateral denoise complete (%s): shape %s, sigma_color=%s, sigma_spatial=%s",
        method, denoised.shape, sigma_color, sigma_spatial,
    )
    return denoised


def anisotropic_denoise_cube(
    cube: np.ndarray,
    mask: np.ndarray,
    *,
    niter: int,
    kappa: float,
    gamma: float,
    option: int,
    workers: int = 1,
    on_channel: Callable[[int, int, int], None] | None = None,
) -> np.ndarray:
    """Apply Perona-Malik anisotropic diffusion independently per channel.

    Each channel is scaled to [0, 1] (using mineral-pixel min/max) before
    diffusion, then scaled back.  Masked pixels are filled with the mineral
    mean before denoising and zeroed after.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) raw [0, 1] element cube.
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
    workers : int
        Number of parallel workers (default 1).
    on_channel : callable, optional
        Called as ``on_channel(done, total, index)`` after each channel
        completes; with ``workers > 1`` in completion order.

    Returns
    -------
    denoised : np.ndarray
        (H, W, C) denoised cube with masked pixels set to 0.
    """
    H, W, C = cube.shape
    denoised = np.zeros_like(cube)

    args = [(cube[:, :, i].copy(), mask, niter, kappa, gamma, option)
            for i in range(C)]
    channels = _run_channels(_anisotropic_channel, args, workers, on_channel)
    for i, ch in enumerate(channels):
        denoised[:, :, i] = ch
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
        (H, W, C) raw [0, 1] element cube.
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
    workers: int = 1,
    device: str = "cpu",
    on_channel: Callable[[int, int, int], None] | None = None,
    guide: np.ndarray | None = None,
) -> np.ndarray:
    """Dispatch to a bilateral or the anisotropic denoiser based on config.

    Parameters
    ----------
    cube : np.ndarray
        (H, W, C) raw [0, 1] element cube.
    mask : np.ndarray
        (H, W) boolean mineral mask.
    config : DenoiseConfig
        Denoising configuration with ``method`` field.
    workers : int
        Number of parallel workers (default 1).
    device : str
        "cpu" or "cuda" (default "cpu"); with cuda the result is a CuPy
        array and CuPy inputs are accepted.
    on_channel : callable, optional
        Called as ``on_channel(done, total, index)`` after each channel
        completes; with ``workers > 1`` in completion order.
    guide : np.ndarray, optional
        (H, W) BSE image, required by ``method="joint_bilateral_bse"``.

    Returns
    -------
    denoised : np.ndarray
        (H, W, C) denoised cube.
    """
    if device == "cuda" and config.method == "anisotropic_diffusion":
        from karak.errors import StageError
        raise StageError(
            "device='cuda' supports only method='bilateral'; "
            "anisotropic diffusion has no GPU path"
        )

    if config.method in BILATERAL_METHODS:
        return bilateral_denoise_cube(
            cube,
            mask,
            sigma_color=config.sigma_color,
            sigma_spatial=config.sigma_spatial,
            workers=workers,
            device=device,
            on_channel=on_channel,
            method=config.method,
            guide=guide,
        )
    elif config.method == "anisotropic_diffusion":
        return anisotropic_denoise_cube(
            cube,
            mask,
            niter=config.niter,
            kappa=config.kappa,
            gamma=config.gamma,
            option=config.option,
            workers=workers,
            on_channel=on_channel,
        )
    else:
        raise ValueError(
            f"Unknown denoise method: {config.method!r}. "
            f"Use one of {BILATERAL_METHODS} or 'anisotropic_diffusion'."
        )
