"""Pipeline runner with Rich TUI, stage checkpointing, and resume logic.

Drives the same processing stages as ``01_compositional_foundation.ipynb``
but as a non-interactive script with intermediate HDF5 saves so the run can
be resumed after a crash.

Usage::

    karak                          # run all stages
    karak --test-mode              # 4 elements, 4x downsample
    karak --from-stage denoise     # resume from denoise
    karak --no-qc                  # skip QC figure generation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from karak.config import (
    ClusterConfig,
    MaskConfig,
    PipelineConfig,
    load_config,
    save_config,
)
from karak.io.loaders import build_compositional_cube, load_element_maps
from karak.io.masks import (
    compute_mask_statistics,
    create_mineral_mask,
    load_valid_mask,
)
from karak.io.storage import (
    create_pipeline_hdf5,
    get_completed_stages,
    load_bse,
    load_cluster_data,
    load_denoised_data,
    load_masks,
    load_normalized_data,
    load_raw_data,
    mark_stage_complete,
    save_bse,
    save_cluster_data,
    save_denoised_data,
    save_mask,
    save_normalized_data,
    save_raw_data,
)
from karak.preprocessing.compositional import (
    validate_normalization,
    zscore_normalize,
)
from karak.preprocessing.denoise import denoise_cube

logger = logging.getLogger(__name__)

console = Console()

STAGES = ["load", "mask", "denoise", "normalize", "cluster"]

TEST_ELEMENTS = ["Fe-K", "Ca", "Mg", "Si"]
TEST_DOWNSAMPLE = 4

# Status symbols
_SYM_DONE = "[green]done[/green]"
_SYM_SKIP = "[yellow]skipped[/yellow]"
_SYM_NOT = "[dim]not started[/dim]"
_SYM_RUN = "[cyan]running[/cyan]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="karak",
        description="SEM-EDS mineral phase mapping pipeline with checkpoint/resume.",
    )
    p.add_argument(
        "-c", "--config",
        default=None,
        help="Path to pipeline_config.yaml (default: auto-detect in data/)",
    )
    p.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        help="Force re-run from this stage onward (loads prior data from HDF5).",
    )
    p.add_argument(
        "--no-qc",
        action="store_true",
        help="Skip QC figure generation.",
    )
    p.add_argument(
        "--test-mode",
        action="store_true",
        help="Run with 4 elements at 4x downsample (fast validation).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    """Format seconds as e.g. '1m 26s' or '8s'."""
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _fmt_bytes(n: int) -> str:
    """Format byte count as human-readable."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _resolve_config(config_arg: str | None) -> tuple[PipelineConfig, Path]:
    """Load or create a PipelineConfig, return (config, config_path).

    Relative paths inside the config are resolved relative to the config
    file's directory so that ``karak`` works from any working directory.
    """
    if config_arg and Path(config_arg).exists():
        cfg_path = Path(config_arg).resolve()
        cfg = load_config(cfg_path)
    else:
        # Try standard locations
        cfg_path = None
        for candidate in ("data/pipeline_config.yaml", "../data/pipeline_config.yaml"):
            p = Path(candidate)
            if p.exists():
                cfg_path = p.resolve()
                cfg = load_config(cfg_path)
                break

        if cfg_path is None:
            # Create default config
            cfg = PipelineConfig(
                mask=MaskConfig(valid_mask_path="data/mask/Valid_mask.csv"),
                exclude_elements=["Fe-L"],
            )
            cfg_path = Path("data/pipeline_config.yaml").resolve()
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            save_config(cfg, cfg_path)
            return cfg, cfg_path

    # Resolve relative paths in config relative to config file directory
    cfg_dir = cfg_path.parent

    def _resolve(p: str) -> str:
        pp = Path(p)
        if not pp.is_absolute():
            return str((cfg_dir / pp).resolve())
        return p

    cfg.input_dir = _resolve(cfg.input_dir)
    cfg.hdf5_output = _resolve(cfg.hdf5_output)
    cfg.figure_dir = _resolve(cfg.figure_dir)
    if cfg.mask.valid_mask_path:
        cfg.mask.valid_mask_path = _resolve(cfg.mask.valid_mask_path)

    return cfg, cfg_path


def _stage_status_table(
    stages_done: dict[str, str | None],
    current: str | None = None,
) -> Table:
    """Build a Rich table showing stage status."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("num", style="dim", width=3)
    table.add_column("stage", width=12)
    table.add_column("status", width=20)
    for i, stage in enumerate(STAGES, 1):
        if stage == current:
            status = _SYM_RUN
        elif stages_done.get(stage):
            status = _SYM_DONE
        else:
            status = _SYM_NOT
        table.add_row(f"{i}.", stage, status)
    return table


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------


def _run_load(
    cfg: PipelineConfig,
    ctx: dict,
    *,
    test_mode: bool,
    do_qc: bool,
) -> None:
    """Stage 1: Load element maps + BSE, build cube, save to HDF5."""
    h5 = cfg.hdf5_output

    include = TEST_ELEMENTS if test_mode else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading PNGs...", total=None)
        elements, bse, element_names = load_element_maps(
            cfg.input_dir,
            cfg.downsample,
            cfg.exclude_elements,
            cfg.bse_channel,
            include_elements=include,
        )
        cube = build_compositional_cube(elements, element_names)
        progress.update(task, completed=len(element_names), total=len(element_names))

    console.print(
        f"  Elements: {len(element_names)} channels, "
        f"BSE shape: {bse.shape}"
    )
    console.print(
        f"  Cube: {cube.shape} {cube.dtype} — "
        f"{_fmt_bytes(cube.nbytes)}"
    )

    # Create HDF5 if it doesn't exist or we're starting fresh
    if not Path(h5).exists():
        create_pipeline_hdf5(h5, cfg)

    console.print("  Saving to HDF5...", end=" ")
    save_raw_data(h5, elements, element_names)
    save_bse(h5, bse, bse.shape, cfg.downsample.downsample_factor)
    console.print("done")

    # Store in context
    ctx["elements"] = elements
    ctx["bse"] = bse
    ctx["element_names"] = element_names
    ctx["cube"] = cube

    # Free individual element arrays (cube keeps the data)
    del elements
    ctx.pop("elements", None)


def _run_mask(
    cfg: PipelineConfig,
    ctx: dict,
    *,
    test_mode: bool,
    do_qc: bool,
) -> None:
    """Stage 2: Create masks, save to HDF5, optionally generate QC."""
    h5 = cfg.hdf5_output
    cube = ctx["cube"]
    bse = ctx["bse"]

    valid_mask: np.ndarray | None = None
    if cfg.mask.valid_mask_path:
        console.print("  Loading valid mask polygon...", end=" ")
        valid_mask = load_valid_mask(
            cfg.mask.valid_mask_path,
            bse.shape,
            cfg.downsample.downsample_factor,
            cfg.downsample.header_trim_px,
        )
        console.print("done")
    else:
        console.print("  [yellow]No valid_mask_path in config — skipping valid mask[/yellow]")

    console.print("  Creating mineral mask...", end=" ")
    mineral_mask = create_mineral_mask(
        cube,
        valid_mask=valid_mask,
        min_object_size=cfg.mask.min_object_size,
    )
    mask_stats = compute_mask_statistics(mineral_mask, valid_mask=valid_mask)
    console.print("done")

    console.print(
        f"  Coverage: {mask_stats['coverage_pct']:.1f}% of image, "
        f"{mask_stats['coverage_of_valid_pct']:.1f}% of valid region"
    )

    console.print("  Saving to HDF5...", end=" ")
    save_mask(h5, mineral_mask, valid_mask, mask_stats, cfg.mask)
    console.print("done")

    if do_qc:
        console.print("  QC figures...", end=" ")
        from karak.qc.figures import generate_mask_qc

        # Look for a TIMA reference phase map for QC comparison
        tima_dir = os.path.join(cfg.input_dir, "TIMA results")
        tima_path = None
        if os.path.isdir(tima_dir):
            import glob as _glob
            matches = _glob.glob(os.path.join(tima_dir, "*-Phases.png"))
            if matches:
                tima_path = matches[0]
        qc_path = generate_mask_qc(
            bse,
            mineral_mask,
            valid_mask,
            mask_stats,
            tima_path,
            cfg.downsample.downsample_factor,
            cfg.downsample.header_trim_px,
            cfg.figure_dir,
        )
        console.print(f"saved {Path(qc_path).name}")

    ctx["mineral_mask"] = mineral_mask
    ctx["valid_mask"] = valid_mask
    ctx["mask_stats"] = mask_stats


def _run_denoise(
    cfg: PipelineConfig,
    ctx: dict,
    *,
    test_mode: bool,
    do_qc: bool,
) -> None:
    """Stage 3: Denoise the raw cube, save to HDF5."""
    h5 = cfg.hdf5_output
    cube = ctx["cube"]
    mineral_mask = ctx["mineral_mask"]
    bse = ctx["bse"]
    element_names = ctx["element_names"]

    console.print(
        f"  Method: {cfg.denoise.method}, "
        f"cube {cube.shape}"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Denoising...", total=None)
        denoised = denoise_cube(cube, mineral_mask, cfg.denoise)
        progress.update(task, completed=1, total=1)

    console.print("  Saving to HDF5...", end=" ")
    save_denoised_data(h5, denoised, element_names, cfg.denoise)
    console.print("done")

    if do_qc:
        console.print("  QC figures...", end=" ")
        from karak.qc.figures import generate_denoise_qc

        qc_path = generate_denoise_qc(
            cube,
            denoised,
            bse,
            mineral_mask,
            element_names,
            cfg.denoise.method,
            cfg.figure_dir,
        )
        console.print(f"saved {Path(qc_path).name}")

    ctx["denoised"] = denoised

    # Free raw cube — no longer needed after denoising
    ctx.pop("cube", None)


def _run_normalize(
    cfg: PipelineConfig,
    ctx: dict,
    *,
    test_mode: bool,
    do_qc: bool,
) -> None:
    """Stage 4: Z-score normalize, validate, save to HDF5."""
    h5 = cfg.hdf5_output
    denoised = ctx["denoised"]
    mineral_mask = ctx["mineral_mask"]
    element_names = ctx["element_names"]

    console.print("  Computing z-scores...", end=" ")
    normalized, means, stds = zscore_normalize(denoised, mineral_mask)
    console.print("done")

    report = validate_normalization(normalized, mineral_mask, element_names)
    all_ok = all(info["ok"] for info in report.values())
    status = "[green]PASS[/green]" if all_ok else "[red]FAIL[/red]"
    console.print(f"  Validation: {status}")

    console.print("  Saving to HDF5...", end=" ")
    save_normalized_data(h5, normalized, means, stds, element_names, cfg.normalize)
    console.print("done")

    if do_qc:
        from karak.qc.figures import (
            generate_correlation_matrix,
            generate_zscore_histograms,
        )

        console.print("  QC figures...", end=" ")
        p1 = generate_zscore_histograms(
            normalized, mineral_mask, element_names, cfg.figure_dir,
        )
        p2 = generate_correlation_matrix(
            normalized, mineral_mask, element_names, cfg.figure_dir,
        )
        console.print(
            f"saved {Path(p1).name}, {Path(p2).name}"
        )

    # Free large arrays
    ctx.pop("denoised", None)
    ctx["normalized"] = normalized
    ctx["means"] = means
    ctx["stds"] = stds


def _run_cluster(
    cfg: PipelineConfig,
    ctx: dict,
    *,
    test_mode: bool,
    do_qc: bool,
) -> None:
    """Stage 5: PCA + HDBSCAN clustering, noise reassignment, save to HDF5."""
    from karak.clustering.hdbscan_cluster import (
        compute_cluster_stats,
        run_hdbscan,
    )
    from karak.clustering.noise_assign import assign_noise_pixels, labels_to_image
    from karak.clustering.pca import fit_pca, select_components

    h5 = cfg.hdf5_output
    normalized = ctx["normalized"]
    mineral_mask = ctx["mineral_mask"]
    bse = ctx["bse"]

    pca_cfg = cfg.cluster.pca
    hdb_cfg = cfg.cluster.hdbscan

    # --- PCA ---
    console.print(f"  PCA: n_components={pca_cfg.n_components or 'all'}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fitting PCA...", total=None)
        pca_model, pca_features, mineral_indices = fit_pca(
            normalized, mineral_mask, pca_cfg,
        )
        progress.update(task, completed=1, total=1)

    evr = pca_model.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    console.print(
        f"  PCA fitted: {len(evr)} components, "
        f"top-5 cumvar: {cumvar[min(4, len(cumvar)-1)]*100:.1f}%, "
        f"top-10 cumvar: {cumvar[min(9, len(cumvar)-1)]*100:.1f}%"
    )

    # Select components: use config value or auto-select at 95% variance
    if pca_cfg.n_components is not None:
        n_keep = pca_cfg.n_components
    else:
        # Auto: first component reaching 95% cumulative variance, minimum 5
        candidates = np.where(cumvar >= 0.95)[0]
        n_keep = int(candidates[0] + 1) if len(candidates) > 0 else len(evr)
        n_keep = max(n_keep, 5)
    n_keep = min(n_keep, pca_features.shape[1])
    pca_reduced = select_components(pca_features, n_keep)
    console.print(
        f"  Using {n_keep} components ({cumvar[n_keep-1]*100:.1f}% variance)"
    )

    strategy = cfg.cluster.strategy
    console.print(f"  Strategy: [bold]{strategy}[/bold]")

    if strategy == "tiled":
        # --- Tiled HDBSCAN ---
        from karak.clustering.tiling import (
            run_tiled_hdbscan,
            recluster_unassigned,
            final_knn_assign,
        )
        from karak.io.storage import save_tiled_metadata

        console.print(
            f"  HDBSCAN (tiled): tile_size={cfg.cluster.tiled.tile_size}, "
            f"min_cluster_size={hdb_cfg.min_cluster_size}, "
            f"merge_threshold={cfg.cluster.tiled.merge_threshold}"
        )

        # Load denoised cube for fingerprint computation
        console.print("  Loading denoised cube for fingerprints...", end=" ")
        denoised_cube, _ = load_denoised_data(h5)
        console.print("done")

        H, W = mineral_mask.shape

        rare_cfg = cfg.cluster.rare_phase
        do_two_pass = rare_cfg.enabled

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            tile_task = progress.add_task("Tiling...", total=None)
            pass2_task = progress.add_task(
                "Pass 2: rare phases", total=1, visible=False,
            )
            knn_task = progress.add_task(
                "Pass 3: k-NN unification", total=1, visible=False,
            )

            def _tile_progress(
                tile_id: int, n_tiles: int, result: object,
            ) -> None:
                if progress.tasks[tile_task].total is None:
                    progress.update(tile_task, total=n_tiles)
                tr = result
                progress.update(
                    tile_task,
                    completed=tile_id + 1,
                    description=(
                        f"Tile {tile_id + 1}/{n_tiles}: "
                        f"{tr.n_clusters} clusters, "
                        f"{tr.n_noise} noise, "
                        f"{len(tr.new_phases)} new phases"
                    ),
                )

            labels, cleaned_labels, probabilities, tile_results, phase_registry = (
                run_tiled_hdbscan(
                    pca_reduced, mineral_indices, (H, W),
                    denoised_cube, cfg.cluster,
                    progress_callback=_tile_progress,
                    skip_knn=do_two_pass,
                )
            )

            if do_two_pass:
                # Free denoised cube before Pass 2 HDBSCAN — it will be
                # reloaded on-demand for fingerprinting (saves ~1.9 GB)
                del denoised_cube

                # Pass 2: recluster unassigned pixels for rare phases
                progress.update(pass2_task, visible=True)
                labels, phase_registry, n_rare, n_still = recluster_unassigned(
                    pca_reduced, labels, h5,
                    mineral_indices, phase_registry, cfg.cluster,
                )
                progress.update(
                    pass2_task, completed=1,
                    description=(
                        f"Pass 2: {n_rare} rare phases, "
                        f"{n_still:,} still unassigned"
                    ),
                )

                # Pass 3: final kNN
                progress.update(knn_task, visible=True)
                cleaned_labels = final_knn_assign(
                    pca_reduced, labels, hdb_cfg.noise_reassign_k,
                )
                progress.update(knn_task, completed=1)
            else:
                progress.update(knn_task, visible=True, completed=1)

        # Free denoised cube (may already be freed in two-pass path)
        try:
            del denoised_cube
        except NameError:
            pass

        # Phase refinement (e.g., pyroxene -> olivine + pigeonite + augite)
        ref_cfg = cfg.cluster.refinement
        if ref_cfg.enabled:
            from karak.clustering.refinement import refine_phases

            console.print("  Phase refinement...", end=" ")
            ref_denoised, ref_elem_names = load_denoised_data(h5)
            cleaned_labels = refine_phases(
                cleaned_labels, ref_denoised, bse,
                mineral_indices, ref_elem_names, ref_cfg,
            )
            del ref_denoised
            n_refined = len(set(cleaned_labels.tolist()) - {-1})
            console.print(f"done ({n_refined} phases)")

        stats = compute_cluster_stats(cleaned_labels, probabilities)

        if do_two_pass:
            n_major = len([e for e in phase_registry if e.discovered_in_tile >= 0])
            n_rare = len(phase_registry) - n_major
            console.print(
                f"  Result: {stats['n_clusters']} total phases "
                f"({n_major} major + {n_rare} rare) from "
                f"{len(tile_results)} tiles"
            )
        else:
            console.print(
                f"  Result: {stats['n_clusters']} phases from "
                f"{len(tile_results)} tiles, "
                f"{int(np.sum(labels == -1))} unassigned pixels unified via k-NN"
            )

        # Save to HDF5
        console.print("  Saving to HDF5...", end=" ")
        save_cluster_data(
            h5, labels, cleaned_labels, probabilities,
            evr, mineral_indices, stats, n_keep, cfg.cluster,
        )
        save_tiled_metadata(h5, tile_results, phase_registry)
        console.print("done")

        # QC figures
        if do_qc:
            from karak.clustering.tiling import compute_tile_grid
            from karak.qc.figures import (
                generate_cluster_summary,
                generate_phase_map,
                generate_scree_plot,
            )
            from karak.qc.tiled_figures import (
                generate_phase_discovery_chart,
                generate_tile_grid_overlay,
            )

            console.print("  QC figures...", end=" ")

            raw_label_image = labels_to_image(labels, mineral_indices, (H, W))
            cleaned_label_image = labels_to_image(cleaned_labels, mineral_indices, (H, W))

            # Reconstruct tile specs for overlay
            min_tile_px = cfg.cluster.tiled.min_tile_pixels
            if min_tile_px is None:
                min_tile_px = 2 * hdb_cfg.min_cluster_size
            tiles = compute_tile_grid(
                mineral_indices, (H, W),
                cfg.cluster.tiled.tile_size, min_tile_px,
            )

            p1 = generate_scree_plot(evr, n_keep, cfg.figure_dir)
            p2 = generate_phase_map(
                raw_label_image, cleaned_label_image, bse, stats, cfg.figure_dir,
            )
            p3 = generate_cluster_summary(stats, cfg.figure_dir)
            p4 = generate_tile_grid_overlay(bse, tiles, tile_results, cfg.figure_dir)
            p5 = generate_phase_discovery_chart(tile_results, phase_registry, cfg.figure_dir)
            console.print(
                f"saved {Path(p1).name}, {Path(p2).name}, {Path(p3).name}, "
                f"{Path(p4).name}, {Path(p5).name}"
            )

    else:
        # --- Global HDBSCAN (existing path) ---
        console.print(
            f"  HDBSCAN: min_cluster_size={hdb_cfg.min_cluster_size}, "
            f"subsample_n={hdb_cfg.subsample_n or 'all'}"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Clustering...", total=None)
            labels, probabilities, clusterer = run_hdbscan(pca_reduced, hdb_cfg)
            progress.update(task, completed=1, total=1)

        stats = compute_cluster_stats(labels, probabilities)
        console.print(
            f"  Result: {stats['n_clusters']} clusters, "
            f"{stats['noise_pct']:.1f}% noise ({stats['n_noise']} pixels)"
        )

        # Flag unexpected cluster counts
        if stats["n_clusters"] < 3:
            console.print("  [yellow]WARNING: fewer than 3 clusters — consider adjusting min_cluster_size[/yellow]")
        elif stats["n_clusters"] > 15:
            console.print("  [yellow]WARNING: more than 15 clusters — consider increasing min_cluster_size[/yellow]")

        if stats["noise_pct"] > 10:
            console.print(f"  [yellow]WARNING: noise fraction {stats['noise_pct']:.1f}% exceeds 10% target[/yellow]")

        # Noise reassignment
        console.print(f"  Reassigning noise via {hdb_cfg.noise_reassign_k}-NN...", end=" ")
        cleaned_labels = assign_noise_pixels(
            pca_reduced, labels, k=hdb_cfg.noise_reassign_k,
        )
        console.print("done")

        # Phase refinement (e.g., pyroxene -> olivine + pigeonite + augite)
        ref_cfg = cfg.cluster.refinement
        if ref_cfg.enabled:
            from karak.clustering.refinement import refine_phases

            console.print("  Phase refinement...", end=" ")
            ref_denoised, ref_elem_names = load_denoised_data(h5)
            cleaned_labels = refine_phases(
                cleaned_labels, ref_denoised, bse,
                mineral_indices, ref_elem_names, ref_cfg,
            )
            del ref_denoised
            n_refined = len(set(cleaned_labels.tolist()) - {-1})
            console.print(f"done ({n_refined} phases)")

        stats = compute_cluster_stats(cleaned_labels, probabilities)

        # Save to HDF5
        console.print("  Saving to HDF5...", end=" ")
        save_cluster_data(
            h5, labels, cleaned_labels, probabilities,
            evr, mineral_indices, stats, n_keep, cfg.cluster,
        )
        console.print("done")

        # QC figures
        if do_qc:
            from karak.qc.figures import (
                generate_cluster_summary,
                generate_phase_map,
                generate_scree_plot,
            )

            console.print("  QC figures...", end=" ")
            H, W = mineral_mask.shape
            raw_label_image = labels_to_image(labels, mineral_indices, (H, W))
            cleaned_label_image = labels_to_image(cleaned_labels, mineral_indices, (H, W))

            p1 = generate_scree_plot(evr, n_keep, cfg.figure_dir)
            p2 = generate_phase_map(
                raw_label_image, cleaned_label_image, bse, stats, cfg.figure_dir,
            )
            p3 = generate_cluster_summary(stats, cfg.figure_dir)
            console.print(
                f"saved {Path(p1).name}, {Path(p2).name}, {Path(p3).name}"
            )

    # Store in context for potential downstream use
    ctx["raw_labels"] = labels
    ctx["cleaned_labels"] = cleaned_labels
    ctx["cluster_stats"] = stats
    ctx["mineral_indices"] = mineral_indices

    # Free large arrays no longer needed
    ctx.pop("normalized", None)


# Stage dispatch table
_STAGE_FUNCS = {
    "load": _run_load,
    "mask": _run_mask,
    "denoise": _run_denoise,
    "normalize": _run_normalize,
    "cluster": _run_cluster,
}


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def _load_prior_stages(
    h5_path: str,
    up_to: str,
    ctx: dict,
) -> None:
    """Load outputs from completed stages into *ctx* for resuming."""
    stage_idx = STAGES.index(up_to)

    # Always need element_names and bse if resuming from mask or later
    if stage_idx >= 0:
        # load stage data
        elements_dict, element_names = load_raw_data(h5_path)
        bse = load_bse(h5_path)
        cube = build_compositional_cube(elements_dict, element_names)
        ctx["bse"] = bse
        ctx["element_names"] = element_names
        ctx["cube"] = cube
        console.print(f"  [dim]Loaded raw data from HDF5 ({len(element_names)} elements)[/dim]")

    if stage_idx >= 1:
        # mask stage data
        mineral_mask, valid_mask, mask_stats = load_masks(h5_path)
        ctx["mineral_mask"] = mineral_mask
        ctx["valid_mask"] = valid_mask
        ctx["mask_stats"] = mask_stats
        console.print(f"  [dim]Loaded masks from HDF5[/dim]")

    if stage_idx >= 2:
        # denoise stage data
        denoised, _ = load_denoised_data(h5_path)
        ctx["denoised"] = denoised
        # Don't need raw cube any more if we already have denoised
        ctx.pop("cube", None)
        console.print(f"  [dim]Loaded denoised cube from HDF5[/dim]")

    if stage_idx >= 3:
        # normalize stage data needed for clustering
        norm_cube, means, stds, element_names_norm = load_normalized_data(h5_path)
        ctx["normalized"] = norm_cube
        ctx["means"] = means
        ctx["stds"] = stds
        # Don't need denoised if we already have normalized
        ctx.pop("denoised", None)
        console.print(f"  [dim]Loaded normalized cube from HDF5[/dim]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point for the pipeline runner."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(name)s - %(message)s",
        force=True,
    )

    # Config
    cfg, cfg_path = _resolve_config(args.config)

    if args.test_mode:
        cfg.downsample.downsample_factor = TEST_DOWNSAMPLE

    # Ensure output dirs exist
    os.makedirs(cfg.figure_dir, exist_ok=True)
    Path(cfg.hdf5_output).parent.mkdir(parents=True, exist_ok=True)

    h5 = cfg.hdf5_output
    do_qc = not args.no_qc

    # Determine element count description for banner
    if args.test_mode:
        mode_desc = f"test ({len(TEST_ELEMENTS)} elements, {TEST_DOWNSAMPLE}x downsample)"
    else:
        mode_desc = f"full ({cfg.downsample.downsample_factor}x downsample)"

    # Banner
    console.print()
    console.print(
        Panel(
            f"  Config: {cfg_path}\n"
            f"  Output: {h5}\n"
            f"  Mode:   {mode_desc}",
            title="karak SEM-EDS Pipeline",
            border_style="blue",
        )
    )

    # Check completed stages
    stages_done = get_completed_stages(h5)

    # Determine which stages to run
    if args.from_stage:
        first_stage = args.from_stage
    else:
        # Auto-detect: first incomplete stage
        first_stage = None
        for s in STAGES:
            if not stages_done.get(s):
                first_stage = s
                break

    if first_stage is None:
        console.print("\n[green]All stages already complete.[/green]\n")
        console.print(_stage_status_table(stages_done))
        console.print()
        _print_summary(h5)
        return

    first_idx = STAGES.index(first_stage)
    stages_to_run = STAGES[first_idx:]

    # Show stage status
    console.print("\nStage Status:")
    console.print(_stage_status_table(stages_done))
    console.print(
        f"\nRunning {len(stages_to_run)} stage(s) from [bold]{first_stage}[/bold]...\n"
    )

    # Load prior data if resuming mid-pipeline
    ctx: dict = {}
    if first_idx > 0:
        console.print("[dim]Loading prior stage data from HDF5...[/dim]")
        _load_prior_stages(h5, STAGES[first_idx - 1], ctx)
        console.print()

    # Run stages
    stage_times: dict[str, float] = {}
    for stage in stages_to_run:
        stage_num = STAGES.index(stage) + 1
        console.rule(f"Stage {stage_num}/{len(STAGES)}: {stage.title()}")

        t0 = time.time()
        try:
            _STAGE_FUNCS[stage](
                cfg, ctx, test_mode=args.test_mode, do_qc=do_qc,
            )
        except Exception:
            console.print(f"[red bold]Stage {stage} failed![/red bold]")
            console.print_exception()
            console.print(
                f"\n[yellow]Prior stages are saved. Resume with:[/yellow]\n"
                f"  karak --from-stage {stage}"
            )
            sys.exit(1)

        elapsed = time.time() - t0
        stage_times[stage] = elapsed
        mark_stage_complete(h5, stage)
        console.print(f"[green]  \u2713 {stage} complete [{_fmt_time(elapsed)}][/green]\n")

    # Save config alongside HDF5
    save_config(cfg, cfg_path)

    # Final summary
    _print_summary(h5, stage_times)


def _print_summary(
    h5_path: str,
    stage_times: dict[str, float] | None = None,
) -> None:
    """Print the final pipeline summary table."""
    console.rule("Pipeline Complete")
    console.print()

    stages_done = get_completed_stages(h5_path)

    table = Table(title=None, show_header=True)
    table.add_column("Stage", style="bold", width=12)
    table.add_column("Status", width=10)
    table.add_column("Time", justify="right", width=10)

    total_time = 0.0
    for stage in STAGES:
        if stages_done.get(stage):
            status = "[green]complete[/green]"
        else:
            status = "[dim]—[/dim]"

        if stage_times and stage in stage_times:
            t = stage_times[stage]
            total_time += t
            time_str = _fmt_time(t)
        else:
            time_str = ""

        table.add_row(stage, status, time_str)

    if stage_times:
        table.add_section()
        table.add_row("Total", "", _fmt_time(total_time))

    console.print(table)

    if Path(h5_path).exists():
        size = Path(h5_path).stat().st_size
        console.print(f"\nHDF5: {h5_path} ({_fmt_bytes(size)})")

    console.print()


if __name__ == "__main__":
    main()
