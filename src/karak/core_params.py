"""Argument bundles for the numeric core.

Plain frozen dataclasses with no default values: a stage builds each one
from its (complete) flow params, and a missing field is a TypeError rather
than a value quietly taken from code. ``None`` fields mean "automatic" and
are documented on the numeric function that reads them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class DownsampleConfig:
    """Edge trims (original-image pixels) and the block-mean factor."""

    header_trim_px: int
    bottom_trim_px: int
    left_trim_px: int
    right_trim_px: int
    downsample_factor: int


@dataclass(frozen=True, kw_only=True)
class LoaderConfig:
    """File discovery, filename parsing, and colormap inversion."""

    file_glob: str
    filename_pattern: str | None   # None = legacy TIMA name heuristic
    bse_filename: str | None       # None = find BSE within file_glob
    colormap: str


@dataclass(frozen=True, kw_only=True)
class DenoiseConfig:
    method: str                    # bilateral | bilateral_sym | joint_bilateral_total | joint_bilateral_bse | anisotropic_diffusion
    sigma_color: float | None      # None = derive from the data range
    sigma_spatial: float
    niter: int
    kappa: float
    gamma: float
    option: int


@dataclass(frozen=True, kw_only=True)
class PCAConfig:
    n_components: int | None       # None = keep all
    subsample_fraction: float | None  # None = fit on every mineral pixel
    random_state: int


@dataclass(frozen=True, kw_only=True)
class HDBSCANConfig:
    min_cluster_size: int
    min_samples: int | None        # None = min_cluster_size
    subsample_n: int | None        # None = fit on every pixel
    random_state: int


@dataclass(frozen=True, kw_only=True)
class TiledConfig:
    tile_size: int
    merge_threshold: float
    min_tile_pixels: int | None    # None = 2 * min_cluster_size
    min_clusters_per_tile: int
    accumulate: str                # float64 | float32 tile-fingerprint sums


@dataclass(frozen=True, kw_only=True)
class RarePhaseConfig:
    """Pass 2: HDBSCAN on the pixels Pass 1 left unassigned."""

    min_cluster_size: int
    min_samples: int | None        # None = min_cluster_size
    subsample_n: int | None        # None = fit on every unassigned pixel
    merge_threshold: float
    accumulate: str                # float64 | float32 rare-cluster fingerprint sums


@dataclass(frozen=True, kw_only=True)
class OlivineExtractionConfig:
    enabled: bool
    fe_threshold: float
    ca_threshold: float


@dataclass(frozen=True, kw_only=True)
class GMMSplitConfig:
    enabled: bool
    n_components: int
    features: tuple[str, ...]      # element names plus "BSE"
    bse_weight: float
    subsample_n: int | None        # None = fit on every pixel
    random_state: int


@dataclass(frozen=True, kw_only=True)
class RefinementConfig:
    enabled: bool
    target_phase: int
    olivine: OlivineExtractionConfig
    gmm_split: GMMSplitConfig
