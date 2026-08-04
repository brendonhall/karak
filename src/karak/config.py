"""Pipeline configuration models and serialization utilities.

All pipeline parameters are captured in a Pydantic model hierarchy
so they can be serialized to YAML for reproducibility and embedded
in HDF5 attributes for provenance tracking.

v1.1 changes:
- DownsampleConfig replaces CropConfig (no ROI crop)
- NormalizeConfig replaces ZeroReplacementConfig + transform field
- Z-score normalization instead of CLR/ILR
- scikit-bio no longer required
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------

class DownsampleConfig(BaseModel):
    """Image downsampling and edge-trimming parameters."""

    header_trim_px: int = Field(
        default=100,
        description="Pixels to trim from top of each image (scale bar region)",
    )
    bottom_trim_px: int = Field(
        default=0,
        description="Pixels to trim from bottom of each image (annotation strip)",
    )
    left_trim_px: int = Field(
        default=0,
        description="Pixels to trim from left edge of each image",
    )
    right_trim_px: int = Field(
        default=0,
        description="Pixels to trim from right edge of each image (annotation/colorbar strip)",
    )
    downsample_factor: int = Field(
        default=2,
        description="Factor to downsample all images (BSE and EDS alike)",
    )


class LoaderConfig(BaseModel):
    """File discovery, filename parsing, and colormap inversion parameters.

    Lets karak ingest datasets that differ from the original TIMA convention
    (jet-colormapped PNGs with ``{sample}-{element}.png`` naming).
    """

    file_glob: str = Field(
        default="*.png",
        description="Glob pattern (relative to input_dir) for element map files",
    )
    filename_pattern: Optional[str] = Field(
        default=None,
        description=(
            "Filename pattern with ``{element}`` placeholder (and optional "
            "``{sample}`` wildcard) used to extract element names from "
            "matched files. Example: 'Map_NWA-5218_eds_{element}.bmp'. "
            "If None, the legacy heuristic is used: split basename on '-' "
            "and join parts[2:] (matches the original TIMA convention)."
        ),
    )
    bse_filename: Optional[str] = Field(
        default=None,
        description=(
            "Exact filename (relative to input_dir) of the BSE/SEM grayscale "
            "channel, used when the BSE file does not match ``file_glob``. "
            "If None, the BSE channel is found within the main glob by "
            "matching the parsed element name to ``bse_channel``."
        ),
    )
    colormap: str = Field(
        default="cmap:jet",
        description=(
            "Colormap specification for inverting false-color element maps. "
            "'cmap:NAME' looks up a matplotlib colormap by name (e.g. "
            "'cmap:jet', 'cmap:gist_ncar'). 'lut:PATH' loads a (N, 3) uint8 "
            "LUT from an .npy file (path is resolved relative to input_dir "
            "if not absolute). Plain 'jet' is accepted as shorthand for "
            "'cmap:jet' for backward compatibility."
        ),
    )


class MaskConfig(BaseModel):
    """Background / epoxy masking parameters."""

    min_object_size: int = Field(
        default=100,
        description="Remove connected components smaller than this from the mask",
    )
    valid_mask_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a napari shapes CSV defining the sample boundary polygon. "
            "Pixels outside this polygon are excluded before any algorithmic masking. "
            "Coordinates are in the original (pre-downsample) image space."
        ),
    )


class NormalizeConfig(BaseModel):
    """Normalization parameters."""

    method: Literal["zscore"] = Field(
        default="zscore",
        description="Normalization method (zscore: per-channel zero-mean unit-variance)",
    )


class DenoiseConfig(BaseModel):
    """Edge-aware denoising parameters."""

    method: Literal["bilateral", "anisotropic_diffusion"] = "bilateral"
    # Bilateral filter params
    sigma_color: Optional[float] = Field(
        default=None,
        description="Bilateral filter color sigma (None = auto from data range)",
    )
    sigma_spatial: float = Field(
        default=1.0,
        description="Bilateral filter spatial sigma",
    )
    # Anisotropic diffusion params
    niter: int = Field(default=10, description="Number of diffusion iterations")
    kappa: float = Field(
        default=50, description="Conductance coefficient for diffusion"
    )
    gamma: float = Field(default=0.1, description="Diffusion speed (0-0.25 stable)")
    option: int = Field(
        default=2,
        description="Perona-Malik option (1=favours high contrast, 2=wide regions)",
    )


class PCAConfig(BaseModel):
    """PCA dimensionality reduction parameters."""

    n_components: int | None = Field(
        default=None,
        description=(
            "Number of PCA components to retain. None = keep all, "
            "then researcher picks cutoff from scree plot at checkpoint."
        ),
    )
    subsample_fraction: float | None = Field(
        default=None,
        description=(
            "Fraction of mineral pixels to subsample for PCA fitting. "
            "None = use all mineral pixels. Set to e.g. 0.2 if memory is tight."
        ),
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class HDBSCANConfig(BaseModel):
    """HDBSCAN clustering parameters."""

    min_cluster_size: int = Field(
        default=1000,
        description="Minimum cluster size for HDBSCAN (in number of pixels)",
    )
    min_samples: int | None = Field(
        default=None,
        description="HDBSCAN min_samples. None = defaults to min_cluster_size.",
    )
    subsample_n: int | None = Field(
        default=None,
        description=(
            "Max number of mineral pixels to use for HDBSCAN fitting. "
            "None = use all. Set to e.g. 500000 for large images — "
            "remaining pixels are assigned via approximate_predict."
        ),
    )
    noise_reassign_k: int = Field(
        default=5,
        description="Number of neighbors for kNN reassignment of noise pixels",
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class TiledConfig(BaseModel):
    """Tiled progressive HDBSCAN parameters."""

    tile_size: int = Field(
        default=512,
        description="Tile side length in pixels for spatial tiling",
    )
    merge_threshold: float = Field(
        default=0.92,
        description="Cosine similarity threshold for matching tile clusters to phase registry",
    )
    min_tile_pixels: int | None = Field(
        default=None,
        description=(
            "Minimum mineral pixels for a tile to be processed. "
            "None = 2 * hdbscan.min_cluster_size."
        ),
    )
    min_clusters_per_tile: int = Field(
        default=3,
        description=(
            "Minimum HDBSCAN clusters for a tile to be trusted. "
            "Tiles with fewer clusters are deferred to the final k-NN pass, "
            "avoiding single-class tile artifacts."
        ),
    )


class RarePhaseConfig(BaseModel):
    """Pass 2 rare-phase reclustering parameters."""

    enabled: bool = Field(
        default=False,
        description="Enable two-pass workflow: Pass 1 major phases, Pass 2 rare phases.",
    )
    min_cluster_size: int = Field(
        default=50,
        description=(
            "Minimum cluster size for rare-phase HDBSCAN (Pass 2). "
            "Typically much smaller than the main HDBSCAN min_cluster_size."
        ),
    )
    min_samples: int | None = Field(
        default=None,
        description="HDBSCAN min_samples for Pass 2. None = defaults to min_cluster_size.",
    )
    subsample_n: int | None = Field(
        default=500_000,
        description=(
            "Max unassigned pixels to fit HDBSCAN on in Pass 2. "
            "Remaining pixels assigned via approximate_predict. "
            "None = use all (may OOM on large images)."
        ),
    )
    merge_threshold: float | None = Field(
        default=None,
        description=(
            "Cosine similarity threshold for matching rare clusters to existing registry. "
            "None = use the same threshold as tiled.merge_threshold."
        ),
    )


class OlivineExtractionConfig(BaseModel):
    """Threshold-based olivine extraction from pyroxene cluster."""

    enabled: bool = Field(
        default=False,
        description="Extract ferroan olivine from the target phase using Fe/Ca thresholds.",
    )
    fe_threshold: float = Field(
        default=0.6,
        description="Minimum denoised Fe-K intensity for olivine pixels.",
    )
    ca_threshold: float = Field(
        default=0.10,
        description="Maximum denoised Ca intensity for olivine pixels.",
    )


class GMMSplitConfig(BaseModel):
    """GMM-based pyroxene split into pigeonite and augite."""

    enabled: bool = Field(
        default=False,
        description="Split target phase into two sub-phases using Gaussian Mixture Model.",
    )
    n_components: int = Field(
        default=2,
        description="Number of GMM components (typically 2 for pigeonite/augite).",
    )
    features: list[str] = Field(
        default=["Ca", "Mg", "Fe-K", "BSE"],
        description=(
            "Feature channels for GMM. Use element names from the cube plus "
            "'BSE' for backscatter electron intensity. A 'Ca/(Ca+Mg)' ratio "
            "feature is automatically added if both Ca and Mg are present."
        ),
    )
    bse_weight: float = Field(
        default=1.0,
        description="Weight multiplier for BSE feature relative to elemental features.",
    )
    subsample_n: int | None = Field(
        default=500_000,
        description=(
            "Max pixels to fit GMM on. Remaining assigned via predict. "
            "None = use all pixels."
        ),
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility.")


class RefinementConfig(BaseModel):
    """Post-clustering phase refinement parameters.

    Applied after kNN noise reassignment to split composite phases
    (e.g., pyroxene → olivine + pigeonite + augite).
    """

    enabled: bool = Field(
        default=False,
        description="Enable post-clustering refinement step.",
    )
    target_phase: int = Field(
        default=2,
        description="Cluster label of the phase to refine (e.g., 2 for pyroxene).",
    )
    olivine: OlivineExtractionConfig = Field(default_factory=OlivineExtractionConfig)
    gmm_split: GMMSplitConfig = Field(default_factory=GMMSplitConfig)


class ClusterConfig(BaseModel):
    """Clustering pipeline parameters."""

    strategy: Literal["global", "tiled"] = Field(
        default="global",
        description="Clustering strategy: 'global' (single HDBSCAN) or 'tiled' (per-tile HDBSCAN with phase registry)",
    )
    pca: PCAConfig = Field(default_factory=PCAConfig)
    hdbscan: HDBSCANConfig = Field(default_factory=HDBSCANConfig)
    tiled: TiledConfig = Field(default_factory=TiledConfig)
    rare_phase: RarePhaseConfig = Field(default_factory=RarePhaseConfig)
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)


# ---------------------------------------------------------------------------
# Top-level pipeline configuration
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    """Complete pipeline configuration for SEM-EDS mineral mapping."""

    # Paths
    input_dir: str = Field(
        default="../data",
        description="Directory containing raw PNG element maps",
    )
    hdf5_output: str = Field(
        default="../data/eds_pipeline.h5",
        description="Path for the output HDF5 file",
    )
    figure_dir: str = Field(
        default="../data/figures",
        description="Directory for QC diagnostic figures",
    )

    # Element handling
    exclude_elements: list[str] = Field(
        default=["Fe-L"],
        description="Element channels to exclude from the compositional cube",
    )
    bse_channel: str = Field(
        default="SEM",
        description="Name of the BSE / SEM channel (kept separate from cube)",
    )

    # Sub-configs
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    downsample: DownsampleConfig = Field(default_factory=DownsampleConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    denoise: DenoiseConfig = Field(default_factory=DenoiseConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def save_config(config: PipelineConfig, path: str | Path) -> None:
    """Serialize a PipelineConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    with open(path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a PipelineConfig from a YAML file.

    Relative paths in the config (``input_dir``, ``hdf5_output``,
    ``figure_dir``, ``mask.valid_mask_path``) are resolved against the
    config file's directory, so a config can travel with its data
    directory regardless of the caller's working directory.
    """
    path = Path(path)
    with open(path) as fh:
        data = yaml.safe_load(fh)

    base = path.resolve().parent

    def _resolve(p: str | None) -> str | None:
        if p is None:
            return None
        candidate = Path(p).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        return str((base / candidate).resolve())

    for key in ("input_dir", "hdf5_output", "figure_dir"):
        if key in data:
            data[key] = _resolve(data[key])
    if isinstance(data.get("mask"), dict) and "valid_mask_path" in data["mask"]:
        data["mask"]["valid_mask_path"] = _resolve(data["mask"]["valid_mask_path"])

    return PipelineConfig(**data)


def get_software_versions() -> dict[str, str]:
    """Return a dict of key library versions for provenance tracking."""
    libs = [
        "numpy",
        "scikit-image",
        "scikit-learn",
        "h5py",
        "hdbscan",
        "medpy",
        "pydantic",
        "imageio",
        "pyyaml",
        "scipy",
    ]
    versions: dict[str, str] = {}
    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = "not installed"
    return versions
